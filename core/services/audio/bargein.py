"""小智通道 SPEAKING 期打断监视器（AEC + VAD）。

设计依据：bridge-aec/design.md（一期）。灰度开关 XIAOZHI_BARGEIN 在
core/xiaozhi.py 里解析（off / observe / on）；本模块只在开关非 off 时
才会被装配，关闭时运行行为与现行版一致。

分层：
- Resampler24kTo16k：24k→16k 流式多相重采样。与
  scipy.signal.resample_poly 使用同一 Kaiser 窗 FIR 设计（61 taps、
  beta=5.0），纯 numpy 实现，运行时不依赖 scipy。
- ReferenceTap：参考信号环形缓冲。AudioCodec.write_audio 解码出的
  TTS PCM 经此入队（带绝对样本计数），监视器按时延对齐偏移读取。
- BargeInCore：纯逻辑核心——时延标定、AEC 消回声、VAD 打断判定、
  消回声预缓冲。不碰线程与设备流，vad_fn 由外部注入，可离线单测。
- BargeInMonitor：生产装配层——独立监视输入流 + 轮询线程 + 专用
  Silero VAD 会话（不与唤醒会话共用的 VAD 实例/状态）。

判定原则（design.md §4/§6）：标定成功后才启用判定；触发需要
「AEC 输出上连续 ≥250ms 高置信语音」且「该段电平高于残差回声底噪
估计 +12dB（且不低于 -50 dBFS）」，避免纯回声误触发。
预热只覆盖标定后实际送进 AEC 的有声参考的前 ARM_SAMPLES（处理
坐标）。标定前麦克风最多积压 4s，且这段音频的样本时长不等于
墙钟；预热终点若取「最新喂入样本 + ARM」，整段积压都落在预热
区内，积压尾部的真人首句会被抑制并清掉段累计。播报期主上行是
关的，这句话只留在 1.5s 预滚里，抑制后不会 abort、也不会送出。
"""

from __future__ import annotations

import threading
import time

import numpy as np

from core.aec import EchoCanceller, to_float, to_int16
from core.utils.logger import logger

SAMPLE_RATE = 16000
VAD_FRAME = 512  # Silero @16k 固定帧长（32ms）

# 标定
CALIB_MIN_SAMPLES = int(0.8 * SAMPLE_RATE)  # 至少收集 0.8s 再尝试标定
CALIB_MAX_SAMPLES = int(4.0 * SAMPLE_RATE)  # 滚动标定窗长度；亦作「有声参考」降级预算（攒满仍无成功相关才降级）
MAX_BULK_DELAY = 8000  # 时延估计搜索上限（0.5s）
SAFETY_MARGIN = 256  # 对齐安全余量（样本），残差由滤波器尾长吸收
CALIB_MIN_PEAK_RATIO = 0.25  # 归一化相关峰下限（启发式，阶段 4 可调）
REF_VOICED_DBFS = -45.0  # 参考窗电平高于此值才算「播报有声」（标定能量门，2026-11-23 修复）
CALIB_EVAL_STEP = int(0.2 * SAMPLE_RATE)  # 标定评估最小步进（限 FFT 频次与预算计量，2026-11-23 修复）
MAX_ONSET_SAMPLES = int(6.0 * SAMPLE_RATE)  # 麦克风时钟与参考时钟的起始偏移上限：start() 早于首帧 TTS 音频数秒（整轮 LLM 往返），标定须在该偏移范围内搜索总时延（2026-11-23 修复）
ARM_SAMPLES = int(0.35 * SAMPLE_RATE)  # 有声参考送进 AEC 后的收敛预热：只处理不触发，防起始回声泄漏。0.75s 时两次吞掉真人首次开口，缩至 0.35s。按「已处理的有声参考样本」累计，不从最新喂入下标起算（2026-11-23）。

# AEC / 缓冲
AEC_FILTER_LEN = 2048
AEC_BLOCK_LEN = 256
REF_RING_CAPACITY = 8 * SAMPLE_RATE  # 参考环形缓冲 8s
PREROLL_SAMPLES = int(1.5 * SAMPLE_RATE)  # 消回声预缓冲 1.5s（原 1s；短句在 relisten 完成前已说完，预滚需多覆盖句头）
# 一次 process_available 最多处理这么多麦克风样本。现场一次调用在
# fed 不变时连续跑了约 7 万样本、墙钟 6 秒以上，播报和 abort 都挤在后面。
PROCESS_BUDGET_SAMPLES = 512  # 32ms @16kHz
# 标定之后待处理队列的上限。比预滚更老的样本已经送不出去，先处理它们
# 会把当前这句话推迟数秒。超出部分丢最老的，保留尾部。
LIVE_PENDING_CAP = PREROLL_SAMPLES
# 已经对齐、并且确实写在参考 tap 里的样本达到这个长度仍无有声参考时，
# 结束预热。对齐下标还没落到已写入区间时读回来的是补零，不能计入这段。
MAX_WARMUP_AEC_SAMPLES = int(0.60 * SAMPLE_RATE)
# 正常 float 满幅是 1。峰值远高于此是滤波器发散，不是麦克风语音。
AEC_DIVERGE_PEAK = 4.0

# 打断判定
VAD_THRESHOLD = 0.5  # 监视器专用高置信阈值（与会话 VAD 的阈值独立）
SPEECH_TRIGGER_SAMPLES = int(0.25 * SAMPLE_RATE)  # 连续语音 ≥250ms
SPEECH_GAP_SAMPLES = int(0.10 * SAMPLE_RATE)  # 允许的语音间隙 100ms
RESIDUAL_GATE_DB = 12.0  # 触发段电平须高于残差底噪估计的余量
ABS_FLOOR_DBFS = -50.0  # 触发段电平绝对下限


def _dbfs(x: np.ndarray) -> float:
    """float 样本（[-1,1]）的 RMS 电平，单位 dBFS。"""
    if len(x) == 0:
        return -120.0
    return float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-20))


def _estimate_delay_with_quality(mic: np.ndarray, ref: np.ndarray, max_delay: int):
    """互相关时延估计，额外返回归一化相关峰（标定质量指标）。

    返回 (lag, peak_ratio)：mic[n] ≈ ref[n - lag]；
    peak_ratio = corr[lag] / (||mic|| * ||ref||)，纯回声时接近
    直达路径能量占比，近端语音/噪声会把它拉低。
    """
    mic64 = np.asarray(mic, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    n = len(mic64) + len(ref64)
    nfft = 1 << int(np.ceil(np.log2(n)))
    m = np.fft.rfft(mic64, nfft)
    r = np.fft.rfft(ref64, nfft)
    corr = np.fft.irfft(m * np.conj(r), nfft)
    hi = min(max_delay, len(corr) - 1)
    lag = int(np.argmax(corr[: hi + 1]))
    denom = float(np.linalg.norm(mic64) * np.linalg.norm(ref64)) + 1e-20
    peak_ratio = float(corr[lag] / denom)
    return lag, peak_ratio


def _estimate_total_delay(mic_w: np.ndarray, ref_big: np.ndarray, max_onset: int):
    """带起始偏移的总时延估计（2026-11-23 修复新增）。

    ref_big 是从标定尾窗起点再往前 max_onset 样本开始读的长参考段；
    麦克风窗 mic_w[k] ≈ ref_big[k + s]，峰值处 s = max_onset − T，
    T 为总时延（起始偏移 + 播放管线时延）。返回 (T, peak_ratio)，
    peak_ratio 以峰值处实际重叠的参考段能量归一化。
    """
    mic64 = np.asarray(mic_w, dtype=np.float64)
    ref64 = np.asarray(ref_big, dtype=np.float64)
    L = len(mic64)
    n = L + len(ref64)
    nfft = 1 << int(np.ceil(np.log2(n)))
    M = np.fft.rfft(mic64, nfft)
    R = np.fft.rfft(ref64, nfft)
    corr = np.fft.irfft(np.conj(M) * R, nfft)
    hi = min(max_onset, len(ref64) - L)
    if hi < 0:
        return 0, 0.0
    s = int(np.argmax(corr[: hi + 1]))
    seg = ref64[s : s + L]
    denom = float(np.linalg.norm(mic64) * np.linalg.norm(seg)) + 1e-20
    ratio = float(corr[s] / denom)
    return max_onset - s, ratio


class Resampler24kTo16k:
    """24kHz → 16kHz 流式多相重采样（up=2, down=3）。

    滤波器与 scipy.signal.resample_poly 内部设计一致：61 taps
    Kaiser 窗（beta=5.0）低通，截止为上采样率（48kHz）奈奎斯特的
    1/3（即 8kHz），且与 resample_poly 一样补偿滤波器群延迟
    （30 个上采样点 = 10 个输出样本），输出与一次性 resample_poly
    样本级一致（浮点误差除外）。定义：
        out[m] = Σ_k x[k]·h[3m + 30 − 2k]
    可任意分块流式喂入。
    """

    UP, DOWN = 2, 3

    def __init__(self):
        max_rate = max(self.UP, self.DOWN)
        n_taps = 2 * 10 * max_rate + 1  # 61
        fc = 1.0 / max_rate / 2.0  #  cycles/sample @48k = 1/6
        n = np.arange(n_taps)
        mid = (n_taps - 1) / 2.0
        h = 2 * fc * np.sinc(2 * fc * (n - mid)) * np.kaiser(n_taps, 5.0)
        self._h = (h * self.UP).astype(np.float64)
        self._delay_up = (n_taps - 1) // 2  # 群延迟（上采样点）= 30
        self._in = np.zeros(0, dtype=np.float64)  # 尚未消费的输入尾部
        self._in_base = 0  # _in[0] 的绝对下标
        self._next_out = 0  # 下一个输出样本的绝对下标

    def _k_lo(self, center: int) -> int:
        """给定 h 中心下标 center，最小需要的输入下标（钳制到 ≥0）。"""
        return max(0, -(-(center - (len(self._h) - 1)) // 2))

    def process(self, x: np.ndarray) -> np.ndarray:
        """喂入 24kHz float 样本，返回可计算出的 16kHz float 样本。"""
        if len(x):
            self._in = np.concatenate([self._in, np.asarray(x, dtype=np.float64)])
        total_in = self._in_base + len(self._in)
        h = self._h
        outs = []
        m = self._next_out
        # 输出 m 的中心下标为 3m + delay_up，需要的最大输入下标为其一半
        while (3 * m + self._delay_up) // 2 <= total_in - 1:
            center = 3 * m + self._delay_up
            k_lo = self._k_lo(center)
            k_hi = center // 2
            ks = np.arange(k_lo, k_hi + 1)
            idx = center - 2 * ks
            valid = (idx >= 0) & (idx < len(h))
            if valid.any():
                seg = self._in[ks[valid] - self._in_base]
                outs.append(float(np.dot(seg, h[idx[valid]])))
            else:
                outs.append(0.0)
            m += 1
        self._next_out = m
        # 丢弃后续输出不再需要的输入前缀
        if m > 0:
            keep_from = self._k_lo(3 * m + self._delay_up)
            drop = keep_from - self._in_base
            if drop > 0:
                self._in = self._in[drop:]
                self._in_base = keep_from
        return np.asarray(outs, dtype=np.float32)


class ReferenceTap:
    """参考信号（bridge 下发给音箱的 TTS PCM）环形缓冲，线程安全。

    push_pcm() 由 AudioCodec.write_audio 在解码后调用（任意线程）；
    监视器以绝对下标 read()（下标 = tap 收到的第 n 个 16k 样本）。
    已被覆盖或尚未写入的区间读回 0。
    """

    def __init__(self, capacity: int = REF_RING_CAPACITY):
        self._cap = int(capacity)
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._total = 0  # 已写入样本总数（= 下一个样本的绝对下标）
        self._lock = threading.Lock()
        self._resampler = Resampler24kTo16k()
        self._rate_warned = False

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total

    def push_samples(self, samples: np.ndarray):
        """直接喂入 16kHz float 样本（测试/已重采样路径）。"""
        x = np.asarray(samples, dtype=np.float32)
        if len(x) == 0:
            return
        with self._lock:
            idx = (self._total + np.arange(len(x))) % self._cap
            self._buf[idx] = x
            self._total += len(x)

    def push_pcm(self, pcm: bytes, sample_rate: int):
        """喂入 write_audio 解码出的 int16 PCM（24k 或 16k）。"""
        if not pcm:
            return
        x = to_float(np.frombuffer(pcm, dtype=np.int16))
        if sample_rate == 24000:
            x = self._resampler.process(x.astype(np.float64))
        elif sample_rate != SAMPLE_RATE:
            if not self._rate_warned:
                self._rate_warned = True
                logger.warning(
                    f"[BargeIn] Unsupported reference sample rate {sample_rate}, tap disabled for this stream",
                    module="BargeIn",
                )
            return
        self.push_samples(x)

    def read(self, start: int, length: int) -> np.ndarray:
        """读取 [start, start+length) 的样本；缺失处补 0。"""
        out = np.zeros(length, dtype=np.float32)
        if length <= 0:
            return out
        with self._lock:
            total = self._total
            lo = max(start, total - self._cap, 0)
            hi = min(start + length, total)
            if hi > lo:
                idx = np.arange(lo, hi) % self._cap
                out[lo - start : hi - start] = self._buf[idx]
        return out


class BargeInCore:
    """AEC + VAD 打断判定核心（纯逻辑，无线程/设备依赖）。

    每个 SPEAKING 期调用 start() 复位；麦克风与参考都按 16kHz
    float 样本喂入（参考经 ReferenceTap，start() 时快照其下标为
    本期原点，两路按到达顺序近似同钟——分块到达抖动由对齐安全
    余量与滤波器尾长吸收）；process_available() 推进标定/AEC/VAD，
    命中打断返回 "bargein"（只报一次），其余返回 None。
    """

    def __init__(self, tap: ReferenceTap, vad_fn):
        self._tap = tap
        self._vad_fn = vad_fn
        self._started = False
        self._reset_state()
        # 跨 SPEAKING 期保留的最近一次成功对齐（不进 _reset_state）：
        # 同一设备/链路的时延近似恒定，本期标定失败时可复用兜底。
        self._last_good_align = None

    # -- 状态 -----------------------------------------------------------

    def _reset_state(self):
        self._aec = EchoCanceller(
            sample_rate=SAMPLE_RATE,
            filter_len=AEC_FILTER_LEN,
            block_len=AEC_BLOCK_LEN,
        )
        self._ref_base = 0
        self._mic_fed = 0  # 本期已喂入的麦克风样本数
        self._mic_proc = 0  # 已送入 AEC 的麦克风样本数
        self._mic_pending = np.zeros(0, dtype=np.float32)
        self._calib_mic = np.zeros(0, dtype=np.float32)
        self._calib_base = 0  # _calib_mic[0] 的本期麦克风下标（滚动窗，2026-11-23 修复）
        self._voiced_samples = 0  # 已评估的「有声参考」样本数（降级预算，2026-11-23 修复）
        self._last_eval_end = 0  # 上次标定评估覆盖到的本期麦克风下标（2026-11-23 修复）
        # 预热终点（本期麦克风 / 处理坐标）。None 表示有声参考
        # 还没累计满 ARM_SAMPLES，此间不触发。
        self._arm_until = None
        self._arm_voiced = 0  # 已送进 AEC 的有声参考样本数
        self._aec_origin = 0  # 本段 AEC 第一个样本的处理坐标
        self._aec_origin_set = False
        self._aec_samples = 0  # 实际送进 AEC 的样本数（不含直接丢弃的积压）
        self._emit_end = 0  # 最近一块 AEC 输出对应的处理坐标终点
        self._out_count = 0  # 已产出的 AEC 输出样本数
        self._backlog_capped_logged = False
        self._live_drop_unlogged = 0
        self._divergence_logged = False
        self._valid_ref_samples = 0  # 对齐后确实落在参考 tap 内的样本
        self._ref_hold_logged = False
        self._calib_off = 0  # 最近一次标定评估的尾窗起点（本期麦克风下标）
        self.calibrated = False
        self.degraded = False
        self.align = 0  # 已应用的参考对齐偏移（样本）
        self.est_delay = None
        self.triggered = False
        self._clean_pending = np.zeros(0, dtype=np.float32)
        self._preroll = np.zeros(0, dtype=np.float32)
        self._speech_run = 0  # 当前连续语音段样本数
        self._gap = 0  # 当前语音间隙样本数
        self._seg_samples = []  # 当前语音段的帧（算段电平用）
        self._seg_start = None  # 当前语音段起点（处理坐标）
        self._seg_end = None  # 当前语音段终点（处理坐标，开区间右端）
        self._floor_db = None  # 残差回声底噪估计（dBFS）

    def _coords(self) -> str:
        """喂入/处理/积压/预热/语音段坐标。只打计数，不打样本。"""
        arm = "pending" if self._arm_until is None else str(self._arm_until)
        return (
            f"fed={self._mic_fed} proc={self._mic_proc} "
            f"backlog={self._mic_fed - self._mic_proc} "
            f"pending={len(self._mic_pending)} "
            f"aec_origin={self._aec_origin} aec_samples={self._aec_samples} "
            f"arm_until={arm} arm_voiced={self._arm_voiced} "
            f"seg={self._seg_start}:{self._seg_end}"
        )

    def _ensure_aec_origin(self):
        if self._aec_origin_set:
            return
        self._aec_origin = self._mic_proc
        self._aec_origin_set = True

    def _aligned_ref_in_tap(self) -> bool:
        """当前这块麦克风对齐到的参考，是否大部分已经写在 tap 里。

        下标在 tap 起点之前、或还没被 write_audio 写入时，read() 补 0。
        那种 0 不是播报静音，不能用来结束预热。
        """
        start = self._ref_base + (self._mic_proc - AEC_BLOCK_LEN) - self.align
        end = start + AEC_BLOCK_LEN
        written = self._tap.total_written
        if end <= 0 or start >= written:
            return False
        covered = min(end, written) - max(start, 0)
        return covered >= AEC_BLOCK_LEN // 2

    def _note_arm_block(self, ref_blk: np.ndarray):
        """预热只统计已经对齐且落在参考 tap 内的样本。

        有声参考满 ARM_SAMPLES 就结束。tap 里的参考持续安静达到
        MAX_WARMUP_AEC_SAMPLES 也结束。对齐位置还在 tap 已写入区间
        之外时不计数：那里读到的是补零，滤波器还没有看见播报。
        """
        if self._arm_until is not None:
            return
        if not self._aligned_ref_in_tap():
            if not self._ref_hold_logged:
                self._ref_hold_logged = True
                logger.info(
                    "[BargeIn] Warm-up holding: aligned reference is outside "
                    f"the written tap; align={self.align} "
                    f"ref_written={self._tap.total_written - self._ref_base} "
                    f"{self._coords()}",
                    module="BargeIn",
                )
            return
        self._valid_ref_samples += len(ref_blk)
        if _dbfs(ref_blk) >= REF_VOICED_DBFS:
            self._arm_voiced += len(ref_blk)
        voiced_done = self._arm_voiced >= ARM_SAMPLES
        quiet_done = self._valid_ref_samples >= MAX_WARMUP_AEC_SAMPLES
        if not voiced_done and not quiet_done:
            return
        self._arm_until = self._mic_proc
        why = "voiced-ref" if voiced_done else "valid-ref-quiet"
        logger.info(
            f"[BargeIn] Warm-up ended ({why}) after "
            f"{self._arm_voiced} voiced-ref samples: {self._coords()}",
            module="BargeIn",
        )

    def _cap_live_pending(self):
        """标定后只保留最近 LIVE_PENDING_CAP 个待处理样本。"""
        extra = len(self._mic_pending) - LIVE_PENDING_CAP
        if extra <= 0:
            return
        drop = extra - (extra % AEC_BLOCK_LEN)
        if drop <= 0:
            return
        self._mic_pending = self._mic_pending[drop:]
        self._mic_proc += drop
        self._emit_end = self._mic_proc
        self._clean_pending = np.zeros(0, dtype=np.float32)
        self._live_drop_unlogged += drop
        if self._live_drop_unlogged < SAMPLE_RATE // 2:
            return
        logger.info(
            "[BargeIn] Live backlog capped at "
            f"{LIVE_PENDING_CAP} samples "
            f"({LIVE_PENDING_CAP / SAMPLE_RATE:.1f}s): "
            f"dropped={self._live_drop_unlogged} {self._coords()}",
            module="BargeIn",
        )
        self._live_drop_unlogged = 0

    def _in_warmup_seg(self) -> bool:
        end = self._seg_end if self._seg_end is not None else self._mic_proc
        return self._arm_until is None or end < self._arm_until

    def _clear_speech_run(self):
        self._speech_run = 0
        self._seg_samples = []
        self._seg_start = None
        self._seg_end = None

    def _suppress_warmup_hit(self):
        """预热期内的命中不上报。日志留在清段之前，带语音段起止。"""
        self.triggered = False
        logger.info(
            f"[BargeIn] Trigger suppressed during AEC warm-up: {self._coords()}",
            module="BargeIn",
        )
        self._clear_speech_run()

    def start(self):
        """进入 SPEAKING 期：复位 AEC/标定/缓冲，快照参考原点。"""
        self._reset_state()
        self._ref_base = self._tap.total_written
        reset = getattr(self._vad_fn, "reset", None)
        if callable(reset):
            reset()
        self._started = True

    def stop(self):
        self._started = False

    @property
    def preroll_bytes(self) -> bytes:
        """触发时的消回声预缓冲（int16 PCM，≤1s）。"""
        return to_int16(self._preroll).tobytes()

    # -- 输入 -----------------------------------------------------------

    def feed_mic(self, samples: np.ndarray):
        x = np.asarray(samples, dtype=np.float32)
        if len(x) == 0 or not self._started:
            return
        self._mic_fed += len(x)
        self._mic_pending = np.concatenate([self._mic_pending, x])
        if not self.calibrated and not self.degraded:
            # 滚动窗：始终保留最近 CALIB_MAX_SAMPLES，旧样本滑出——
            # TTS 参考迟到时标定窗能跟着滑进有声区，不再冻结过期
            # （2026-11-23 修复；原实现攒满 4s 即冻结，播报迟到必降级）。
            self._calib_mic = np.concatenate([self._calib_mic, x])
            excess = len(self._calib_mic) - CALIB_MAX_SAMPLES
            if excess > 0:
                self._calib_mic = self._calib_mic[excess:]
                self._calib_base += excess

    # -- 主推进 -----------------------------------------------------------

    def process_available(self):
        if not self._started or self.triggered:
            return None
        if not self.calibrated:
            if self.degraded:
                # 降级期丢弃待处理音频（保持下标计数一致），不判定
                self._mic_proc += len(self._mic_pending)
                self._mic_pending = np.zeros(0, dtype=np.float32)
                return None
            self._try_calibrate()
            if not self.calibrated:
                # 标定窗口外的积压对 AEC 无意义，丢弃以限制缓冲
                if len(self._mic_pending) > CALIB_MAX_SAMPLES:
                    drop = len(self._mic_pending) - CALIB_MAX_SAMPLES
                    self._mic_pending = self._mic_pending[drop:]
                    self._mic_proc += drop
                    if drop > 0 and not self._backlog_capped_logged:
                        self._backlog_capped_logged = True
                        logger.info(
                            "[BargeIn] Pre-calibration backlog capped at "
                            f"{CALIB_MAX_SAMPLES} samples "
                            f"({CALIB_MAX_SAMPLES / SAMPLE_RATE:.1f}s): "
                            f"dropped={drop} {self._coords()}",
                            module="BargeIn",
                        )
                return None
        self._cap_live_pending()
        return self._process_aligned()

    def _degrade_or_fallback(self, reason: str):
        # 本期标定失败：有历史成功对齐就复用兜底（触发判定仍有 AEC
        # 残差 + 电平双门把关）；从未成功过才整期降级不判定。
        if self._last_good_align is not None:
            self.align = self._last_good_align
            self.calibrated = True
            logger.warning(
                "[BargeIn] Calibration degraded "
                f"({reason}); using last-good align={self.align}; {self._coords()}",
                module="BargeIn",
            )
            return
        self.degraded = True
        logger.warning(
            f"[BargeIn] Calibration degraded: {reason}",
            module="BargeIn",
        )

    def _try_calibrate(self):
        n = len(self._calib_mic)
        if n < CALIB_MIN_SAMPLES:
            return
        tail_end = self._calib_base + n
        if tail_end - self._last_eval_end < CALIB_EVAL_STEP:
            return  # 评估步进未到，避免对同一窗反复做 FFT
        # 只对最近 CALIB_MIN_SAMPLES 做相关：段首常有静音头（TTS 首帧
        # 迟到、句首停顿），整窗相关会被稀释；滚动窗在两路同偏移处
        # 切取（off 为本期麦克风下标），lag 语义与整窗一致。
        off = tail_end - CALIB_MIN_SAMPLES
        self._calib_off = off
        mic_w = self._calib_mic[n - CALIB_MIN_SAMPLES:]
        # 参考段往前多读 MAX_ONSET：两路时钟的起始偏移（start() 到
        # 首帧 TTS 音频之间整轮 LLM 往返，可达数秒）连同管线时延
        # 一起估出总时延 T（2026-11-23 修复；原实现只搜 0.5s，
        # 偏移超限时相关峰永远找不到，实测 ratio 仅 0.09）。
        ref_big = self._tap.read(
            self._ref_base + off - MAX_ONSET_SAMPLES,
            CALIB_MIN_SAMPLES + MAX_ONSET_SAMPLES,
        )
        prev_end = self._last_eval_end
        self._last_eval_end = tail_end
        ref_db = _dbfs(ref_big)
        if ref_db < REF_VOICED_DBFS:
            # 参考还没出声（TTS 迟到/句间停顿）：滑窗等待，不判失败
            # （2026-11-23 修复；原实现按麦克风墙钟攒满 4s 即降级，
            # 播报迟到时标定窗在有声区到来前就已冻结过期）。
            return
        # 参考有声：按本次新覆盖的样本数计入降级预算
        self._voiced_samples += tail_end - prev_end
        lag, ratio = _estimate_total_delay(mic_w, ref_big, MAX_ONSET_SAMPLES)
        if ratio < CALIB_MIN_PEAK_RATIO:
            if self._voiced_samples >= CALIB_MAX_SAMPLES:
                self._degrade_or_fallback(
                    f"corr peak ratio {ratio:.2f} < {CALIB_MIN_PEAK_RATIO}"
                )
            return
        self.est_delay = lag
        self.align = max(0, lag - SAFETY_MARGIN)
        self.calibrated = True
        self._last_good_align = self.align
        logger.info(
            "[BargeIn] Calibrated: "
            f"est delay={lag}, applied align={self.align}, "
            f"peak ratio={ratio:.2f}, tail_end={tail_end}, {self._coords()}",
            module="BargeIn",
        )

    def _process_aligned(self):
        # 以 AEC 块长推进，但单次调用有预算。麦克风块 p 处的参考取
        # tap[ref_base + p - align]。预热命中只清本段，预算内更后面的
        # 真人语音仍可触发。
        self._ensure_aec_origin()
        processed = 0
        while processed < PROCESS_BUDGET_SAMPLES and (
            len(self._mic_pending) >= AEC_BLOCK_LEN
            or len(self._clean_pending) >= VAD_FRAME
        ):
            if (
                len(self._mic_pending) >= AEC_BLOCK_LEN
                and processed < PROCESS_BUDGET_SAMPLES
            ):
                blk = self._mic_pending[:AEC_BLOCK_LEN]
                self._mic_pending = self._mic_pending[AEC_BLOCK_LEN:]
                ref_blk = self._tap.read(
                    self._ref_base + self._mic_proc - self.align, AEC_BLOCK_LEN
                )
                self._mic_proc += AEC_BLOCK_LEN
                self._aec_samples += AEC_BLOCK_LEN
                processed += AEC_BLOCK_LEN
                self._note_arm_block(ref_blk)
                out = self._aec.process(blk, ref_blk)
                if len(out):
                    peak = float(np.max(np.abs(out)))
                    if peak > AEC_DIVERGE_PEAK:
                        self._aec.reset()
                        out = np.zeros_like(out)
                        if not self._divergence_logged:
                            self._divergence_logged = True
                            logger.warning(
                                "[BargeIn] AEC output diverged "
                                f"(peak={peak:.1f}); filter reset; {self._coords()}",
                                module="BargeIn",
                            )
                    self._out_count += len(out)
                    self._clean_pending = np.concatenate([self._clean_pending, out])
                    self._preroll = np.concatenate([self._preroll, out])
                    if len(self._preroll) > PREROLL_SAMPLES:
                        self._preroll = self._preroll[-PREROLL_SAMPLES:]
                self._emit_end = self._mic_proc
            ev = self._vad_step()
            if ev == "bargein":
                if self._in_warmup_seg():
                    self._suppress_warmup_hit()
                    continue
                return "bargein"
            if (
                len(self._mic_pending) < AEC_BLOCK_LEN
                and len(self._clean_pending) < VAD_FRAME
            ):
                break
        return None

    def _vad_step(self):
        while len(self._clean_pending) >= VAD_FRAME:
            frame = self._clean_pending[:VAD_FRAME]
            self._clean_pending = self._clean_pending[VAD_FRAME:]
            frame_end = self._emit_end - len(self._clean_pending)
            frame_start = frame_end - VAD_FRAME
            warming = self._arm_until is None or frame_end < self._arm_until
            frame_db = _dbfs(frame)
            # 残差底噪跟踪：只采纳与当前底噪相符的帧（近端语音帧的
            # 电平远高于底噪，不会污染估计）。预热期例外：此间输出
            # 由未收敛回声主导，无条件采纳，让底噪估计先贴近真实
            # 残差水平，避免预热刚结束就被残差越过门限（2026-11-23
            # 修复；原规则底噪只能向下贴，静室开播时会被残差击穿）。
            if self._floor_db is None:
                self._floor_db = frame_db
            elif warming or frame_db < self._floor_db + 6.0:
                self._floor_db = 0.92 * self._floor_db + 0.08 * frame_db

            try:
                prob = float(self._vad_fn(frame))
            except Exception:
                prob = 0.0

            if prob >= VAD_THRESHOLD:
                if self._speech_run == 0:
                    self._seg_start = frame_start
                self._speech_run += VAD_FRAME
                self._gap = 0
                self._seg_samples.append(frame)
                self._seg_end = frame_end
                if self._speech_run >= SPEECH_TRIGGER_SAMPLES:
                    seg = np.concatenate(self._seg_samples)
                    seg_db = _dbfs(seg)
                    gate = max(ABS_FLOOR_DBFS, (self._floor_db or -70.0) + RESIDUAL_GATE_DB)
                    if seg_db >= gate:
                        self.triggered = True
                        logger.info(
                            "[BargeIn] Barge-in detected: "
                            f"seg={seg_db:.1f} dBFS, gate={gate:.1f} dBFS, "
                            f"{self._coords()}",
                            module="BargeIn",
                        )
                        return "bargein"
                    # 电平不够：更像残余回声，清掉本段重新累计
                    logger.debug(
                        "[BargeIn] Speech segment below level gate: "
                        f"seg={seg_db:.1f} dBFS, gate={gate:.1f} dBFS, "
                        f"{self._coords()}",
                        module="BargeIn",
                    )
                    self._clear_speech_run()
            else:
                self._gap += VAD_FRAME
                if self._gap > SPEECH_GAP_SAMPLES:
                    self._clear_speech_run()
        return None


class _SileroVadFn:
    """监视器专用 Silero VAD 推理（独立 ONNX 会话与状态）。

    唤醒会话共用的 VAD（core.services.audio.vad 的单例）有自己的
    pause/resume 节奏，不能复用其状态；这里单独持有一份模型会话
    （模型文件仅 ~2MB），reset() 对应每个 SPEAKING 期复位。
    """

    def __init__(self):
        from core.services.audio.vad.silero import OnnxWrapper
        from core.utils.file import get_model_file_path

        self._model = OnnxWrapper(path=get_model_file_path("silero_vad.onnx"))

    def reset(self):
        self._model.reset_states()

    def __call__(self, frame: np.ndarray) -> float:
        out = self._model(np.asarray(frame, dtype=np.float32), SAMPLE_RATE)
        return float(out.item())


class BargeInMonitor:
    """生产装配层：独立监视输入流 + 轮询线程 + 参考 tap + 判定核心。

    - feed_reference_pcm() 挂到 AudioCodec 的参考回调上；
    - attach_stream() 注入一条独立输入流（AudioCodec.create_monitor_stream）；
    - start()/stop() 随 SPEAKING 状态启停（由 XiaoZhi.set_device_state 调用）；
    - 命中打断时回调 on_trigger(preroll_bytes)；observe 模式只打日志。
    """

    def __init__(self, on_trigger, observe: bool = False):
        self.tap = ReferenceTap()
        self._core = BargeInCore(self.tap, _SileroVadFn())
        self._on_trigger = on_trigger
        self._observe = observe
        self._stream = None
        self._thread = None
        self._running = False
        self._fired = False

    # 参考信号入口（codec 线程调用）
    def feed_reference_pcm(self, pcm: bytes, sample_rate: int):
        self.tap.push_pcm(pcm, sample_rate)

    def attach_stream(self, stream):
        self._stream = stream

    @property
    def core(self) -> BargeInCore:
        return self._core

    def start(self):
        if self._running or self._stream is None:
            return
        self._core.start()
        self._fired = False
        try:
            self._stream.clear_input()
        except Exception:
            pass
        self._stream.start_stream()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="bargein-monitor", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
        self._core.stop()

    def _loop(self):
        while self._running:
            try:
                data = self._stream.read() if self._stream else b""
            except Exception:
                data = b""
            if data:
                self._core.feed_mic(to_float(np.frombuffer(data, dtype=np.int16)))
            pending = len(self._core._mic_pending)
            if data or pending >= AEC_BLOCK_LEN:
                ev = self._core.process_available()
                if ev == "bargein" and not self._fired:
                    self._fired = True
                    preroll = self._core.preroll_bytes
                    if self._observe:
                        logger.info(
                            f"[BargeIn] (observe) would trigger barge-in, preroll={len(preroll)} bytes",
                            module="BargeIn",
                        )
                    else:
                        try:
                            self._on_trigger(preroll)
                        except Exception as exc:
                            logger.error(
                                f"[BargeIn] on_trigger failed: {exc}", module="BargeIn"
                            )
                # 积压还在时让出一会儿，避免 AEC 持续占满 CPU，播放下发进不了主循环。
                if len(self._core._mic_pending) >= AEC_BLOCK_LEN:
                    time.sleep(0.001)
            else:
                time.sleep(0.01)
