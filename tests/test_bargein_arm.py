"""小智打断预热坐标的离线测试。

用合成 16kHz 信号驱动 BargeInCore，核对积压、预热、预滚和 abort 文案。
这里不能说明真实音箱已经停播，也不能说明服务端 ASR 识别正确。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.aec import to_float
from core.services.audio.bargein import (
    AEC_BLOCK_LEN,
    AEC_DIVERGE_PEAK,
    ARM_SAMPLES,
    CALIB_MIN_SAMPLES,
    LIVE_PENDING_CAP,
    MAX_WARMUP_AEC_SAMPLES,
    PROCESS_BUDGET_SAMPLES,
    PREROLL_SAMPLES,
    SAFETY_MARGIN,
    SAMPLE_RATE,
    BargeInCore,
    ReferenceTap,
)
from core.services.protocols.protocol import Protocol
from core.services.protocols.typing import AbortReason

SR = SAMPLE_RATE


def _dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return -120.0
    return float(10.0 * np.log10(np.mean(x * x) + 1e-20))


def _noise(n: int, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(n)).astype(np.float64)
    kernel = np.ones(5) / 5.0
    x = np.convolve(x, kernel, mode="same")
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    return (x * (amp / rms)).astype(np.float32)


def _delayed(ref: np.ndarray, delay: int, gain: float) -> np.ndarray:
    mic = np.zeros(len(ref), dtype=np.float32)
    if 0 < delay < len(ref):
        mic[delay:] = ref[:-delay] * gain
    elif delay == 0:
        mic[:] = ref * gain
    return mic


def _energy_vad(threshold_db: float = -18.0):
    def vad(frame: np.ndarray) -> float:
        return 1.0 if _dbfs(frame) >= threshold_db else 0.0

    return vad


def _always_speech(_frame: np.ndarray) -> float:
    return 1.0


def _drain(core: BargeInCore, max_rounds: int = 20000):
    """按监视线程的方式把预算循环跑完。单次 process_available 不再吞掉整段积压。"""
    for _ in range(max_rounds):
        pending_before = len(core._mic_pending)
        proc_before = core._mic_proc
        event = core.process_available()
        if event:
            return event
        if (
            core._mic_proc == proc_before
            and len(core._mic_pending) == pending_before
        ):
            return None
    return None


def _feed(core: BargeInCore, tap: ReferenceTap, mic: np.ndarray, ref: np.ndarray,
          chunk: int, pause: float = 0.0):
    """按块喂入等长的麦克风/参考。返回第一次 bargein，没有则 None。"""
    if len(mic) != len(ref):
        raise AssertionError(f"mic/ref length {len(mic)} != {len(ref)}")
    pos = 0
    while pos < len(mic):
        step = min(chunk, len(mic) - pos)
        core.feed_mic(mic[pos:pos + step])
        tap.push_samples(ref[pos:pos + step])
        if pause:
            time.sleep(pause)
        event = _drain(core)
        pos += step
        if event == "bargein":
            return "bargein"
    return None


def _new_core(vad):
    tap = ReferenceTap()
    core = BargeInCore(tap, vad)
    core.start()
    return tap, core


class BargeInArmCoordinateTest(unittest.TestCase):
    def test_four_second_backlog_keeps_first_sentence(self):
        """标定前积压 4s 时，积压后部的首句必须打断，不能按最新喂入点预热掉。"""
        delay = 320
        total = 4 * SR
        # 落在「保留的 1.5s 尾部」里、预热之后，并且在标定尾窗之前。
        speech_at = 45000
        speech_n = 4608
        self.assertLessEqual(speech_at + speech_n, total - CALIB_MIN_SAMPLES)

        ref = _noise(total, 0.10, 1)
        mic = _delayed(ref, delay, 0.55)
        ramp = np.linspace(0.16, 0.70, speech_n, dtype=np.float32)
        speech = _noise(speech_n, 1.0, 2) * ramp
        mic = mic.copy()
        mic[speech_at:speech_at + speech_n] += speech

        tap, core = _new_core(_energy_vad())
        with self.assertLogs("xiaozhi", level="INFO") as logs:
            event = _feed(core, tap, mic, ref, chunk=1024)
        self.assertEqual("bargein", event)
        self.assertIsNotNone(core._arm_until)
        self.assertGreaterEqual(core._seg_end, core._arm_until)
        self.assertGreater(core._mic_proc, speech_at)
        # 决策时不能还背着整段 4s。比预滚更老的样本应已被丢掉。
        self.assertLessEqual(len(core._mic_pending), LIVE_PENDING_CAP + PROCESS_BUDGET_SAMPLES)
        self.assertLessEqual(core._mic_fed - core._mic_proc, LIVE_PENDING_CAP + PROCESS_BUDGET_SAMPLES)

        preroll = np.frombuffer(core.preroll_bytes, dtype=np.int16)
        self.assertGreater(len(preroll), SR // 2)
        self.assertLessEqual(len(preroll), PREROLL_SAMPLES)
        self.assertEqual(len(core.preroll_bytes), len(preroll) * 2)
        head = to_float(preroll[:2000])
        tail = to_float(preroll[-2000:])
        self.assertGreater(_dbfs(tail), _dbfs(head) + 6.0)
        # 句内幅度递增，预滚尾部应比其前一段更响，顺序没有倒过来。
        window = to_float(preroll[-3200:])
        self.assertGreater(_dbfs(window[1600:]), _dbfs(window[:1600]) + 1.0)

        text = "\n".join(logs.output)
        for token in ("fed=", "proc=", "backlog=", "arm_until=", "seg="):
            self.assertIn(token, text)
        self.assertIn("Barge-in detected", text)
        self.assertNotIn("suppressed", text.lower())

    def test_zero_backlog_first_sentence_after_calibration(self):
        """只有标定窗、没有额外积压时，窗后立刻说出的首句要能触发。"""
        delay = 256
        n = CALIB_MIN_SAMPLES + delay
        ref = _noise(n, 0.10, 3)
        mic = _delayed(ref, delay, 0.55)
        tap, core = _new_core(_energy_vad())
        self.assertIsNone(_feed(core, tap, mic, ref, chunk=n))
        self.assertTrue(core.calibrated)
        self.assertIsNotNone(core._arm_until)
        self.assertLessEqual(core._arm_until, ARM_SAMPLES + AEC_BLOCK_LEN)
        self.assertLess(core._arm_until, n)

        speech_n = int(0.40 * SR)
        speech = _noise(speech_n, 0.50, 4)
        ref2 = _noise(speech_n, 0.10, 5)
        mic2 = _delayed(ref2, delay, 0.30) + speech
        event = _feed(core, tap, mic2, ref2, chunk=speech_n)
        self.assertEqual("bargein", event)
        self.assertGreaterEqual(core._seg_end, n)

    def test_warmup_boundary_suppresses_then_accepts_next_sentence(self):
        """预热边界内的一声被抑制；过了边界的下一句仍然打断。"""
        delay = 128
        total = int(1.6 * SR)
        ref = _noise(total, 0.12, 6)
        mic = _delayed(ref, delay, 0.05)
        early_at = 512
        early_n = 4608  # 触发点落在 0.35s 预热结束之前
        mic = mic.copy()
        mic[early_at:early_at + early_n] += _noise(early_n, 0.80, 7)
        tap, core = _new_core(_energy_vad(-20.0))
        with self.assertLogs("xiaozhi", level="INFO") as logs:
            event = _feed(core, tap, mic, ref, chunk=1024)
        self.assertIsNone(event)
        self.assertIsNotNone(core._arm_until)
        self.assertLess(early_at + early_n, core._arm_until)
        self.assertIn("Trigger suppressed during AEC warm-up", "\n".join(logs.output))
        self.assertIn("seg=", "\n".join(logs.output))

        late_n = int(0.40 * SR)
        ref2 = _noise(late_n, 0.12, 8)
        mic2 = _delayed(ref2, min(delay, late_n // 4), 0.05) + _noise(late_n, 0.70, 9)
        event = _feed(core, tap, mic2, ref2, chunk=late_n)
        self.assertEqual("bargein", event)
        self.assertGreaterEqual(core._seg_end, core._arm_until)

    def test_fast_and_slow_processing_share_sample_decision(self):
        """同一段样本，一次喂完和慢速分块（含墙钟等待）都要打断，预热点不跟墙钟走。"""
        delay = 256
        echo_n = SR
        speech_n = int(0.40 * SR)
        echo_ref = _noise(echo_n, 0.10, 10)
        echo_mic = _delayed(echo_ref, delay, 0.55)
        speech_ref = _noise(speech_n, 0.08, 11)
        speech_mic = _delayed(speech_ref, min(delay, speech_n // 4), 0.2) + _noise(speech_n, 0.55, 12)

        def run(chunk: int, pause: float):
            tap, core = _new_core(_energy_vad())
            started = time.perf_counter()
            self.assertIsNone(_feed(core, tap, echo_mic, echo_ref, chunk=chunk, pause=pause))
            event = _feed(core, tap, speech_mic, speech_ref, chunk=chunk, pause=0.0)
            return event, core._arm_until, time.perf_counter() - started

        fast_event, fast_arm, fast_dt = run(echo_n, 0.0)
        slow_event, slow_arm, slow_dt = run(640, 0.002)
        self.assertEqual("bargein", fast_event)
        self.assertEqual("bargein", slow_event)
        self.assertIsNotNone(fast_arm)
        self.assertIsNotNone(slow_arm)
        self.assertLess(abs(fast_arm - slow_arm), AEC_BLOCK_LEN * 2)
        self.assertLess(fast_dt, 0.30)
        self.assertGreater(slow_dt, fast_dt + 0.03)

    def test_pure_echo_does_not_trigger(self):
        """纯回声：能量门 VAD 不报，始终报语音的 VAD 也被电平门挡住。"""
        delay = 400
        n = 2 * SR
        ref = _noise(n, 0.12, 12)
        mic = _delayed(ref, delay, 0.60)

        tap, core = _new_core(_energy_vad(-18.0))
        self.assertIsNone(_feed(core, tap, mic, ref, chunk=2048))
        self.assertTrue(core.calibrated)
        self.assertIsNotNone(core._arm_until)
        self.assertFalse(core.triggered)

        tap, core = _new_core(_always_speech)
        self.assertIsNone(_feed(core, tap, mic, ref, chunk=2048))
        self.assertTrue(core.calibrated)
        self.assertFalse(core.triggered)

    def test_silence_prefix_does_not_consume_warmup_or_false_trigger(self):
        """参考还没出声时不消耗预热；随后的纯回声也不应打断。"""
        silence = 2 * SR
        echo = int(1.5 * SR)
        delay = 300
        ref = np.concatenate([
            np.zeros(silence, dtype=np.float32),
            _noise(echo, 0.12, 13),
        ])
        mic = np.concatenate([
            np.zeros(silence, dtype=np.float32),
            _delayed(ref[silence:], delay, 0.60),
        ])
        tap, core = _new_core(_energy_vad(-12.0))
        self.assertIsNone(_feed(core, tap, mic, ref, chunk=2048))
        self.assertFalse(core.triggered)

    def test_long_delay_calibration_and_reset(self):
        """大于旧版 0.5s 搜索上限的管线时延仍能对齐；start() 清空本期状态。"""
        delay = 20000
        n = delay + CALIB_MIN_SAMPLES + 2000
        ref = _noise(n, 0.12, 14)
        mic = _delayed(ref, delay, 0.70)
        tap, core = _new_core(_energy_vad())
        self.assertIsNone(_feed(core, tap, mic, ref, chunk=4096))
        self.assertTrue(core.calibrated)
        self.assertIsNotNone(core.est_delay)
        self.assertLess(abs(core.est_delay - delay), 40)
        self.assertEqual(max(0, core.est_delay - SAFETY_MARGIN), core.align)
        kept = core.align

        core.start()
        self.assertFalse(core.calibrated)
        self.assertEqual(0, core._mic_fed)
        self.assertEqual(0, core._mic_proc)
        self.assertIsNone(core._arm_until)
        self.assertFalse(core.triggered)
        self.assertEqual(kept, core._last_good_align)

    def test_fallback_align_still_hears_backlog_speech(self):
        """本期相关失败时复用上次对齐，积压里的人声仍可打断。"""
        delay = 480
        n = CALIB_MIN_SAMPLES + delay + 1000
        ref = _noise(n, 0.12, 15)
        mic = _delayed(ref, delay, 0.65)
        tap, core = _new_core(_energy_vad())
        self.assertIsNone(_feed(core, tap, mic, ref, chunk=n))
        self.assertTrue(core.calibrated)
        kept = core.align

        core.start()
        bad_n = 5 * SR
        t = np.arange(bad_n, dtype=np.float32) / SR
        ref_bad = (0.15 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        mic_bad = (0.12 * np.sin(2 * np.pi * 470 * t)).astype(np.float32)
        speech_at = int(4.45 * SR)
        speech_n = int(0.40 * SR)
        mic_bad = mic_bad.copy()
        mic_bad[speech_at:speech_at + speech_n] += _noise(speech_n, 0.55, 16)
        event = _feed(core, tap, mic_bad, ref_bad, chunk=2048)
        self.assertTrue(core.calibrated)
        self.assertIsNone(core.est_delay)
        self.assertEqual(kept, core.align)
        self.assertEqual("bargein", event)
        self.assertLess(core._arm_until, speech_at)

    def test_repeated_bargein_across_speaking_periods(self):
        delay = 280
        speech_n = int(0.40 * SR)

        def once(core, tap, seed: int):
            n = CALIB_MIN_SAMPLES + delay + 500
            ref = _noise(n, 0.10, seed)
            mic = _delayed(ref, delay, 0.55)
            self.assertIsNone(_feed(core, tap, mic, ref, chunk=n))
            ref2 = _noise(speech_n, 0.08, seed + 1)
            mic2 = _delayed(ref2, delay, 0.20) + _noise(speech_n, 0.50, seed + 2)
            return _feed(core, tap, mic2, ref2, chunk=speech_n)

        tap, core = _new_core(_energy_vad())
        self.assertEqual("bargein", once(core, tap, 21))
        first_preroll = core.preroll_bytes
        core.start()
        self.assertEqual("bargein", once(core, tap, 31))
        second = core.preroll_bytes
        self.assertGreater(len(first_preroll), 0)
        self.assertGreater(len(second), 0)
        self.assertNotEqual(first_preroll, second)

    def test_backlog_cap_keeps_speech_after_dropped_prefix(self):
        """超过 4s 的前缀被截断后，留下的积压里、预热之后的人声仍然打断。"""
        silence = int(4.2 * SR)
        voiced = int(1.8 * SR)
        total = silence + voiced
        delay = 250
        ref = np.concatenate([
            np.zeros(silence, dtype=np.float32),
            _noise(voiced, 0.12, 17),
        ])
        mic = np.concatenate([
            np.zeros(silence, dtype=np.float32),
            _delayed(ref[silence:], delay, 0.55),
        ])
        speech_at = silence + int(0.50 * SR)
        speech_n = int(0.40 * SR)
        self.assertLess(speech_at + speech_n, total - CALIB_MIN_SAMPLES)
        mic = mic.copy()
        mic[speech_at:speech_at + speech_n] += _noise(speech_n, 0.55, 18)

        tap, core = _new_core(_energy_vad())
        with self.assertLogs("xiaozhi", level="INFO") as logs:
            event = _feed(core, tap, mic, ref, chunk=2048)
        self.assertEqual("bargein", event)
        self.assertIn("backlog capped", "\n".join(logs.output))
        self.assertGreater(core._aec_origin, 0)
        self.assertLess(core._arm_until, speech_at)
        self.assertGreaterEqual(core._seg_end, core._arm_until)
        self.assertGreater(core._mic_proc - core._aec_origin, ARM_SAMPLES)
        self.assertLess(core._arm_until, core._mic_fed)
        preroll = np.frombuffer(core.preroll_bytes, dtype=np.int16)
        self.assertGreater(len(preroll), 0)
        self.assertLessEqual(len(preroll), PREROLL_SAMPLES)

    def test_one_call_does_not_drain_multi_second_backlog(self):
        """16k 分块喂入后，单次处理不得超过预算。当前补丁会一次啃完整段。"""
        delay = 256
        total = 3 * SR
        ref = _noise(total, 0.12, 41)
        mic = _delayed(ref, delay, 0.55)
        tap, core = _new_core(_energy_vad())
        # 先攒满再标定，模拟监视线程被上一轮堵住之后一次性读到数秒。
        core.feed_mic(mic)
        tap.push_samples(ref)
        proc_before = core._mic_proc
        core.process_available()
        advanced = core._mic_proc - proc_before
        self.assertLessEqual(advanced, LIVE_PENDING_CAP + PROCESS_BUDGET_SAMPLES)
        self.assertLessEqual(len(core._mic_pending), LIVE_PENDING_CAP)
        self.assertGreater(len(core._mic_pending), 0)

    def test_slow_aec_yields_so_playback_can_tick(self):
        """慢 AEC 时单次调用必须返回，播放节拍不能等整段积压结束。"""
        delay = 200
        total = 2 * SR
        ref = _noise(total, 0.12, 42)
        mic = _delayed(ref, delay, 0.5)
        tap, core = _new_core(_energy_vad())
        core.feed_mic(mic)
        tap.push_samples(ref)
        real = core._aec.process

        def slow(mic_blk, ref_blk):
            time.sleep(0.004)
            return real(mic_blk, ref_blk)

        core._aec.process = slow
        ticks = []
        for _ in range(6):
            started = time.perf_counter()
            core.process_available()
            ticks.append(time.perf_counter() - started)
            # 调用返回后主循环才能接着写播放；这里用间隔代表一次播放节拍。
        self.assertGreater(len(core._mic_pending), SR // 2)
        self.assertTrue(all(dt < 0.05 for dt in ticks))
        full_drain_blocks = total / AEC_BLOCK_LEN
        self.assertLess(sum(ticks), full_drain_blocks * 0.004 * 0.5)

    def test_diverged_aec_output_does_not_count_as_speech(self):
        """峰值远超满幅是滤波器发散，不能当成打断。"""
        delay = 200
        n = 2 * SR
        ref = _noise(n, 0.12, 44)
        mic = _delayed(ref, delay, 0.5)
        tap, core = _new_core(_always_speech)
        core.feed_mic(mic[: SR])
        tap.push_samples(ref[: SR])
        _drain(core)
        self.assertTrue(core.calibrated)

        def explode(mic_blk, ref_blk):
            return np.full(len(mic_blk), AEC_DIVERGE_PEAK * 2, dtype=np.float32)

        core._aec.process = explode
        core.feed_mic(mic[SR // 2:])
        tap.push_samples(ref[SR // 2:])
        with self.assertLogs("xiaozhi", level="WARNING") as logs:
            event = _drain(core)
        self.assertIsNone(event)
        self.assertFalse(core.triggered)
        self.assertIn("diverged", "\n".join(logs.output))

    def test_large_delay_does_not_end_warmup_on_zero_fill(self):
        """对齐点还在参考 tap 之前时，0.6s 样本上限不能把预热放行。

        几何和一次失败播报相同量级：估计时延约 1.4s，丢到 1.5s 尾部后
        proc 仍小于 align。旧逻辑会在 arm_voiced 仍为 0 时结束预热。
        """
        delay = 23000
        n = delay + CALIB_MIN_SAMPLES + 400
        ref = _noise(n, 0.12, 51)
        mic = _delayed(ref, delay, 0.65)
        tap, core = _new_core(_energy_vad())
        core.feed_mic(mic)
        tap.push_samples(ref)
        for _ in range(80):
            core.process_available()
            if core.calibrated and core._aec_samples > MAX_WARMUP_AEC_SAMPLES:
                break
        self.assertTrue(core.calibrated)
        self.assertGreater(core.align, MAX_WARMUP_AEC_SAMPLES)
        self.assertLess(core._mic_proc, core.align)
        self.assertIsNone(core._arm_until)
        self.assertEqual(0, core._arm_voiced)
        self.assertFalse(core.triggered)

    def test_bargein_still_fires_after_aligned_reference_is_seen(self):
        """预热改成等待真实参考之后，对齐完成仍要能自然打断，不能靠关掉打断保播放。"""
        delay = 20000
        echo_n = delay + CALIB_MIN_SAMPLES + 8000
        ref = _noise(echo_n, 0.12, 52)
        mic = _delayed(ref, delay, 0.6)
        tap, core = _new_core(_energy_vad())
        self.assertIsNone(_feed(core, tap, mic, ref, chunk=4096))
        self.assertTrue(core.calibrated)
        self.assertIsNotNone(core._arm_until)
        self.assertGreaterEqual(core._arm_voiced, ARM_SAMPLES)
        speech_n = int(0.45 * SR)
        speech_ref = _noise(speech_n, 0.08, 53)
        speech_mic = _delayed(speech_ref, min(delay, speech_n // 5), 0.15) + _noise(speech_n, 0.55, 54)
        self.assertEqual("bargein", _feed(core, tap, speech_mic, speech_ref, chunk=speech_n))


class AbortMessageRegressionTest(unittest.TestCase):
    def test_protocol_abort_type_is_abort(self):
        class Capture(Protocol):
            def __init__(self):
                super().__init__()
                self.sent = []

            async def send_text(self, message):
                self.sent.append(json.loads(message))

        protocol = Capture()
        protocol.session_id = "session"

        async def run():
            await protocol.send_abort_speaking(AbortReason.ABORT)
            await protocol.send_start_listening(
                __import__(
                    "core.services.protocols.typing", fromlist=["ListeningMode"]
                ).ListeningMode.MANUAL
            )

        asyncio.run(run())
        self.assertEqual(
            {"session_id": "session", "type": "abort"},
            protocol.sent[0],
        )
        self.assertEqual("listen", protocol.sent[1]["type"])
        self.assertEqual("start", protocol.sent[1]["state"])
        self.assertEqual("manual", protocol.sent[1]["mode"])
        self.assertNotEqual("wake_word_detected", protocol.sent[0]["type"])

    def test_relisten_keeps_abort_and_single_app_loop_send(self):
        text = (ROOT / "core" / "xiaozhi.py").read_text(encoding="utf-8")
        start = text.index("async def _relisten_after_bargein")
        end = text.index("async def _play_notify")
        body = text[start:end]
        stop_fn = text[text.index("async def _abort_and_stop_speaker"):text.index("async def start_wakeup_session")]
        self.assertIn("_abort_and_stop_speaker", body)
        self.assertIn("send_abort_speaking(AbortReason.ABORT)", stop_fn)
        self.assertIn("stop_device_audio", stop_fn)
        self.assertIn("stop_playing", stop_fn)
        self.assertLess(
            stop_fn.index("send_abort_speaking(AbortReason.ABORT)"),
            stop_fn.index("await speaker.stop_device_audio()"),
        )
        self.assertNotIn("WAKE_WORD_DETECTED", body + stop_fn)
        self.assertNotIn("run_coroutine_threadsafe", body)
        self.assertEqual(2, body.count("_await_on_app_loop"))
        self.assertLess(body.index("output_stream.stop_stream"), body.index("_abort_and_stop_speaker"))
        preroll_at = body.index("set_speech_frames(preroll)")
        listen_at = body.index("send_start_listening(ListeningMode.MANUAL)")
        self.assertLess(body.index("_abort_and_stop_speaker"), preroll_at)
        self.assertLess(preroll_at, listen_at)
        self.assertLess(
            body.index("Barge-in abort sent"),
            body.index("Barge-in preroll queued"),
        )
        self.assertLess(
            body.index("Barge-in preroll queued"),
            body.index("Barge-in listen sent"),
        )

    def test_duplicate_tts_stop_does_not_raise(self):
        """两份 set_result(True) 入队时，后一份必须看到已完成，而不是抛 InvalidStateError。"""
        loop = asyncio.new_event_loop()
        errors = []

        def _finish(fut):
            if not fut.done():
                fut.set_result(True)

        async def scenario():
            fut = loop.create_future()
            loop.call_soon(_finish, fut)
            loop.call_soon(_finish, fut)
            loop.call_soon(lambda: errors.append("scheduled-ok"))
            result = await fut
            await asyncio.sleep(0)
            return result

        try:
            loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx.get("message", "")))
            result = loop.run_until_complete(scenario())
        finally:
            loop.close()
        self.assertTrue(result)
        self.assertNotIn("invalid state", " ".join(errors).lower())
        self.assertTrue(any("scheduled-ok" == item for item in errors))
        stop = text_of_stop_handler()
        self.assertIn("if not fut.done()", stop)
        self.assertNotIn(".set_result, True", stop)
        self.assertIn("fut.set_result(True)", stop)


def text_of_stop_handler() -> str:
    text = (ROOT / "core" / "xiaozhi.py").read_text(encoding="utf-8")
    return text[text.index("def _handle_tts_stop"):text.index("def _handle_stt_message")]


if __name__ == "__main__":
    unittest.main()
