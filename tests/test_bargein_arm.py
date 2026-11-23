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
    ARM_SAMPLES,
    CALIB_MIN_SAMPLES,
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


def _feed(core: BargeInCore, tap: ReferenceTap, mic: np.ndarray, ref: np.ndarray,
          chunk: int, pause: float = 0.0):
    """按块喂入等长的麦克风/参考。返回第一次 bargein，没有则 None。"""
    if len(mic) != len(ref):
        raise AssertionError(f"mic/ref length {len(mic)} != {len(ref)}")
    pos = 0
    event = None
    while pos < len(mic):
        step = min(chunk, len(mic) - pos)
        core.feed_mic(mic[pos:pos + step])
        tap.push_samples(ref[pos:pos + step])
        if pause:
            time.sleep(pause)
        ev = core.process_available()
        pos += step
        if ev == "bargein":
            return "bargein"
        if ev is not None and event is None:
            event = ev
    return event


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
        speech_at = int(2.6 * SR)
        speech_n = int(0.45 * SR)
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
        self.assertLess(core._arm_until, speech_at)
        self.assertGreater(core._seg_end, core._arm_until)
        self.assertGreater(core._mic_proc, speech_at)
        self.assertLess(core._mic_proc, speech_at + speech_n + SR // 2)
        # 旧锚点是 tail_end + ARM（接近最新喂入 + 0.35s），会盖住这段语音。
        self.assertLess(core._arm_until, core._mic_fed - ARM_SAMPLES)

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
        total = 2 * SR
        # 落在预热之后、标定尾窗之前，快路径和慢路径看到的都是同一句。
        speech_at = int(0.90 * SR)
        speech_n = 4608
        ref = _noise(total, 0.10, 10)
        mic = _delayed(ref, delay, 0.55)
        mic = mic.copy()
        mic[speech_at:speech_at + speech_n] += _noise(speech_n, 0.50, 11)

        def run(chunk: int, pause: float):
            tap, core = _new_core(_energy_vad())
            started = time.perf_counter()
            event = _feed(core, tap, mic, ref, chunk=chunk, pause=pause)
            return event, core._arm_until, time.perf_counter() - started

        fast_event, fast_arm, fast_dt = run(total, 0.0)
        slow_event, slow_arm, slow_dt = run(1024, 0.008)
        self.assertEqual("bargein", fast_event)
        self.assertEqual("bargein", slow_event)
        self.assertIsNotNone(fast_arm)
        self.assertIsNotNone(slow_arm)
        self.assertLess(abs(fast_arm - slow_arm), AEC_BLOCK_LEN * 2)
        self.assertLess(fast_arm, speech_at)
        self.assertLess(fast_dt, 0.20)
        self.assertGreater(slow_dt, fast_dt + 0.10)

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
        self.assertIsNotNone(core._arm_until)
        self.assertGreater(core._arm_until, silence)

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
        speech_at = int(2.5 * SR)
        speech_n = int(0.45 * SR)
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
        self.assertIn("send_abort_speaking(AbortReason.ABORT)", body)
        self.assertNotIn("WAKE_WORD_DETECTED", body)
        self.assertNotIn("send_abort_speaking(AbortReason.wake_word_detected)", body)
        self.assertNotIn("run_coroutine_threadsafe", body)
        self.assertEqual(2, body.count("_await_on_app_loop"))
        abort_at = body.index("send_abort_speaking(AbortReason.ABORT)")
        preroll_at = body.index("set_speech_frames(preroll)")
        listen_at = body.index("send_start_listening(ListeningMode.MANUAL)")
        self.assertLess(abort_at, preroll_at)
        self.assertLess(preroll_at, listen_at)
        self.assertLess(
            body.index("Barge-in abort sent"),
            body.index("Barge-in preroll queued"),
        )
        self.assertLess(
            body.index("Barge-in preroll queued"),
            body.index("Barge-in listen sent"),
        )


if __name__ == "__main__":
    unittest.main()
