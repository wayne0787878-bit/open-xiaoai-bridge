"""XiaoZhi protocol client - handles WebSocket connection to XiaoZhi server.

This module is responsible for:
- Connecting to XiaoZhi WebSocket server
- Sending/receiving audio and text messages
- Protocol-level message handling
- Audio codec management and stream control
- JSON message parsing (TTS/STT/LLM)
- Wakeup session: VAD-driven listen/stop cycle

The main application flow is managed by app.py.
"""

import asyncio
import json
import os
import threading
import time

import open_xiaoai_server

from core.ref import get_audio_codec, get_speaker, get_vad, set_speech_frames
from core.services.protocols.protocol import Protocol
from core.services.protocols.typing import (
    AbortReason,
    DeviceState,
    ListeningMode,
)
from core.utils.config import ConfigManager
from core.utils.logger import logger

_NOTIFY_SOUND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "sounds", "tts_notify.mp3",
)


def _load_notify_sound() -> bytes | None:
    """Decode tts_notify.mp3 to PCM at startup."""
    if not os.path.isfile(_NOTIFY_SOUND_PATH):
        return None
    try:
        with open(_NOTIFY_SOUND_PATH, "rb") as f:
            mp3_data = f.read()
        return open_xiaoai_server.decode_audio(mp3_data, format="mp3", sample_rate=24000)
    except Exception:
        return None


_NOTIFY_PCM = _load_notify_sound()

# 打断监视器灰度开关（环境变量 XIAOZHI_BARGEIN）：
#   未设/0/off → off（不装配监视器，行为与现行版一致）
#   observe/log → observe（监视器运行并记录判定，不真打断）
#   1/true/yes/on → on（命中后发 {"type":"abort"} 并续听；服务端忽略 wake_word_detected）
_BARGEIN_ENV = os.environ.get("XIAOZHI_BARGEIN", "").strip().lower()
if _BARGEIN_ENV in ("1", "true", "yes", "on"):
    BARGEIN_MODE = "on"
elif _BARGEIN_ENV in ("observe", "log"):
    BARGEIN_MODE = "observe"
else:
    BARGEIN_MODE = "off"


class XiaoZhi:
    """XiaoZhi protocol client."""

    _instance = None

    @classmethod
    def instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = XiaoZhi()
        return cls._instance

    def __init__(self):
        """Initialize XiaoZhi client."""
        if XiaoZhi._instance is not None:
            raise Exception("XiaoZhi is singleton, use instance() to get instance")
        XiaoZhi._instance = self

        self.config = ConfigManager.instance()

        # Protocol
        self.protocol = None

        # References (set by MainApp)
        self._app = None
        self._audio_codec = None

        # Wakeup session state
        self._session_loop: asyncio.AbstractEventLoop | None = None
        self._vad_future: asyncio.Future | None = None
        self._tts_stop_future: asyncio.Future | None = None
        self._is_first_round = False

        # Barge-in monitor state（灰度开关关闭时保持 None/False，不装配）
        self._bargein_monitor = None
        self._bargein_future: asyncio.Future | None = None
        self._in_wakeup_session = False
        logger.info(
            f"[XiaoZhi] Barge-in monitor mode: {BARGEIN_MODE} "
            f"(XIAOZHI_BARGEIN={os.environ.get('XIAOZHI_BARGEIN') or 'not set'})"
        )

    def set_app(self, app):
        """Set reference to main app."""
        self._app = app

    def set_audio_codec(self, codec):
        """Set audio codec reference."""
        self._audio_codec = codec

    @property
    def device_state(self):
        """Proxy device state from app."""
        if self._app:
            return self._app.device_state
        return None

    def set_device_state(self, state):
        """Set device state and manage audio streams accordingly."""
        if not self._app:
            return
        self._app.device_state = state

        from core.services.audio.vad import VAD
        VAD.pause()

        if self._audio_codec:
            self._audio_codec.stop_streams()

        if state == DeviceState.LISTENING:
            if self._audio_codec:
                if self._audio_codec.output_stream.is_active():
                    self._audio_codec.output_stream.stop_stream()
                if not self._audio_codec.input_stream.is_active():
                    self._audio_codec.input_stream.start_stream()
        elif state == DeviceState.SPEAKING:
            if self._audio_codec:
                if self._audio_codec.input_stream.is_active():
                    self._audio_codec.input_stream.stop_stream()
                if not self._audio_codec.output_stream.is_active():
                    self._audio_codec.output_stream.start_stream()

        # 打断监视器（灰度）：仅在 SPEAKING 期运行，其余状态停。
        # 监视流与主 input_stream 独立，不受上面的流管理影响。
        monitor = getattr(self, "_bargein_monitor", None)
        if monitor is not None:
            try:
                if state == DeviceState.SPEAKING:
                    monitor.start()
                else:
                    monitor.stop()
            except Exception:
                pass

    # Connection management

    async def connect(self):
        """Connect to XiaoZhi server."""
        from core.services.protocols.websocket_protocol import WebsocketProtocol

        self.protocol = WebsocketProtocol()

        # Wire up callbacks directly
        self.protocol.on_network_error = self._on_network_error
        self.protocol.on_incoming_audio = self._on_incoming_audio
        self.protocol.on_incoming_json = self._on_incoming_json
        self.protocol.on_audio_channel_opened = self._on_audio_channel_opened
        self.protocol.on_audio_channel_closed = self._on_audio_channel_closed

        return await self.protocol.connect()

    async def disconnect(self):
        """Disconnect from XiaoZhi server."""
        if self.protocol:
            await self.protocol.close_audio_channel()

    def is_connected(self) -> bool:
        """Check if connected to XiaoZhi server."""
        return self.protocol is not None and self.protocol.is_audio_channel_opened()

    # Audio initialization

    def init_audio(self):
        """Initialize audio codec and start audio input trigger."""
        try:
            from core.services.audio.codec import AudioCodec

            self._audio_codec = AudioCodec()
        except Exception as e:
            logger.warning(f"[XiaoZhi] Failed to initialize audio: {e}")
            return

        # 打断监视器装配（灰度开关非 off 时）；装配失败只禁用监视器，
        # 不影响主流程。
        if BARGEIN_MODE != "off":
            try:
                from core.services.audio.bargein import BargeInMonitor

                monitor = BargeInMonitor(
                    on_trigger=self._on_bargein_detected,
                    observe=(BARGEIN_MODE == "observe"),
                )
                stream = self._audio_codec.create_monitor_stream()
                if stream is not None:
                    monitor.attach_stream(stream)
                    self._audio_codec.set_reference_sink(monitor.feed_reference_pcm)
                    self._bargein_monitor = monitor
                    logger.info(
                        f"[XiaoZhi] Barge-in monitor ready (mode={BARGEIN_MODE})"
                    )
                else:
                    logger.warning(
                        "[XiaoZhi] Barge-in monitor stream unavailable, monitor disabled"
                    )
            except Exception as e:
                logger.warning(f"[XiaoZhi] Failed to set up barge-in monitor: {e}")
                self._bargein_monitor = None

        threading.Thread(target=self._audio_input_trigger, daemon=True).start()

    def _on_bargein_detected(self, preroll: bytes):
        """打断监视器命中回调（监视器线程调用）。

        仅在唤醒会话正等待 TTS 结束时才兑现：把消回声预缓冲经
        session loop 交给 _wait_tts_stop_or_bargein。其余时机
        （无会话/未在等待）忽略，不改变任何现有行为。
        """
        future = self._bargein_future
        accepted = (
            self._in_wakeup_session
            and self._session_loop is not None
            and future is not None
            and not future.done()
        )
        logger.info(
            "[XiaoZhi] Barge-in trigger: "
            f"preroll_bytes={len(preroll)} accepted={accepted} "
            f"in_session={self._in_wakeup_session}",
            module="XiaoZhi",
        )
        if accepted:
            self._session_loop.call_soon_threadsafe(future.set_result, preroll)

    # Protocol callbacks

    def _on_network_error(self, message):
        """Handle network error."""
        self.set_device_state(DeviceState.IDLE)
        if self.device_state != DeviceState.CONNECTING:
            self.set_device_state(DeviceState.IDLE)
            asyncio.run_coroutine_threadsafe(
                self.disconnect(), self._app.loop
            )

    def _on_incoming_audio(self, data):
        """Handle incoming audio."""
        if self.device_state == DeviceState.SPEAKING:
            self._audio_codec.write_audio(data)

    def _on_incoming_json(self, json_data):
        """Handle incoming JSON message."""
        try:
            if not json_data:
                return

            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            msg_type = data.get("type", "")

            if msg_type == "tts":
                self._handle_tts_message(data)
            elif msg_type == "stt":
                self._handle_stt_message(data)
            elif msg_type == "llm":
                self._handle_llm_message(data)
        except Exception as exc:
            logger.error(
                f"[XiaoZhi] Failed to handle incoming json: {type(exc).__name__}: {exc}"
            )

    def _on_audio_channel_opened(self):
        """Handle audio channel opened."""
        self.set_device_state(DeviceState.IDLE)

    def _on_audio_channel_closed(self):
        """Handle audio channel closed."""
        self.set_device_state(DeviceState.IDLE)
        if self._audio_codec:
            self._audio_codec.stop_streams()

    # JSON message handlers

    def _handle_tts_message(self, data):
        """Handle TTS message."""
        state = data.get("state", "")
        if state == "start":
            self._app.schedule(lambda: self._handle_tts_start())
        elif state == "stop":
            self._app.schedule(lambda: self._handle_tts_stop())
        elif state == "sentence_start":
            text = data.get("text", "")
            if text:
                logger.ai_response(text, module="XiaoZhi")
                self._app.schedule(lambda: self._app.set_chat_message("assistant", text))

    def _handle_tts_start(self):
        """Handle TTS start."""
        if self.device_state in [DeviceState.IDLE, DeviceState.LISTENING]:
            self.set_device_state(DeviceState.SPEAKING)

    def _handle_tts_stop(self):
        """Handle TTS stop.

        完成回调必须在 session loop 上再次检查 done()。stop 可能来两份
        （一句结束、abort 再来一次），两份都在入队前看到 future 未完成，
        后一份 set_result(True) 会在回调里抛 InvalidStateError。
        """
        future = self._tts_stop_future
        loop = self._session_loop
        if future is None or loop is None:
            return

        def _finish(fut=future):
            if not fut.done():
                fut.set_result(True)

        loop.call_soon_threadsafe(_finish)

    def _handle_stt_message(self, data):
        """Handle STT message."""
        text = data.get("text", "")
        if text:
            logger.user_speech(text, module="XiaoZhi")
            self._app.schedule(lambda: self._app.set_chat_message("user", text))

    def _handle_llm_message(self, data):
        """Handle LLM message."""
        text = data.get("text", "")
        if text:
            logger.ai_response(text, module="XiaoZhi")
            self._app.schedule(lambda: self._app.set_chat_message("assistant", text))
        emotion = data.get("emotion", "")
        if emotion:
            self._app.schedule(lambda: self._app.set_emotion(emotion))

    # Audio input handling

    def handle_input_audio(self):
        """Handle audio input - read and send to server."""
        if self.device_state != DeviceState.LISTENING:
            return

        encoded_data = self._audio_codec.read_audio()
        if encoded_data and self.is_connected():
            asyncio.run_coroutine_threadsafe(
                self.send_audio(encoded_data), self._app.loop
            )

    def _audio_input_trigger(self):
        """Trigger audio input event periodically."""
        while self._app and self._app.running:
            if self._audio_codec and self._audio_codec.input_stream.is_active():
                from core.services.protocols.typing import EventType
                self._app.events[EventType.AUDIO_INPUT_READY_EVENT].set()
            import time
            time.sleep(0.01)

    # User actions

    def start_listening(self):
        """Start listening."""
        self._app.schedule(self._start_listening_impl)

    def _start_listening_impl(self):
        """Start listening implementation."""
        self.set_device_state(DeviceState.IDLE)
        asyncio.run_coroutine_threadsafe(
            self.send_abort_speaking(AbortReason.ABORT),
            self._app.loop,
        )
        asyncio.run_coroutine_threadsafe(
            self.send_start_listening(ListeningMode.MANUAL), self._app.loop
        )
        self.set_device_state(DeviceState.LISTENING)

    def stop_listening(self):
        """Stop listening."""
        self._app.schedule(self._stop_listening_impl)

    def _stop_listening_impl(self):
        """Stop listening implementation."""
        asyncio.run_coroutine_threadsafe(
            self.send_stop_listening(), self._app.loop
        )
        self.set_device_state(DeviceState.IDLE)

    def abort_speaking(self, reason):
        """Abort speaking."""
        self.set_device_state(DeviceState.IDLE)
        asyncio.run_coroutine_threadsafe(
            self.send_abort_speaking(AbortReason.ABORT),
            self._app.loop,
        )

    # Send methods

    async def send_audio(self, frames):
        """Send audio frames."""
        if self.protocol:
            await self.protocol.send_audio(frames)

    async def send_text(self, text: str):
        """Send text message."""
        if self.protocol and self.protocol.is_audio_channel_opened():
            try:
                message = {
                    "type": "listen",
                    "mode": "manual",
                    "state": "detect",
                    "text": text,
                }
                await self.protocol.send_text(json.dumps(message))
                logger.info(f"[XiaoZhi] Text sent: {text}")
            except Exception as e:
                logger.error(f"[XiaoZhi] Failed to send text: {e}")
        else:
            logger.warning("[XiaoZhi] Not connected, cannot send text")

    async def send_start_listening(self, mode):
        """Send start listening command."""
        if self.protocol:
            await self.protocol.send_start_listening(mode)

    async def send_stop_listening(self):
        """Send stop listening command."""
        if self.protocol:
            await self.protocol.send_stop_listening()

    async def send_abort_speaking(self, reason):
        """Send abort speaking command."""
        if self.protocol:
            await self.protocol.send_abort_speaking(reason)

    async def _abort_and_stop_speaker(self):
        """在主事件循环上发 abort，并停掉设备上已经缓冲的播放。

        只发 websocket abort 不会杀掉 aplay / miplayer 里已经排队的音频，
        音箱会把缓冲播完。停播走 SpeakerManager.stop_device_audio，
        与唤醒打断同一条本地通道，不另开一条 websocket 发送。
        """
        await self.send_abort_speaking(AbortReason.ABORT)
        speaker = get_speaker()
        if speaker is not None:
            await speaker.stop_device_audio()
            return
        await open_xiaoai_server.stop_playing()

    # Wakeup session

    def stop_wakeup_session(self):
        """Stop any active wakeup session and abort server-side audio."""
        if self._session_loop and self._vad_future and not self._vad_future.done():
            self._session_loop.call_soon_threadsafe(self._vad_future.cancel)
        if self._session_loop and self._tts_stop_future and not self._tts_stop_future.done():
            self._session_loop.call_soon_threadsafe(self._tts_stop_future.cancel)
        if self._session_loop and self._bargein_future and not self._bargein_future.done():
            self._session_loop.call_soon_threadsafe(self._bargein_future.cancel)

        self.set_device_state(DeviceState.IDLE)

        vad = get_vad()
        if vad:
            vad.pause()

        if self.is_connected() and self._session_loop:
            asyncio.run_coroutine_threadsafe(
                self.send_abort_speaking(AbortReason.ABORT), self._session_loop
            )

    async def start_wakeup_session(self):
        """Start a VAD-driven wakeup session: notify → listen → silence → stop."""
        if not self.protocol:
            logger.warning("XiaoZhi is not ready, skip wakeup session", module="XiaoZhi")
            return

        self._session_loop = asyncio.get_running_loop()
        vad = get_vad()
        codec = get_audio_codec()
        speaker = get_speaker()
        self._in_wakeup_session = True

        try:
            while True:
                self.set_device_state(DeviceState.IDLE)
                await self.send_abort_speaking(AbortReason.ABORT)

                if self._is_first_round:
                    self._is_first_round = False
                    await self._play_notify(speaker)

                # Wait for speech
                vad.resume("speech")
                result = await self._wait_vad_event(
                    timeout=self.config.get_app_config("wakeup.timeout", 20)
                )
                if result is None:
                    self.set_device_state(DeviceState.IDLE)
                    logger.info("Wakeup timeout, exit listening", module="XiaoZhi")
                    after_wakeup = self.config.get_app_config("wakeup.after_wakeup")
                    if after_wakeup and speaker:
                        await after_wakeup(speaker, source="xiaozhi")
                    return

                event_type, event_data = result
                if event_type != "speech":
                    logger.debug(f"Expected speech, got {event_type}", module="XiaoZhi")
                    return

                # Speech detected — send buffered audio and start listening
                logger.debug(
                    f"VAD detected speech, buffer size: {len(event_data or b'')}",
                    module="XiaoZhi",
                )
                set_speech_frames(event_data)
                if codec:
                    codec.input_stream.start_stream()
                await self.send_start_listening(ListeningMode.MANUAL)
                self.set_device_state(DeviceState.LISTENING)

                # 对话轮次：等待静音 → 停止监听 → 等待 TTS 结束。
                # 播报期间若被真人打断（barge-in，灰度开启时），
                # _wait_tts_stop_or_bargein 内部已完成 abort + 续听，
                # 本轮直接回到等待静音，不退出会话；barge-in 关闭时
                # 该等待与原 _wait_tts_stop 行为一致，内层只走一遍。
                tts_finished = False
                while True:
                    # Wait for silence
                    vad.resume("silence")
                    result = await self._wait_vad_event()
                    if result:
                        event_type, _ = result
                        logger.debug(f"VAD detected silence, stop listening", module="XiaoZhi")

                    # Prepare TTS stop future before sending stop_listening,
                    # so we don't miss the tts_stop event.
                    self._tts_stop_future = self._session_loop.create_future()

                    await self.send_stop_listening()
                    self.set_device_state(DeviceState.IDLE)

                    # Wait for TTS to finish (or barge-in), then start next round
                    outcome = await self._wait_tts_stop_or_bargein(timeout=30)
                    if outcome == "bargein":
                        continue
                    tts_finished = outcome == "tts"
                    break
                if not tts_finished:
                    break
        except asyncio.CancelledError:
            logger.debug("Wakeup session cancelled", module="XiaoZhi")
            raise
        finally:
            self._in_wakeup_session = False
            if vad:
                vad.pause()
            self.set_device_state(DeviceState.IDLE)
            self._vad_future = None
            self._tts_stop_future = None
            self._bargein_future = None

    async def _wait_vad_event(self, timeout=None):
        """Wait for a VAD event (speech/silence)."""
        from core.wakeup_session import EventManager

        self._vad_future = self._session_loop.create_future()
        original_on_speech = EventManager.on_speech
        original_on_silence = EventManager.on_silence

        def _on_speech(speech_buffer):
            if self._vad_future and not self._vad_future.done():
                self._session_loop.call_soon_threadsafe(
                    self._vad_future.set_result, ("speech", speech_buffer)
                )

        def _on_silence():
            if self._vad_future and not self._vad_future.done():
                self._session_loop.call_soon_threadsafe(
                    self._vad_future.set_result, ("silence", None)
                )

        EventManager.on_speech = _on_speech
        EventManager.on_silence = _on_silence

        try:
            return await asyncio.wait_for(self._vad_future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            EventManager.on_speech = original_on_speech
            EventManager.on_silence = original_on_silence
            self._vad_future = None

    async def _wait_tts_stop(self, timeout=None):
        """Wait for TTS playback to finish."""
        if not self._tts_stop_future:
            self._tts_stop_future = self._session_loop.create_future()
        try:
            return await asyncio.wait_for(self._tts_stop_future, timeout=timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self._tts_stop_future = None

    async def _wait_tts_stop_or_bargein(self, timeout=None):
        """等待 TTS 结束或播报中打断，返回 "tts" / "timeout" / "bargein"。

        barge-in 未装配（灰度关闭）时与 _wait_tts_stop 行为一致，
        仅返回形式不同。返回 "bargein" 时已完成 abort + 续听
        （见 _relisten_after_bargein），调用方直接回到等待静音。
        """
        if not self._tts_stop_future:
            self._tts_stop_future = self._session_loop.create_future()
        tts_future = self._tts_stop_future

        if self._bargein_monitor is None:
            try:
                await asyncio.wait_for(tts_future, timeout=timeout)
                return "tts"
            except asyncio.TimeoutError:
                return "timeout"
            finally:
                if self._tts_stop_future is tts_future:
                    self._tts_stop_future = None

        self._bargein_future = self._session_loop.create_future()
        bargein_future = self._bargein_future
        try:
            done, _ = await asyncio.wait(
                {tts_future, bargein_future},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if bargein_future in done and not bargein_future.cancelled():
                await self._relisten_after_bargein(
                    bargein_future.result(), tts_future
                )
                return "bargein"
            if tts_future in done:
                if tts_future.cancelled():
                    # 与 _wait_tts_stop 的 wait_for 语义对齐：
                    # 外部 stop 取消 future 时按会话取消处理
                    raise asyncio.CancelledError()
                return "tts"
            return "timeout"
        finally:
            if self._tts_stop_future is tts_future:
                self._tts_stop_future = None
            if self._bargein_future is bargein_future:
                self._bargein_future = None
            if not bargein_future.done():
                bargein_future.cancel()

    async def _await_on_app_loop(self, coro):
        """在 MainApp.loop 上执行协程并等待其完成。

        websocket 连接归 MainApp.loop 所有（协议的音频发送都在该
        loop 上）；从 session loop 直接 await 协议发送，会与 app
        loop 的音频发送并发写同一条连接，帧可能损坏——2026-11-23
        on 档实测：连续第二次打断后服务端以 1007（invalid frame
        payload）关闭连接。本方法与既有用户动作路径（start/stop/
        abort 均经 run_coroutine_threadsafe 投递 app loop）对齐。
        """
        loop = self._app.loop if self._app else None
        if loop is None or not loop.is_running():
            return await coro
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return await asyncio.wrap_future(future)

    async def _relisten_after_bargein(self, preroll: bytes, tts_future):
        """打断命中后的处理：中止播报 → 等 tts stop → 带预缓冲续听。

        - abort 用 AbortReason.ABORT：协议层把 reason 直接写进消息
          type 字段，本机小智服务端只认 {"type": "abort"}，收到
          {"type": "wake_word_detected"} 只会记「未知类型」并忽略
          （2026-11-23 17:08 服务端日志实测）——曾导致 tts stop 等待
          烧满 3s 超时、relisten 完成时用户短句早已说完；
        - 等待被中止 TTS 的 tts stop（短超时兜底），并消费掉它，
          避免迟到的 stop 误触发下一轮等待；
        - 消回声预缓冲经 set_speech_frames 作为新一轮开头送出，
          与 VAD 唤醒时的语音缓冲走同一条通道。
        """
        t0 = time.monotonic()
        logger.info(
            "[XiaoZhi] Barge-in relisten begin: "
            f"preroll_bytes={len(preroll)}",
            module="XiaoZhi",
        )
        codec = get_audio_codec()
        if codec is not None and getattr(codec, "output_stream", None) is not None:
            try:
                if codec.output_stream.is_active():
                    codec.output_stream.stop_stream()
            except Exception as exc:
                logger.warning(
                    f"[XiaoZhi] Barge-in output stop failed: {type(exc).__name__}",
                    module="XiaoZhi",
                )
        await self._await_on_app_loop(self._abort_and_stop_speaker())
        logger.info(
            "[XiaoZhi] Barge-in abort sent: "
            f"type={AbortReason.ABORT} dt_ms={(time.monotonic() - t0) * 1000:.0f}",
            module="XiaoZhi",
        )
        if tts_future is not None and not tts_future.done():
            try:
                await asyncio.wait_for(asyncio.shield(tts_future), timeout=3)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        logger.info(
            "[XiaoZhi] Barge-in tts-stop wait done: "
            f"dt_ms={(time.monotonic() - t0) * 1000:.0f}",
            module="XiaoZhi",
        )
        if preroll:
            set_speech_frames(preroll)
            logger.info(
                "[XiaoZhi] Barge-in preroll queued: "
                f"bytes={len(preroll)} dt_ms={(time.monotonic() - t0) * 1000:.0f}",
                module="XiaoZhi",
            )
        codec = get_audio_codec()
        if codec:
            codec.input_stream.start_stream()
        await self._await_on_app_loop(
            self.send_start_listening(ListeningMode.MANUAL)
        )
        self.set_device_state(DeviceState.LISTENING)
        logger.info(
            "[XiaoZhi] Barge-in listen sent: type=listen state=start "
            f"mode=manual dt_ms={(time.monotonic() - t0) * 1000:.0f}",
            module="XiaoZhi",
        )

    async def _play_notify(self, speaker):
        """Play the listening-ready notification sound."""
        if not _NOTIFY_PCM or not speaker:
            return
        try:
            await speaker.play(buffer=_NOTIFY_PCM)
            duration = len(_NOTIFY_PCM) / (24000 * 2)
            await asyncio.sleep(duration)
        except Exception as exc:
            logger.debug(f"Notify sound error: {exc}", module="XiaoZhi")

    # Shutdown

    def shutdown(self):
        """Shutdown XiaoZhi - close audio and disconnect."""
        if self._audio_codec:
            self._audio_codec.close()

        if self._app and self._app.loop:
            asyncio.run_coroutine_threadsafe(
                self.disconnect(), self._app.loop
            )
