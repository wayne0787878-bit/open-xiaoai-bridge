import asyncio
import inspect
import json
import re
import time

import websockets

from core.ref import get_xiaozhi
from core.services.protocols.protocol import Protocol
from core.services.protocols.typing import DeviceState
from core.utils.config import ConfigManager
from core.utils.logger import logger

_SAFE_FAILURE_KEYS = {
    "phase": re.compile(r"[a-z_]{1,32}"),
    "exc": re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,64}"),
    "http": re.compile(r"\d{3}"),
    "close": re.compile(r"\d{1,5}"),
    "errno": re.compile(r"-?\d{1,6}"),
    "elapsed_ms": re.compile(r"\d{1,8}"),
}


def safe_ws_failure(phase: str, exc: BaseException | None = None, elapsed_ms: int | None = None) -> str:
    """白名单连接失败摘要。不包含异常文本、地址、query 或 header。"""
    parts = [f"phase={phase if _SAFE_FAILURE_KEYS['phase'].fullmatch(phase or '') else 'unknown'}"]
    if isinstance(elapsed_ms, int) and elapsed_ms >= 0:
        parts.append(f"elapsed_ms={elapsed_ms}")
    if exc is not None:
        name = type(exc).__name__
        if _SAFE_FAILURE_KEYS["exc"].fullmatch(name):
            parts.append(f"exc={name}")
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if not isinstance(status, int) and response is not None:
            status = getattr(response, "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            parts.append(f"http={status}")
        close = getattr(exc, "code", None)
        for frame_name in ("rcvd", "sent"):
            if isinstance(close, int):
                break
            frame = getattr(exc, frame_name, None)
            if frame is not None:
                close = getattr(frame, "code", None)
        if isinstance(close, int) and 0 <= close <= 4999:
            parts.append(f"close={close}")
        err_no = getattr(exc, "errno", None)
        if isinstance(err_no, int):
            parts.append(f"errno={err_no}")
    return " ".join(parts)


def redact_failure_text(message: object) -> str:
    """只保留已知安全字段。无法识别时不回显原文。"""
    if not isinstance(message, str):
        return "phase=unspecified"
    kept = []
    for token in message.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        rule = _SAFE_FAILURE_KEYS.get(key)
        if rule is not None and rule.fullmatch(value):
            kept.append(f"{key}={value}")
    return " ".join(kept) if kept else "phase=unspecified"


class WebsocketProtocol(Protocol):
    def __init__(self):
        super().__init__()
        # 获取配置管理器实例
        self.config = ConfigManager.instance()
        self.websocket = None
        self.server_sample_rate = 24000
        self.server_frame_duration = 60
        self.server_frame_size = int(
            self.server_sample_rate * (self.server_frame_duration / 1000)
        )
        self.connected = False
        self.hello_received = None  # 初始化时先设为 None
        self.WEBSOCKET_URL = self.config.get_config("NETWORK.WEBSOCKET_URL")
        self.WEBSOCKET_ACCESS_TOKEN = self.config.get_config(
            "NETWORK.WEBSOCKET_ACCESS_TOKEN"
        )
        self.CLIENT_ID = self.config.get_client_id()
        self.DEVICE_ID = self.config.get_device_id()
        self._handler_task = None
        self._reconnect_task = None
        self._reconnect_attempts = 0
        self._reconnect_stop = False
        self._in_reconnect_loop = False
        self._suppress_reader_reconnect = False
        self._is_heartbeat_running = False
        self._hello_timeout = 10.0
        self._retry_base_s = None
        self._last_failure = "phase=unknown"
        self._connect_lock = asyncio.Lock()

    async def _close_websocket(self):
        websocket = self.websocket
        self.websocket = None
        self.connected = False
        if not websocket:
            return
        try:
            await websocket.close()
        except Exception as exc:
            logger.warning(
                "[xiaozhi-esp32-server] close failed "
                f"{safe_ws_failure('close', exc)}",
                module="xiaozhi-esp32-server",
            )

    async def _invoke_callback(self, callback, *args):
        """Run sync or async callbacks transparently."""
        if not callback:
            return None

        result = callback(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def connect(self) -> bool:
        """连接到WebSocket服务器"""
        async with self._connect_lock:
            return await self._connect_locked()

    async def _connect_locked(self) -> bool:
        started = time.perf_counter()
        try:
            await self._close_websocket()
            self.WEBSOCKET_URL = self.config.get_config("NETWORK.WEBSOCKET_URL")
            self.WEBSOCKET_ACCESS_TOKEN = self.config.get_config(
                "NETWORK.WEBSOCKET_ACCESS_TOKEN"
            )
            self.CLIENT_ID = self.config.get_client_id()
            self.DEVICE_ID = self.config.get_device_id()

            # 在连接时创建 Event，确保在正确的事件循环中
            self.hello_received = asyncio.Event()

            # 配置连接
            headers = {
                "Authorization": f"Bearer {self.WEBSOCKET_ACCESS_TOKEN}",
                "Protocol-Version": "1",
                "Device-Id": self.DEVICE_ID,  # 获取设备MAC地址
                "Client-Id": self.CLIENT_ID,
            }

            # 建立WebSocket连接
            self.websocket = await websockets.connect(
                uri=self.WEBSOCKET_URL, additional_headers=headers
            )

            await self._cancel_handler()

            # 启动消息处理循环
            self._handler_task = asyncio.create_task(self._message_handler())

            # 发送客户端hello消息
            hello_message = {
                "type": "hello",
                "version": 1,
                "transport": "websocket",
                "audio_params": {
                    "format": "opus",
                    "sample_rate": 16000,
                    "channels": 1,
                    "frame_duration": 60,
                },
            }
            await self.send_text(json.dumps(hello_message))

            # 等待服务器hello响应
            try:
                await asyncio.wait_for(
                    self.hello_received.wait(), timeout=self._hello_timeout
                )
                self.connected = True
                self._last_failure = "phase=ok"
                logger.info(f"[xiaozhi-esp32-server] 连接 {self.WEBSOCKET_URL} 成功")
                return True
            except asyncio.TimeoutError:
                return await self._fail_connect("hello_timeout", None, started)
        except Exception as exc:
            return await self._fail_connect("connect", exc, started)

    async def _fail_connect(self, phase: str, exc: BaseException | None, started: float) -> bool:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        summary = safe_ws_failure(phase, exc, elapsed_ms)
        self._last_failure = summary
        logger.warning(
            f"[xiaozhi-esp32-server] connect failed {summary}",
            module="xiaozhi-esp32-server",
        )
        await self._cancel_handler()
        await self._close_websocket()
        if self.on_network_error and not self._in_reconnect_loop:
            self.on_network_error(summary)
        return False

    async def _cancel_handler(self):
        task = self._handler_task
        self._handler_task = None
        if task is None or task.done():
            return
        self._suppress_reader_reconnect = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(
                "[xiaozhi-esp32-server] reader cancel failed "
                f"{safe_ws_failure('reader_cancel', exc)}",
                module="xiaozhi-esp32-server",
            )
        finally:
            self._suppress_reader_reconnect = False

    async def _message_handler(self):
        """处理接收到的WebSocket消息"""
        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        session_id = data.get("session_id")
                        if session_id and session_id != self.session_id:
                            self.session_id = session_id
                            logger.info(
                                f"[xiaozhi-esp32-server] Updated session_id: {session_id}"
                            )
                        msg_type = data.get("type")
                        if msg_type == "hello":
                            await self._handle_server_hello(data)
                        else:
                            if self.on_incoming_json:
                                self.on_incoming_json(data)
                    except json.JSONDecodeError:
                        pass
                elif self.on_incoming_audio:
                    self.on_incoming_audio(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._suppress_reader_reconnect:
                return
            summary = safe_ws_failure("reader", exc)
            self._last_failure = summary
            logger.warning(
                f"[xiaozhi-esp32-server] reader stopped {summary}",
                module="xiaozhi-esp32-server",
            )
            self.connected = False
            if self.on_audio_channel_closed:
                await self._invoke_callback(self.on_audio_channel_closed)
            if not self._reconnect_stop:
                self.ensure_reconnect()

    async def send_audio(self, frames: list[bytes]):
        """发送音频数据"""
        if not self.is_audio_channel_opened():  # 使用已有的 is_connected 方法
            return

        try:
            for frame in frames:
                await self.websocket.send(frame)
        except Exception as exc:
            logger.error(
                f"[xiaozhi-esp32-server] Failed to send audio: {type(exc).__name__}: {exc}"
            )

    async def send_text(self, message: str):
        """发送文本消息"""
        if self.websocket:
            try:
                await self.websocket.send(message)
            except Exception as e:
                raise e

    def is_audio_channel_opened(self) -> bool:
        """检查音频通道是否打开"""
        return self.websocket is not None and self.connected

    async def open_audio_channel(self):
        if not self._is_heartbeat_running:
            self._is_heartbeat_running = True
            asyncio.create_task(self.heartbeat())
        await self.connect()

    async def _handle_server_hello(self, data: dict):
        """处理服务器的 hello 消息

        解析服务器返回的 hello 消息，设置相关参数并通知音频通道已打开

        Args:
            data: 服务器返回的 hello 消息数据
        """
        try:
            # 验证传输方式
            transport = data.get("transport")
            if not transport or transport != "websocket":
                return

            # TODO 使用默认的 24k 采样率
            # xiaozhi-esp32-server 返回的参数是 16k 采样率，但实际用的是 24k 采样率

            # 获取音频参数
            # audio_params = data.get("audio_params")
            # if audio_params:
            #     # 获取服务器的采样率
            #     sample_rate = audio_params.get("sample_rate")
            #     if sample_rate:
            #         self.server_sample_rate = sample_rate
            #     frame_duration = audio_params.get("frame_duration")
            #     if frame_duration:
            #         self.server_frame_duration = frame_duration
            #     self.server_frame_size = int(
            #         self.server_sample_rate * (self.server_frame_duration / 1000)
            #     )

            # 设置 hello 接收事件
            self.hello_received.set()

            # 通知音频通道已打开
            if self.on_audio_channel_opened:
                await self._invoke_callback(self.on_audio_channel_opened)

        except Exception as exc:
            summary = safe_ws_failure("hello_parse", exc)
            self._last_failure = summary
            logger.warning(
                f"[xiaozhi-esp32-server] hello failed {summary}",
                module="xiaozhi-esp32-server",
            )
            if self.on_network_error and not self._in_reconnect_loop:
                self.on_network_error(summary)

    def _next_retry_delay(self) -> float:
        if self._retry_base_s is not None:
            return float(self._retry_base_s)
        return float(min(5 * (2 ** (self._reconnect_attempts - 1)), 60))

    def ensure_reconnect(self):
        """保证最多一个重连任务。重复的失败回调直接并入当前任务。"""
        if self._reconnect_stop or self.connected:
            return
        task = self._reconnect_task
        if task is not None and not task.done():
            return
        self._reconnect_task = asyncio.get_running_loop().create_task(
            self._reconnect_loop()
        )

    async def _reconnect_loop(self):
        self._in_reconnect_loop = True
        try:
            while not self._reconnect_stop:
                self._reconnect_attempts += 1
                delay = self._next_retry_delay()
                logger.info(
                    "[xiaozhi-esp32-server] channel closed "
                    f"retry_in_s={delay:.0f} attempt={self._reconnect_attempts} "
                    f"{self._last_failure}",
                    module="xiaozhi-esp32-server",
                )
                await asyncio.sleep(delay)
                if self._reconnect_stop:
                    return
                if await self.connect():
                    self._reconnect_attempts = 0
                    return
        except asyncio.CancelledError:
            raise
        finally:
            self._in_reconnect_loop = False
            current = asyncio.current_task()
            if self._reconnect_task is current:
                self._reconnect_task = None

    async def close_audio_channel(self):
        """关闭音频通道并停止重连。重连只由 ensure_reconnect 启动一次。"""
        self._reconnect_stop = True
        task = self._reconnect_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._cancel_handler()
        await self._close_websocket()
        if self.on_audio_channel_closed:
            await self._invoke_callback(self.on_audio_channel_closed)

    async def heartbeat(self):
        while not self._reconnect_stop:
            if self.websocket and get_xiaozhi().device_state == DeviceState.IDLE:
                try:
                    await self.websocket.ping()
                except Exception as exc:
                    summary = safe_ws_failure("heartbeat", exc)
                    self._last_failure = summary
                    logger.warning(
                        f"[xiaozhi-esp32-server] heartbeat failed {summary}",
                        module="xiaozhi-esp32-server",
                    )
                    self.ensure_reconnect()
            await asyncio.sleep(1)
