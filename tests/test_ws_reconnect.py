"""离线验证小智 WebSocket 失败摘要和单飞重连。不访问网络。"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")

if "websockets" not in sys.modules:
    sys.modules["websockets"] = types.ModuleType("websockets")


class _Config:
    def get_config(self, key, default=None):
        if str(key).endswith("WEBSOCKET_URL"):
            return "wss://secret.example/socket?token=abc"
        if str(key).endswith("WEBSOCKET_ACCESS_TOKEN"):
            return "secret-token"
        return default

    def get_client_id(self):
        return "client"
    def get_device_id(self):
        return "device"


class _ConfigManager:
    @classmethod
    def instance(cls):
        return _Config()


_config_mod = types.ModuleType("core.utils.config")
_config_mod.ConfigManager = _ConfigManager
sys.modules["core.utils.config"] = _config_mod

_native = types.ModuleType("open_xiaoai_server")
_native.decode_audio = lambda *_args, **_kwargs: b""
sys.modules["open_xiaoai_server"] = _native

from core.services.protocols.websocket_protocol import WebsocketProtocol  # noqa: E402
from core.xiaozhi import (  # noqa: E402
    GuardedWebsocketProtocol,
    XiaoZhi,
    redact_failure_text,
    safe_ws_failure,
)
import websockets  # noqa: E402


class HandshakeError(Exception):
    def __init__(self):
        super().__init__("rejected wss://secret.example/socket?token=abc")
        self.response = types.SimpleNamespace(status_code=401)


class RemoteClose(Exception):
    def __init__(self):
        super().__init__("closed wss://secret.example/socket?token=abc")
        self.rcvd = types.SimpleNamespace(code=1000)


class Script:
    def __init__(self):
        self.fail_remaining = 0
        self.fail_exc = HandshakeError()
        self.hello = True
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.sockets = []
        self.hold = 0.0


class FakeWS:
    def __init__(self, script: Script):
        self.script = script
        self.queue = asyncio.Queue()
        self.closed = False

    async def send(self, message):
        data = json.loads(message)
        if data.get("type") == "hello" and self.script.hello:
            await self.queue.put(
                json.dumps({"type": "hello", "transport": "websocket"})
            )

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def ping(self):
        return None


def _install(script: Script):
    async def connect(*_args, **_kwargs):
        script.calls += 1
        script.active += 1
        script.max_active = max(script.max_active, script.active)
        try:
            if script.hold:
                await asyncio.sleep(script.hold)
            if script.fail_remaining > 0:
                script.fail_remaining -= 1
                raise script.fail_exc
            ws = FakeWS(script)
            script.sockets.append(ws)
            return ws
        finally:
            script.active -= 1

    websockets.connect = connect


def _protocol(script: Script) -> GuardedWebsocketProtocol:
    _install(script)
    proto = GuardedWebsocketProtocol()
    proto._hello_timeout = 0.05
    proto._retry_base_s = 0.01
    return proto


def _failure_logs(cm) -> str:
    kept = []
    for line in cm.output:
        if any(
            token in line
            for token in (
                "connect failed",
                "channel closed",
                "reader stopped",
                "close failed",
                "heartbeat failed",
                "network error",
            )
        ):
            kept.append(line)
    return "\n".join(kept)


class SafeFailureTest(unittest.TestCase):
    def test_summary_keeps_codes_and_drops_text(self):
        text = safe_ws_failure("connect", HandshakeError(), 12)
        self.assertIn("phase=connect", text)
        self.assertIn("exc=HandshakeError", text)
        self.assertIn("http=401", text)
        self.assertIn("elapsed_ms=12", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("wss://", text)
        closed = safe_ws_failure("reader", RemoteClose())
        self.assertIn("close=1000", closed)
        self.assertNotIn("token", closed)
        leaked = redact_failure_text(
            "phase=connect wss://secret.example/socket?token=abc exc=OSError errno=111"
        )
        self.assertEqual("phase=connect exc=OSError errno=111", leaked)
        self.assertNotIn("secret", leaked)


class ReconnectStateTest(unittest.IsolatedAsyncioTestCase):
    async def test_failed_handshake_records_status_without_secret(self):
        script = Script()
        script.fail_remaining = 1
        proto = _protocol(script)
        with self.assertLogs("xiaozhi", level="WARNING") as cm:
            ok = await proto.connect()
        self.assertFalse(ok)
        self.assertIn("http=401", proto._last_failure)
        self.assertNotIn("secret", _failure_logs(cm))
        self.assertNotIn("wss://", _failure_logs(cm))
        self.assertIsNone(proto._reconnect_task)

    async def test_hello_timeout_then_one_reconnect_succeeds(self):
        script = Script()
        script.hello = False
        proto = _protocol(script)
        proto.on_network_error = lambda _message: proto.ensure_reconnect()
        with self.assertLogs("xiaozhi", level="INFO") as cm:
            ok = await proto.connect()
            self.assertFalse(ok)
            self.assertIn("phase=hello_timeout", proto._last_failure)
            script.hello = True
            task = proto._reconnect_task
            self.assertIsNotNone(task)
            await asyncio.wait_for(task, timeout=1)
        self.assertTrue(proto.connected)
        self.assertEqual(0, proto._reconnect_attempts)
        self.assertEqual(1, script.max_active)
        self.assertNotIn("secret", _failure_logs(cm))
        proto.ensure_reconnect()
        proto.ensure_reconnect()
        self.assertTrue(proto._reconnect_task is None or proto._reconnect_task.done())

    async def test_duplicate_failure_callbacks_share_one_task(self):
        script = Script()
        script.fail_remaining = 5
        proto = _protocol(script)
        proto.on_network_error = lambda _message: proto.ensure_reconnect()
        await proto.connect()
        proto.ensure_reconnect()
        proto.ensure_reconnect()
        task = proto._reconnect_task
        self.assertIsNotNone(task)
        proto.ensure_reconnect()
        self.assertIs(task, proto._reconnect_task)
        await proto.close_audio_channel()
        self.assertTrue(task.cancelled() or task.done())

    async def test_remote_close_reconnects_once_and_hides_reason(self):
        script = Script()
        proto = _protocol(script)
        proto.on_network_error = lambda _message: proto.ensure_reconnect()
        with self.assertLogs("xiaozhi", level="INFO") as cm:
            self.assertTrue(await proto.connect())
            script.fail_remaining = 100
            await script.sockets[0].queue.put(RemoteClose())
            await asyncio.sleep(0.05)
            self.assertIsNotNone(proto._reconnect_task)
            first = proto._reconnect_task
            proto.ensure_reconnect()
            self.assertIs(first, proto._reconnect_task)
            await proto.close_audio_channel()
        text = _failure_logs(cm)
        self.assertIn("close=1000", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("wss://", text)

    async def test_close_cancels_reconnect_and_handler(self):
        script = Script()
        script.fail_remaining = 10
        proto = _protocol(script)
        proto.on_network_error = lambda _message: proto.ensure_reconnect()
        await proto.connect()
        task = proto._reconnect_task
        handler_before = proto._handler_task
        await proto.close_audio_channel()
        self.assertTrue(task.cancelled() or task.done())
        self.assertTrue(proto._reconnect_stop)
        if handler_before is not None:
            self.assertTrue(handler_before.cancelled() or handler_before.done())
        calls = script.calls
        await asyncio.sleep(0.05)
        self.assertEqual(calls, script.calls)
        proto.ensure_reconnect()
        self.assertTrue(proto._reconnect_task is None or proto._reconnect_task.done())

    async def test_parallel_connect_does_not_overlap(self):
        script = Script()
        script.hold = 0.05
        proto = _protocol(script)
        await asyncio.gather(proto.connect(), proto.connect())
        self.assertEqual(1, script.max_active)
        self.assertTrue(proto.connected)
        await proto.close_audio_channel()
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        self.assertEqual([], pending)

    async def test_mounted_xiaozhi_file_owns_reconnect(self):
        script = Script()
        script.fail_remaining = 1
        _install(script)
        client = XiaoZhi.__new__(XiaoZhi)
        client._app = None
        client._audio_codec = None
        client.protocol = None
        with self.assertLogs("xiaozhi", level="WARNING") as cm:
            ok = await client.connect()
        self.assertFalse(ok)
        proto = client.protocol
        self.assertIsInstance(proto, GuardedWebsocketProtocol)
        self.assertIsInstance(proto, WebsocketProtocol)
        self.assertEqual("xiaozhi.py", Path(type(proto).connect.__code__.co_filename).name)
        self.assertEqual(
            "websocket_protocol.py",
            Path(WebsocketProtocol.close_audio_channel.__code__.co_filename).name,
        )
        stock_source = Path(WebsocketProtocol.close_audio_channel.__code__.co_filename).read_text(
            encoding="utf-8"
        )
        self.assertIn("通道被关闭", stock_source)
        self.assertNotIn("ensure_reconnect", stock_source)
        text = _failure_logs(cm)
        self.assertIn("http=401", text)
        self.assertIn("exc=HandshakeError", text)
        for secret in ("secret", "wss://", "Bearer", "token"):
            self.assertNotIn(secret, text)
        task = proto._reconnect_task
        self.assertIsNotNone(task)
        proto.ensure_reconnect()
        self.assertIs(task, proto._reconnect_task)
        await client.disconnect()
        self.assertTrue(task.cancelled() or task.done())
        calls = script.calls
        await asyncio.sleep(0.05)
        self.assertEqual(calls, script.calls)
        pending = [
            item
            for item in asyncio.all_tasks()
            if item is not asyncio.current_task() and not item.done()
        ]
        self.assertEqual([], pending)


if __name__ == "__main__":
    unittest.main()
