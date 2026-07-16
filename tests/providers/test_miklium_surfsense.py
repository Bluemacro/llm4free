"""Tests for the no-browser free providers Miklium and Surfsense.

These mock ``curl_cffi.requests.Session`` so no network calls are made. They
verify the request payloads/headers and the response parsing (non-stream for
Miklium, SSE for Surfsense).
"""

import json as json_module
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from llm4free.llm.miklium import Miklium
from llm4free.llm.surfsense import Surfsense
from tests.providers.utils import FakeResp


class _FakeStreamResponse(FakeResp):
    """A minimal curl_cffi-like streaming response with line iteration."""

    def __init__(self, status_code: int = 200, lines: Optional[List[str]] = None):
        super().__init__(status_code=status_code, text="\n".join(lines or []))
        self.reason = "OK"
        self._lines = lines or []

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line.encode("utf-8")

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")


class TestMiklium(unittest.TestCase):
    def test_non_stream_request_and_parse(self):
        captured: Dict[str, Any] = {}

        def fake_post(url, *, headers=None, json=None, stream=None, timeout=None, proxies=None, **kwargs):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["stream"] = stream
            return _FakeStreamResponse(
                status_code=200, lines=[json_module.dumps({"success": "true", "response": "Hi there!"})]
            )

        provider = Miklium()
        with patch.object(provider.session, "post", side_effect=fake_post):
            resp = provider.chat.completions.create(
                model="miklium",
                messages=[{"role": "user", "content": "hello"}],
            )

        self.assertEqual(
            captured["url"], "https://miklium.vercel.app/api/chatbot"
        )
        self.assertIn("hello", captured["json"]["message"])
        self.assertEqual(captured["json"]["personality"], "miklium")
        self.assertEqual(captured["json"]["response_stacking"], 4)
        self.assertIn("miklium.vercel.app", captured["headers"]["origin"])
        self.assertEqual(resp.choices[0].message.content, "Hi there!")

    def test_failure_raises(self):
        def fake_post(url, **kwargs):
            return _FakeStreamResponse(
                status_code=200, lines=[json_module.dumps({"success": "false", "response": "nope"})]
            )

        provider = Miklium()
        with patch.object(provider.session, "post", side_effect=fake_post):
            with self.assertRaises((RuntimeError, IOError)):
                provider.chat.completions.create(
                    model="miklium", messages=[{"role": "user", "content": "hi"}]
                )


class TestSurfsense(unittest.TestCase):
    def test_non_stream_sse(self):
        lines = [
            "data: " + json_module.dumps({"type": "text-delta", "delta": "Hello "}),
            "data: " + json_module.dumps({"type": "text-delta", "delta": "world"}),
            "data: [DONE]",
        ]
        captured: Dict[str, Any] = {}

        def fake_post(url, *, headers=None, json=None, stream=None, timeout=None, proxies=None, **kwargs):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _FakeStreamResponse(status_code=200, lines=lines)

        provider = Surfsense()
        with patch.object(provider.session, "post", side_effect=fake_post):
            resp = provider.chat.completions.create(
                model="gpt-o4-mini-no-login",
                messages=[{"role": "user", "content": "hi"}],
            )

        self.assertEqual(
            captured["url"], "https://api.surfsense.com/api/v1/public/anon-chat/stream"
        )
        self.assertEqual(captured["json"]["model_slug"], "gpt-o4-mini-no-login")
        self.assertEqual(resp.choices[0].message.content, "Hello world")

    def test_alias_resolution(self):
        captured: Dict[str, Any] = {}
        lines = ["data: " + json_module.dumps({"type": "text-delta", "delta": "x"}), "data: [DONE]"]

        def fake_post(url, *, headers=None, json=None, stream=None, timeout=None, proxies=None, **kwargs):
            captured["json"] = json
            return _FakeStreamResponse(status_code=200, lines=lines)

        provider = Surfsense()
        with patch.object(provider.session, "post", side_effect=fake_post):
            provider.chat.completions.create(
                model="o4-mini", messages=[{"role": "user", "content": "hi"}]
            )
        self.assertEqual(captured["json"]["model_slug"], "gpt-o4-mini-no-login")

    def test_streaming_yields_chunks(self):
        lines = [
            "data: " + json_module.dumps({"type": "text-delta", "delta": "a"}),
            "data: " + json_module.dumps({"type": "text-delta", "delta": "b"}),
            "data: [DONE]",
        ]

        def fake_post(url, **kwargs):
            return _FakeStreamResponse(status_code=200, lines=lines)

        provider = Surfsense()
        with patch.object(provider.session, "post", side_effect=fake_post):
            chunks = list(
                provider.chat.completions.create(
                    model="gpt-o4-mini-no-login",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
            )
        contents = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
        self.assertEqual("".join(contents), "ab")
        # final chunk should carry finish_reason="stop"
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
