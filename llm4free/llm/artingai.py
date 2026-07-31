import base64
import hashlib
import hmac
import json
import struct
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

from curl_cffi import CurlError, requests

from llm4free.litagent import LitAgent
from llm4free.llm.base import (
    BaseChat,
    BaseCompletions,
    OpenAICompatibleProvider,
    SimpleModelList,
)
from llm4free.llm.utils import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    CompletionUsage,
    format_prompt,
)


class Completions(BaseCompletions):
    def __init__(self, client: "ArtingAI"):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        question = format_prompt(messages, add_special_tokens=True)

        model = self._client.convert_model_name(model)

        payload = {
            "generation_type": model,
            "task_type": "ai-chat",
            "session_id": str(uuid.uuid4()),
            "stream": True,
            "files": [],
            "text": question,
        }

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload, timeout, proxies)
        else:
            return self._create_non_stream(
                request_id, created_time, model, payload, timeout, proxies
            )

    def _post(
        self,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
    ):
        self._client._ensure_ready()
        body = json.dumps(payload, separators=(",", ":"))
        headers = self._client._signed_headers(
            "POST",
            self._client.api_path,
            body,
            "text/event-stream, application/json, text/plain, */*",
        )
        return self._client.session.post(
            self._client.url,
            data=body,
            headers=headers,
            stream=True,
            timeout=timeout or self._client.timeout,
            proxies=cast(Any, proxies or getattr(self._client, "proxies", None)),
            impersonate="chrome110",
        )

    def _iter_text(self, response):
        for chunk in response.iter_content(chunk_size=None):
            if not chunk:
                continue
            if isinstance(chunk, bytes):
                try:
                    chunk = chunk.decode("utf-8", errors="replace")
                except Exception:
                    continue
            if not chunk:
                continue
            yield chunk

    def _create_stream(
        self,
        request_id: str,
        created_time: int,
        model: str,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            response = self._post(payload, timeout, proxies)
            response.raise_for_status()

            it = self._iter_text(response)
            buffered = []
            first = next(it, None)
            if first is None:
                delta = ChoiceDelta(content=None)
                choice = Choice(index=0, delta=delta, finish_reason="stop")
                yield ChatCompletionChunk(
                    id=request_id, choices=[choice], created=created_time, model=model
                )
                return

            buffered.append(first)
            joined = first
            if first.lstrip().startswith("{"):
                while len(joined) < 8192:
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    joined += nxt
                    buffered.append(nxt)
                    if "}" in joined:
                        try:
                            json.loads(joined)
                            break
                        except Exception:
                            continue
                error = self._client._error_message(joined)
                if error is not None:
                    raise IOError(f"ArtingAI request failed: {error}")

            for text in buffered:
                delta = ChoiceDelta(content=text)
                choice = Choice(index=0, delta=delta, finish_reason=None)
                yield ChatCompletionChunk(
                    id=request_id, choices=[choice], created=created_time, model=model
                )
            for text in it:
                delta = ChoiceDelta(content=text)
                choice = Choice(index=0, delta=delta, finish_reason=None)
                yield ChatCompletionChunk(
                    id=request_id, choices=[choice], created=created_time, model=model
                )

            delta = ChoiceDelta(content=None)
            choice = Choice(index=0, delta=delta, finish_reason="stop")
            yield ChatCompletionChunk(
                id=request_id, choices=[choice], created=created_time, model=model
            )
        except CurlError as e:
            raise IOError(f"ArtingAI request failed: {e}") from e

    def _create_non_stream(
        self,
        request_id: str,
        created_time: int,
        model: str,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
    ) -> ChatCompletion:
        try:
            response = self._post(payload, timeout, proxies)
            response.raise_for_status()

            full_text = "".join(self._iter_text(response))
            error = self._client._error_message(full_text)
            if error is not None:
                raise IOError(f"ArtingAI request failed: {error}")

            prompt_tokens = len(payload.get("text", "").split())
            completion_tokens = len(full_text.split())
            total_tokens = prompt_tokens + completion_tokens
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            message = ChatCompletionMessage(role="assistant", content=full_text)
            choice = Choice(index=0, message=message, finish_reason="stop")
            return ChatCompletion(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
                usage=usage,
            )
        except IOError:
            raise
        except Exception as e:
            raise IOError(f"ArtingAI request failed: {e}") from e


class Chat(BaseChat):
    def __init__(self, client: "ArtingAI"):
        self.completions = Completions(client)


_RSA_PRIVATE_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDCNDgxYbbANun4
AsXaB8c12HtEPz9w7dnjGwWrWWXv5ozXc9Wxb86k0ka5I1j5hKftRcDHYGcdXGBc
PQFK1ReOsosOsoJPZkSuwYONk39O3CHaRls+Fy93+HKlEafafhGdFOeEMAoMty4m
atKi7XVuVryh1NyPBAoSwLzGTl31VNkfLlX8+pxG8LMDSH6G78HivdnhGXotGmlX
iBXZ3KzYYT6cjJpapQyF7Dzg2k+5vxJ5DOSQFGahz0X6aD2ITVBg9YhtUcv/wOWg
bdlz5vKCJhswEGu0q8jcWrKyNnt4j1dg5smzFF7RxN879w/PuByCciFDk/Th55YU
nbX3YRcjAgMBAAECggEACRX8sQwYyupmzOSC/DM+oJLLGu4jROB5QApXcWUCg2mp
GjrAUIe3cSJTY723H8loj+MiZKbqfBxXzR5KOAWQj2u5XMvWPpUSzo2diYX76qUa
7QWQpu7abjBQsM9gJ9/xFAUn2xk3nHuvgZBK4DY1jHQraXOLiXdkv0GG3iJJfBJS
I2+KgB3R9sAj8h3jTo0ezPzANwOUkZDwIxhvoaczc8rFP0FAWHcXuCFiu2Y60U3A
xObq5Wr+Oq45+zYoxlX0nDOYtwGJWtVzYLwJ8RQIjcUGh9ksZeNfoJkAdK9QF+tm
ZquA4MqDfK0g5ziBz5mgdvzCcCWXSHik6+iBPxBvjQKBgQDjMcRjGmIUAYCZ47Yb
cfCmyWLpbUg6vmTpDkl6+fF5rs0U/aQtrD/5NdaBWO6kLRJQZjqcEo2ZJJKwVj3r
Pr89H0nxAsF9XzBS45zsg4JBQ4eGzAal4YYSadiT+2th3E4AAx95lS/S0zEYEQjX
eIs8N8BWYAJjfDx2yiJnkNtvTwKBgQDa06fSUvaJlljCeqIzgrmOObCrJV2zuXKU
TrUIffgMfgo+oXfsvVvjzR5/K5wOcV36OQf8O/eEspd3QWH4TCNOuzspqFrCO4cp
pceBCzlge4/Ymaona/FOazhEVuFpL8ISqRwxPe9SaTNdt1RiN3et78fskmJfgWvx
FnVdjGmF7QKBgDSTSb0dV+EFT/tMxNGpFmWiaO9XyMU/Vh7QnZSFzqm4F+FpqNqg
59UF7nPUXrVDcN+GKL4BVR9BZWjFLGMKDDtayEOrvZcDti0YWzIoZLYxqGU7RbaR
b/NG50WngvwMfUhncJs0OPLyyIOnPYKPdLkkta/HXAYls+BRepC45u7lAoGAMOVi
doi7Nfs2Uh585+2p8LHLXDK5QVOK2sDLit468u+m8l+6IFgflENdMSVZdZC3YxYj
RqVPpYMSfT9K2OSKbyk/CwvnW8dZaGD2t0r+wyRY/Bk6AB0Kim9C32Jac9qMDwdi
mU4xj8SaCbLRVDD4uRD/J0l+WcDdkb1m9ERPv/ECgYEAx0dWw6J+XY031bEPW+WQ
jVKvM4OyxgXbeWPhCWgm+C0P2DB8uPKybPVwmf+8/uijiUkJXLJCfXWAMUaZl/1L
fWutjh1nNWQb3rKN2H7GvuOiJ6NLIcA8oUhZoJ34vcVkEeCRRbs3ZMBhGbc9Aoru
QvFQg5UJ0NMN5gLjnvqVmD0=
-----END PRIVATE KEY-----"""


def _der_length(data: bytes, i: int) -> Tuple[int, int]:
    length = data[i]
    i += 1
    if length & 0x80:
        num = length & 0x7F
        length = int.from_bytes(data[i : i + num], "big")
        i += num
    return length, i


def _parse_pkcs8_rsa(pem: str) -> Tuple[int, int, int]:
    body = b"".join(
        line.encode() for line in pem.splitlines() if not line.startswith("--")
    )
    data = base64.b64decode(body)
    i = 0
    assert data[i] == 0x30
    i += 1
    _, i = _der_length(data, i)
    assert data[i] == 0x02
    i += 1
    vlen, i = _der_length(data, i)
    i += vlen
    assert data[i] == 0x30
    i += 1
    alen, i = _der_length(data, i)
    i += alen
    assert data[i] == 0x04
    i += 1
    olen, i = _der_length(data, i)
    pk = data[i : i + olen]
    j = 0
    assert pk[j] == 0x30
    j += 1
    _, j = _der_length(pk, j)

    def next_int() -> int:
        nonlocal j
        assert pk[j] == 0x02
        j += 1
        length, j = _der_length(pk, j)
        value = pk[j : j + length]
        j += length
        return int.from_bytes(value, "big")

    next_int()
    return next_int(), next_int(), next_int()


def _mgf1(seed: bytes, length: int) -> bytes:
    out = b""
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(seed + struct.pack(">I", counter)).digest()
        counter += 1
    return out[:length]


def _rsa_oaep_sha256_decrypt(
    ciphertext: bytes, n: int, d: int, modulus_len: int
) -> str:
    hash_len = 32
    message = pow(int.from_bytes(ciphertext, "big"), d, n)
    encoded = message.to_bytes(modulus_len, "big")
    masked_seed = encoded[1 : hash_len + 1]
    masked_db = encoded[hash_len + 1 :]
    seed_mask = _mgf1(masked_db, hash_len)
    seed = bytes(a ^ b for a, b in zip(masked_seed, seed_mask))
    db_mask = _mgf1(seed, len(masked_db))
    db = bytes(a ^ b for a, b in zip(masked_db, db_mask))
    label_hash = hashlib.sha256(b"").digest()
    if db[:hash_len] != label_hash:
        raise ValueError("invalid OAEP label hash")
    i = hash_len
    while i < len(db) and db[i] == 0:
        i += 1
    if i >= len(db) or db[i] != 0x01:
        raise ValueError("invalid OAEP padding")
    return db[i + 1 :].decode("utf-8")


class ArtingAI(OpenAICompatibleProvider):
    required_auth = False
    AVAILABLE_MODELS = [
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-4o-mini",
        "o4-mini",
        "gemini-2.5-pro",
        "gemini-3-pro-preview",
        "deepseek-chat",
        "deepseek-reasoner",
    ]

    _BOOTSTRAP_TOKEN = "4xH7LpQ8KjF2aR9cVbN3mW6yT1uE5iA0sD8fG7hJ9kL2zXArting"
    _API_PATH = "/api/aigc/comprehensive/chat/create-task"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.base_url = "https://arting.ai"
        self.api_path = self._API_PATH
        self.url = self.base_url + self.api_path
        self.proxies = {}

        self._identity = str(uuid.uuid4())
        self._secret: Optional[str] = None
        self._ready = False

        self._rsa_n, self._rsa_e, self._rsa_d = _parse_pkcs8_rsa(_RSA_PRIVATE_KEY_PEM)
        self._rsa_modulus_len = (self._rsa_n.bit_length() + 7) // 8

        agent = LitAgent()
        self.headers = {
            "User-Agent": agent.random(),
            "Accept-Language": "en-GB,en;q=0.9",
            "Referer": "https://arting.ai/ai-chat",
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.chat = Chat(self)

    def convert_model_name(self, model: str) -> str:
        if model in self.AVAILABLE_MODELS:
            return model
        for m in self.AVAILABLE_MODELS:
            if model.lower() in m.lower():
                return m
        return "gpt-5"

    @property
    def models(self) -> SimpleModelList:
        return SimpleModelList(type(self).AVAILABLE_MODELS)

    def _sign(self, method: str, path: str, body: str, timestamp: str) -> str:
        if not self._secret:
            raise IOError("ArtingAI identity bootstrap failed")
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        message = f"{timestamp}\n{method.upper()}\n{path}\n{body_hash}\n{self._identity}"
        return hmac.new(self._secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    def _signed_headers(self, method: str, path: str, body: str, accept: str) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        return {
            "Authorization": self._identity,
            "X-Identity-Id": self._identity,
            "X-Timestamp": timestamp,
            "X-Signature": self._sign(method, path, body, timestamp),
            "Content-Type": "application/json",
            "Accept": accept,
        }

    def _bootstrap(self) -> None:
        response = self.session.get(
            self.base_url + "/nav-logo.webp",
            headers={
                "X-Bootstrap-Token": self._BOOTSTRAP_TOKEN,
                "X-Identity-Id": self._identity,
            },
        )
        encrypted = response.headers.get("x-next-secret")
        if not encrypted:
            raise IOError("ArtingAI bootstrap failed: missing x-next-secret")
        self._secret = _rsa_oaep_sha256_decrypt(
            base64.b64decode(encrypted), self._rsa_n, self._rsa_d, self._rsa_modulus_len
        )

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self._ready = True
        try:
            self._bootstrap()
        except Exception:
            self._secret = None
            raise IOError("ArtingAI identity bootstrap failed") from None
        try:
            for path in ("/api/aigc/config", "/api/aigc/comprehensive/chat/free-usage"):
                timestamp = str(int(time.time()))
                headers = {
                    "Authorization": self._identity,
                    "X-Identity-Id": self._identity,
                    "X-Timestamp": timestamp,
                    "X-Signature": self._sign("GET", path, "", timestamp),
                    "Accept": "application/json, text/plain, */*",
                }
                self.session.get(self.base_url + path, headers=headers)
            self.session.post(
                self.base_url + "/api/aigc/track/log",
                json={"signals": ["webdriver"]},
                headers={"X-Identity-Id": self._identity, "Content-Type": "application/json"},
            )
        except Exception:
            pass

    @staticmethod
    def _error_message(text: str) -> Optional[str]:
        try:
            obj = json.loads(text)
        except Exception:
            return None
        if isinstance(obj, dict) and "code" in obj:
            code = obj.get("code")
            if code in (0, "0"):
                return None
            return str(obj.get("message") or f"error code {code}")
        return None


if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in ArtingAI.AVAILABLE_MODELS:
        try:
            client = ArtingAI(timeout=60)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'Hello' in one word"}],
                stream=False,
            )
            if (
                isinstance(response, ChatCompletion)
                and response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                status = "✓"
                display_text = response.choices[0].message.content.strip()
                display_text = display_text[:50] + "..." if len(display_text) > 50 else display_text
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"{model:<50} {'✗':<10} {str(e)[:80]}")
