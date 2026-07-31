import json
import re
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Union, cast

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
    count_tokens,
)

BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"


def _extract_answer(text: str) -> str:
    text = re.sub(r"\[REASON_START\].*?\[REASON_DONE\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[REASON_DONE\]", "", text)
    text = re.sub(r"\[REASON_START\]", "", text)
    return text.strip()


_SKIP_MARKERS = {
    "[ANSWER_START]",
    "[ANSWER_DONE]",
    "[ANSWER_END]",
    "[RELATE_Q_START]",
    "[RELATE_Q_DONE]",
    "[SOURCE_START]",
    "[SOURCE_DONE]",
}


def _parse_sse_payload(data: str) -> tuple:
    """Parse a single ``data:`` payload.

    Returns ``(content, error)`` where at most one is non-empty. Content lines may
    be raw text, JSON strings, or JSON objects carrying ``text``/``content``/``c``.
    Error objects (``error`` + ``message``) surface their message.
    """
    if data.startswith('"') and data.endswith('"'):
        try:
            data = json.loads(data)
        except Exception:
            pass
    if isinstance(data, str) and data.startswith("{") and data.endswith("}"):
        try:
            obj = json.loads(data)
        except Exception:
            return data, None
        if not isinstance(obj, dict):
            return None, None
        if obj.get("error") is not None and obj.get("message"):
            return None, str(obj["message"])
        for key in ("text", "content", "c"):
            value = obj.get(key)
            if isinstance(value, str):
                return value, None
        return None, None
    return (str(data) if data else None), None


class Completions(BaseCompletions):
    def __init__(self, client: "HeckAI"):
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
        model = self._client.convert_model_name(model)

        question = ""
        for msg in messages:
            if msg.get("role") == "user":
                question = msg.get("content", "")

        payload: Dict[str, Any] = {
            "model": model,
            "question": question,
            "language": self._client.language,
            "sessionId": self._client.session_id,
            "previousQuestion": None,
            "previousAnswer": None,
            "imgUrls": [],
            "superSmartMode": False,
        }

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload, timeout, proxies)
        else:
            return self._create_non_stream(
                request_id, created_time, model, payload, timeout, proxies
            )

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
            self._client._ensure_session()
            payload["sessionId"] = self._client.session_id
            response = self._client.session.post(
                self._client.url,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies or getattr(self._client, "proxies", None),  # ty:ignore[invalid-argument-type]
            )
            response.raise_for_status()

            error_msg: Optional[str] = None
            error_mode = False
            buffering_reason = False

            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                stripped = data.strip()

                if stripped == "[ERROR]":
                    error_mode = True
                    continue
                if error_mode:
                    try:
                        obj = json.loads(data)
                        if isinstance(obj, dict) and obj.get("message"):
                            error_msg = str(obj["message"])
                    except Exception:
                        pass
                    continue
                if stripped == "[REASON_START]":
                    buffering_reason = True
                    continue
                if stripped == "[REASON_DONE]":
                    buffering_reason = False
                    continue
                if buffering_reason:
                    continue
                if stripped in _SKIP_MARKERS or (
                    stripped.startswith("[") and stripped.endswith("]")
                ):
                    continue

                content, err = _parse_sse_payload(data)
                if err:
                    error_msg = err
                    continue
                if not content:
                    continue
                content = _extract_answer(content)
                if not content:
                    continue
                delta = ChoiceDelta(content=content)
                choice = Choice(index=0, delta=delta, finish_reason=None)
                yield ChatCompletionChunk(
                    id=request_id,
                    choices=[choice],
                    created=created_time,
                    model=model,
                )

            if error_msg:
                raise IOError(f"HeckAI request failed: {error_msg}")

            delta = ChoiceDelta(content=None)
            choice = Choice(index=0, delta=delta, finish_reason="stop")
            yield ChatCompletionChunk(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
            )
        except CurlError as e:
            print(f"{RED}Error during HeckAI stream request: {e}{RESET}")
            raise IOError(f"HeckAI request failed: {e}") from e

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
            self._client._ensure_session()
            payload["sessionId"] = self._client.session_id
            answer_parts: List[str] = []
            error_msg: Optional[str] = None
            error_mode = False
            buffering_reason = False
            response = self._client.session.post(
                self._client.url,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies or getattr(self._client, "proxies", None),  # ty:ignore[invalid-argument-type]
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                stripped = data.strip()

                if stripped == "[ERROR]":
                    error_mode = True
                    continue
                if error_mode:
                    try:
                        obj = json.loads(data)
                        if isinstance(obj, dict) and obj.get("message"):
                            error_msg = str(obj["message"])
                    except Exception:
                        pass
                    continue
                if stripped == "[REASON_START]":
                    buffering_reason = True
                    continue
                if stripped == "[REASON_DONE]":
                    buffering_reason = False
                    continue
                if buffering_reason:
                    continue
                if stripped in _SKIP_MARKERS or (
                    stripped.startswith("[") and stripped.endswith("]")
                ):
                    continue

                content, err = _parse_sse_payload(data)
                if err:
                    error_msg = err
                    continue
                if content:
                    answer_parts.append(content)

            if error_msg:
                raise IOError(f"HeckAI request failed: {error_msg}")

            full_text = _extract_answer("".join(answer_parts))

            prompt_tokens = count_tokens(payload.get("question", ""))
            completion_tokens = count_tokens(full_text)
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
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
            print(f"{RED}Error during HeckAI non-stream request: {e}{RESET}")
            raise IOError(f"HeckAI request failed: {e}") from e


class Chat(BaseChat):
    def __init__(self, client: "HeckAI"):
        self.completions = Completions(client)


class HeckAI(OpenAICompatibleProvider):
    required_auth = False
    AVAILABLE_MODELS = [
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
        "tencent/hy3-preview",
        "qwen/qwen3.7-plus",
        "stepfun/step-3.7-flash",
        "google/gemini-3.1-flash-lite",
        "google/gemini-3-flash-preview",
        "openai/gpt-5.4-mini",
        "minimax/minimax-m3",
    ]

    def __init__(self, timeout: int = 30, language: str = "English"):
        self.timeout = timeout
        self.language = language
        self.url = "https://api.heckai.weight-wave.com/api/ha/v1/chat"
        self.session_url = "https://api.heckai.weight-wave.com/api/ha/v1/session/create"
        self.session_id = str(uuid.uuid4())
        self._session_ready = False

        agent = LitAgent()
        self.headers = {
            "User-Agent": agent.random(),
            "Content-Type": "application/json",
            "Origin": "https://heck.ai",
            "Referer": "https://heck.ai/",
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.chat = Chat(self)

    def _ensure_session(self) -> None:
        if self._session_ready:
            return
        self._session_ready = True
        try:
            response = self.session.post(
                self.session_url, json={"title": "Chat"}, timeout=self.timeout
            )
            data = response.json()
            session_id = data.get("id")
            if session_id:
                self.session_id = session_id
        except Exception:
            pass

    def convert_model_name(self, model: str) -> str:
        if model in self.AVAILABLE_MODELS:
            return model
        for available_model in self.AVAILABLE_MODELS:
            if model.lower() in available_model.lower():
                return available_model
        print(
            f"{BOLD}Warning: Model '{model}' not found, using default 'openai/gpt-5.4-mini'{RESET}"
        )
        return "openai/gpt-5.4-mini"

    @property
    def models(self) -> SimpleModelList:
        return SimpleModelList(type(self).AVAILABLE_MODELS)


if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in HeckAI.AVAILABLE_MODELS:
        try:
            client = HeckAI(timeout=60)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Say 'Hello' in one word"},
                ],
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
            print(f"{model:<50} {'✗':<10} {str(e)}")
