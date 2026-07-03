import json
import os
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Union, cast

from curl_cffi import requests

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


class Completions(BaseCompletions):
    def __init__(self, client: "Supercode"):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        if "/" in model:
            provider_prefix, model_name = model.split("/", 1)
        else:
            provider_prefix = "concentrateai"
            model_name = model

        payload: Dict[str, Any] = {
            "provider": provider_prefix,
            "model": model_name,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        payload.update(kwargs)

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(
                request_id, created_time, model, payload, timeout, proxies
            )
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
            response = self._client.session.post(
                f"{self._client.base_url}/api/ai/chat",
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=cast(Any, proxies or getattr(self._client, "proxies", None)),
                impersonate="chrome120",
            )
            if response.status_code >= 400:
                raise IOError(
                    f"Supercode stream HTTP {response.status_code}: {response.text!r}"
                )

            prompt_tokens = count_tokens(
                [msg.get("content", "") for msg in payload.get("messages", [])]
            )
            completion_tokens = 0
            total_tokens = 0
            full_text = ""

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8", errors="replace")
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ev_type = event.get("type")
                if ev_type == "text":
                    content = event.get("content", "")
                    if content:
                        full_text += content
                        completion_tokens += count_tokens(content)
                        total_tokens = prompt_tokens + completion_tokens
                        delta = ChoiceDelta(content=content)
                        choice = Choice(index=0, delta=delta, finish_reason=None)
                        yield ChatCompletionChunk(
                            id=request_id,
                            choices=[choice],
                            created=created_time,
                            model=model,
                            usage={
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens,
                            },
                        )
                elif ev_type == "reasoning":
                    content = event.get("content", "")
                    if content:
                        delta = ChoiceDelta(content=content, reasoning_content=content)
                        choice = Choice(index=0, delta=delta, finish_reason=None)
                        yield ChatCompletionChunk(
                            id=request_id,
                            choices=[choice],
                            created=created_time,
                            model=model,
                        )

            delta = ChoiceDelta(content=None)
            choice = Choice(index=0, delta=delta, finish_reason="stop")
            yield ChatCompletionChunk(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
            )
        except Exception as e:
            raise IOError(f"Supercode stream request failed: {e}") from e

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
            response = self._client.session.post(
                f"{self._client.base_url}/api/ai/chat",
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=cast(Any, proxies or getattr(self._client, "proxies", None)),
                impersonate="chrome120",
            )
            if response.status_code >= 400:
                raise IOError(
                    f"Supercode non-stream HTTP {response.status_code}: {response.text!r}"
                )

            full_text = ""
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8", errors="replace")
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "text":
                    full_text += event.get("content", "")

            prompt_tokens = count_tokens(
                [msg.get("content", "") for msg in payload.get("messages", [])]
            )
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
        except Exception as e:
            raise IOError(f"Supercode non-stream request failed: {e}") from e


class Chat(BaseChat):
    def __init__(self, client: "Supercode"):
        self.completions = Completions(client)


class Supercode(OpenAICompatibleProvider):
    """
    Supercode API - unified LLM gateway.

    Authentication is loaded from ~/.better-auth/token.json by default,
    or can be passed explicitly via the api_key parameter.
    """

    required_auth = False
    AVAILABLE_MODELS = [
        "concentrateai/deepseek-v4-flash",
        "concentrateai/glm-5.2",
        "concentrateai/glm-5.1",
        "concentrateai/kimi-k2-6",
        "concentrateai/minimax-m3",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "nvidia/minimaxai/minimax-m3",
        "nvidia/deepseek-ai/deepseek-v4-flash",
        "nvidia/meta/llama-3.3-70b-instruct",
        "openrouter/openai/gpt-oss-120b:free",
        "openrouter/deepseek/deepseek-v4-flash",
        "openrouter/minimax/minimax-m3",
        "openrouter/z-ai/glm-5.1",
        "openrouter/moonshotai/kimi-k2.6",
    ]

    def __init__(self, api_key: Optional[str] = None, timeout: int = 120, **kwargs: Any):
        self.base_url = "https://supercode-8w7e.onrender.com"
        self.timeout = timeout
        self.token = api_key or self._load_token()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.headers = headers
        self.session = requests.Session()
        self.session.headers.update(headers)
        self.chat = Chat(self)

    def _load_token(self) -> str:
        path = os.path.expanduser("~/.better-auth/token.json")
        try:
            with open(path) as f:
                data = json.load(f)
                token = data.get("access_token", "")
                if not token:
                    raise RuntimeError(
                        "No token provided and ~/.better-auth/token.json is empty."
                    )
                return token
        except FileNotFoundError:
            raise RuntimeError(
                "No token provided and ~/.better-auth/token.json not found."
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load token from {path}: {exc}"
            ) from exc

    @property
    def models(self) -> SimpleModelList:
        return SimpleModelList(type(self).AVAILABLE_MODELS)


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        try:
            client = Supercode()
        except RuntimeError as exc:
            print(f"Auth error: {exc}")
            return

        print("Available models:", client.models.list())
        print()

        try:
            response = client.chat.completions.create(
                model="concentrateai/deepseek-v4-flash",
                messages=[{"role": "user", "content": "Say 'Hello'"}],
                stream=False,
            )
            if (
                isinstance(response, ChatCompletion)
                and response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                print(response.choices[0].message.content)
            else:
                print("Empty or invalid response")
        except Exception as exc:
            print(f"Request failed: {exc}")

    asyncio.run(main())
