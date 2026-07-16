# API Reference

> Last updated: 2026-07-16
> Type: Technical Reference
> Audience: Developers

This document describes the real, importable classes and methods in `llm4free`. Examples use only verified import paths.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Provider Base Class](#provider-base-class)
3. [LLM Providers (OpenAI-Compatible)](#llm-providers-openai-compatible)
4. [Client API](#client-api)
5. [Text-to-Image (TTI)](#text-to-image-tti)
6. [Text-to-Speech (TTS)](#text-to-speech-tts)
7. [Model Registry](#model-registry)
8. [Exceptions](#exceptions)
9. [Type Definitions](#type-definitions)

---

## Core Classes

### `Client` (unified entry point)

The `Client` is the recommended entry point for everyday use. It wraps all OpenAI-compatible chat, image, and audio providers behind one OpenAI-compatible interface and performs automatic provider selection, model resolution, and failover.

```python
from llm4free.client import Client

client = Client()  # optional: api_key=, proxies=, exclude=, exclude_images=, exclude_tts=, print_provider_info=
```

**Three model-specification modes** (identical across `chat`, `images`, and `audio`):

| `model` value | Meaning |
| ------------- | ------- |
| `"auto"` | The `Client` picks any working provider/model automatically (the smartest default). |
| `"ProviderName/ModelName"` | Forces a specific provider, e.g. `"ChatGPT/gpt-4o"` or `"PollinationsAI/flux"`. |
| `"ModelName"` (bare) | Fuzzily matched across all providers, e.g. `"gpt-4o"`. |

```python
# 1. Auto — pick any working provider/model
client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hi"}])

# 2. Force a specific provider/model
client.chat.completions.create(model="ChatGPT/gpt-4o", messages=[{"role": "user", "content": "Hi"}])

# 3. Bare model name — fuzzily matched
client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])
```

Class-method helpers give runtime-discovered provider lists: `Client.get_chat_providers()`, `Client.get_free_chat_providers()`, `Client.get_image_providers()`, `Client.get_tts_providers()`. See [Client API](#client-api) for the full surface (chat/image/audio + `run_api`/`start_server`).

### `AIbase.Provider`

The abstract base class for the legacy (non-OpenAI-compatible) providers. Subclasses implement `ask()` and `get_message()`; `chat()` is provided and supports an automatic XML tool-calling loop when tools are supplied.

```python
from llm4free.AIbase import Provider
```

**Key attributes:**

- `required_auth` (bool) — Whether the provider requires API authentication. Defaults to `False`.
- `conversation` (Any) — Stores conversation history when `is_conversation=True`.
- `last_response` (ResponseType) — The most recent raw response from `ask()`.

**Key methods:**

- `ask(prompt, **kwargs)` — Send a prompt; returns the raw response (str / dict / generator).
- `chat(prompt, **kwargs)` — Send a prompt and return the extracted message string (or generator when `stream=True`).
- `get_message(response)` — Extract the message text from a raw response.
- `register_tools(tools)` — Register `Tool` instances for function calling.

> [!NOTE]
> Most everyday usage goes through the OpenAI-compatible providers in `llm4free.llm` (see below) or the unified `Client`. The `Provider` base class is primarily relevant when writing or extending legacy providers.

### `AIbase.Tool`

A dataclass describing a callable tool for function calling, convertible to an OpenAI tool definition.

```python
from llm4free.AIbase import Tool

def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 75F"

tool = Tool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={"city": {"type": "string", "description": "City name"}},
    implementation=get_weather,
)
print(tool.to_dict())   # OpenAI-compatible tool definition
print(tool.execute({"city": "London"}))
```

---

## Provider Base Class

### Abstract / Overridable Methods

All `Provider` subclasses implement these:

#### `ask()`

Sends a prompt and returns raw response data.

```python
def ask(
    self,
    prompt: str,
    stream: bool = False,
    raw: bool = False,
    optimizer: Optional[str] = None,
    conversationally: bool = False,
    **kwargs: Any,
) -> Response:
    """Send a prompt to the provider and return the raw response."""
```

- `prompt` (str): The user's input prompt.
- `stream` (bool): Return a streaming response if True. Default: `False`.
- `raw` (bool): Return raw data without post-processing. Default: `False`.
- `optimizer` (Optional[str]): Apply a system prompt optimizer.
- `conversationally` (bool): Maintain conversation history. Default: `False`.
- **Returns:** `str`, `dict`, or generator depending on parameters.

#### `chat()`

Sends a prompt and extracts the message from the response.

```python
def chat(
    self,
    prompt: str,
    stream: bool = False,
    optimizer: Optional[str] = None,
    conversationally: bool = False,
    **kwargs: Any,
) -> Union[str, Generator[str, None, None]]:
    """Send a prompt and get a clean message response."""
```

- **Returns:** `str` or `Generator[str, None, None]`.

```python
from llm4free.AIbase import Provider

# Subclass usage; for built-in providers prefer the llm.* modules below.
# provider = SomeProvider()
# response = provider.chat("Explain machine learning")
# print(response)
```

#### `get_message()`

Extracts the message text from a raw response object.

```python
def get_message(self, response: Response) -> str:
    """Extract the message text from a provider response."""
```

---

## LLM Providers (OpenAI-Compatible)

All providers in `llm4free.llm` subclass `OpenAICompatibleProvider` and expose the familiar `chat.completions.create(...)` interface. Import them by their module path.

```python
from llm4free.llm.heckai import HeckAI        # Free, no API key
from llm4free.llm.artingai import ArtingAI    # Free, no API key
from llm4free.llm.Auth.groq import Groq       # Requires API key
from llm4free.llm.Auth.deepinfra import DeepInfra  # Requires API key
```

> [!TIP]
> `HeckAI` and `ArtingAI` are free and need no key. `Groq` and `DeepInfra` require an `api_key`. The full set of auth-required providers (e.g. `Cerebras`, `HuggingFace`, `Nvidia`, `OpenRouter`, `Sambanova`, `TogetherAI`, `Upstage`, `Zenmux`) is exported from `llm4free.llm.Auth`.

Example:

```python
from llm4free.llm.heckai import HeckAI

client = HeckAI()
response = client.chat.completions.create(
    model="google/gemini-2.5-flash-preview",
    messages=[{"role": "user", "content": "What is artificial intelligence?"}],
)
print(response.choices[0].message.content)
```

Auth-required providers take an `api_key`:

```python
from llm4free.llm.Auth.groq import Groq

client = Groq(api_key="your-groq-key")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Explain machine learning simply"}],
)
print(response.choices[0].message.content)
```

### Streaming and Standard Parameters

OpenAI-compatible providers accept `stream`, `temperature`, `max_tokens`, `top_p`, and `tools`:

```python
from llm4free.llm.heckai import HeckAI

client = HeckAI()
stream = client.chat.completions.create(
    model="google/gemini-2.5-flash-preview",
    messages=[{"role": "user", "content": "Write a short story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Client API

> [!NOTE]
> **Start here.** The unified `llm4free.client.Client` is the recommended way to use LLM4Free. It provides a unified interface over all providers with automatic provider selection and failover, and accepts the same `model` strings (`"auto"`, `"Provider/Model"`, or a bare model name) as the server.

```python
from llm4free.client import Client

# Create client
client = Client()

# Chat with automatic provider selection
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)

# Specify a provider/model explicitly (ProviderName/ModelName)
response = client.chat.completions.create(
    model="ChatGPT/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)

# With streaming
stream = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Available Providers

The client discovers providers dynamically. A representative sample:

| Provider        | Module Path                          | Required Auth | Notes                       |
| --------------- | ------------------------------------ | ------------- | --------------------------- |
| `HeckAI`        | `llm4free.llm.heckai`                | No            | Free, no API key needed     |
| `ArtingAI`      | `llm4free.llm.artingai`              | No            | Free, no API key needed     |
| `Groq`          | `llm4free.llm.Auth.groq`             | Yes           | Fast inference, free tier   |
| `DeepInfra`     | `llm4free.llm.Auth.deepinfra`        | Yes           | Open models                |
| `OpenRouter`    | `llm4free.llm.Auth`                  | Yes           | Multi-model provider        |
| `TogetherAI`    | `llm4free.llm.Auth`                  | Yes           | Open models                |

For the full list, see [`Provider.md`](../Provider.md) and `client.list_providers()`.

---

## Text-to-Image (TTI)

TTI providers subclass `TTICompatibleProvider` and expose an `images.generate(...)` method. Import them from `llm4free.TTI`.

```python
from llm4free.TTI import PollinationsAI

generator = PollinationsAI()
response = generator.images.generate(
    prompt="A peaceful river in autumn",
    size="1024x1024",
)
print(response.data)  # List of generated image objects
```

> [!NOTE]
> TTI provider classes are `PollinationsAI`, `BingImageAI`, `Lexica`, `RaphaelAI`, `StableHordeAI`, `TogetherImage`, `MiragicAI`, `PerchanceAI`, `VisualGPT`, `MagicHourAI`, `MagicStudioAI`, `NoLoginTool`, `OneFreeAI`. Verify the exact `generate` parameters per provider via `help()`.

---

## Text-to-Speech (TTS)

TTS providers subclass `BaseTTSProvider` and expose `audio.speech.create(...)` (or `create_speech(...)`). Import them from `llm4free.TTS`.

```python
from llm4free.TTS import ElevenlabsTTS

tts = ElevenlabsTTS(api_key="your-key")
audio_path = tts.audio.speech.create(
    input="Hello from LLM4Free",
    voice="alloy",
)
print(f"Audio saved to: {audio_path}")
```

> [!NOTE]
> TTS provider classes include `ElevenlabsTTS`, `DeepgramTTS`, `KittenTTS`, `LuxTTS`, `MurfAITTS`, `OpenAIFMTTS`, `ParlerTTS`, `PocketTTS`, `QwenTTS`, `SherpaTTS`, `StreamElements`, `TTSAI`, `TTSOpenTTS`, `XLNKTTS`. Some require an `api_key`.

---

## Model Registry

The `model` object exposes the available models and voices across providers.

```python
from llm4free import model

# LLM models
model.llm.list()       # Dict[provider, List[model_name]]
model.llm.get("HeckAI")  # List of models for a provider
model.llm.summary()    # Counts of providers and models

# TTS voices
model.tts.list()       # Dict[provider, voices]
model.tts.get("ElevenlabsTTS")

# TTI models
model.tti.list()       # Dict[provider, List[model_name]]
model.tti.get("PollinationsAI")
model.tti.providers()  # Detailed provider metadata
```

---

## Exceptions

### `AIProviderError`

Base exception for provider-related errors.

```python
from llm4free.exceptions import AIProviderError

try:
    response = client.chat.completions.create(...)
except AIProviderError as e:
    print(f"Provider error: {e}")
```

> [!TIP]
> When using the unified `Client`, inspect the error string for HTTP status hints (e.g. `"401"` for auth, `"429"` for rate limits) and retry with backoff.

---

## Type Definitions

### Response Type

```python
from typing import Union, Dict, Generator, Any

Response = Union[Dict[str, Any], Generator[Any, None, None], str]
```

The response can be:

- **str** — Simple string response.
- **Dict** — Complex response with metadata (raw response from `ask()`).
- **Generator** — Stream of chunks when `stream=True`.

### Message Format

```python
# Typical message structure for OpenAI-compatible providers
message = {
    "role": "assistant",  # "assistant" or "user"
    "content": "Response text"
}

# With tool calls (if supported)
message = {
    "role": "assistant",
    "content": "Text response",
    "tool_calls": [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "NYC"}'
            }
        }
    ]
}
```

---

## See Also

- [Getting Started](getting-started.md) — Quick start guide
- [Provider Development](provider-development.md) — Create custom providers
- [Troubleshooting](troubleshooting.md) — Solutions to common issues
- [Examples](examples/README.md) — Real-world code examples
