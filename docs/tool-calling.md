# Tool Calling

> **Last updated:** 2026-07-16
> **Status:** Current & maintained
> **Target audience:** Developers building agentic AI applications

## Table of Contents

- [Overview](#overview)
- [Creating Tools](#creating-tools)
- [Registering Tools](#registering-tools)
- [Using Tools in Chat](#using-tools-in-chat)
- [How It Works Internally](#how-it-works-internally)
- [Tool Definition Reference](#tool-definition-reference)
- [Examples](#examples)

---

## Overview

LLM4Free provides a built-in tool calling system that lets a model invoke
Python functions during a conversation. There are two complementary paths,
depending on which provider base class you use.

**Key concepts:**

- **`Tool`** — a dataclass that wraps a name, description, parameter schema, and an optional callable implementation. It is defined in `llm4free/AIbase.py` and re-exported from `llm4free/llm`.
- **`register_tools()`** — stores tools on the provider instance so the provider can find them.
- **OpenAI-compatible providers** (`llm4free.llm.*`) — such as `Apriel`, `LLMChat`, `Toolbaz`, `HeckAI`. Pass tools to `provider.chat.completions.create(model=..., messages=..., tools=[...])`. The base class injects the tool definitions, parses `<invoke>` blocks, and returns native `tool_calls` in the response.
- **Non-OpenAI providers** (subclasses of `llm4free.AIbase.Provider`) — these expose a convenience `chat(prompt, tools=[...])` method that runs a full automatic tool-calling loop (inject definitions, execute tools, feed `<tool_result>` back) and returns final text.

> [!NOTE]
> `Tool` is a standalone dataclass, **not** a method on the base class. Always import it explicitly: `from llm4free.AIbase import Tool`.
> The OpenAI-compatible providers (`llm4free.llm`) do **not** have a `chat(prompt, tools=...)` method — use `chat.completions.create(tools=[...])` instead.

---

## Tool Calling Through the Unified Client

Tool calling also works through the unified `Client`. Pass `tools=[...]` to `client.chat.completions.create(model="auto", messages=[...], tools=[...])` — the `Client` resolves the model to an OpenAI-compatible provider behind the scenes, then runs the standard tool loop and returns native `tool_calls`. This gives you auto-failover **and** function calling in one call.

```python
from llm4free.client import Client
from llm4free.AIbase import Tool

def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 25C"

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city.",
    parameters={
        "city": {"type": "string", "description": "City name."},
    },
    implementation=get_weather,
)

client = Client(print_provider_info=True)
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What is the weather in London?"}],
    tools=[weather_tool],
)
print(response.choices[0].message.tool_calls)
```

> [!TIP]
> Prefer the unified `Client` for tool calling when you want automatic provider selection and failover. Use a raw provider class (e.g. `from llm4free.llm.apriel import Apriel`) when you need to pin a specific provider or register tools at init time.
>
> [!WARNING]
> Streaming is disabled when tools are present on OpenAI-compatible providers. If you pass `stream=True` together with `tools=`, the provider silently ignores `stream` and returns a single `ChatCompletion` so it can inspect the full response before continuing the loop.

---

## Creating Tools

Import `Tool` from `llm4free.AIbase` and create instances with a name,
description, parameter schema, and an optional implementation function.

```python
from llm4free.AIbase import Tool

# 1. Define the Python function that does the work
def get_weather(city: str, unit: str = "celsius") -> str:
    """Fetch current weather (stub)."""
    return f"Weather in {city}: 22 degrees {unit[0].upper()}, sunny"

# 2. Wrap it in a Tool
weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a given city.",
    parameters={
        "city": {
            "type": "string",
            "description": "The city to look up weather for.",
        },
        "unit": {
            "type": "string",
            "description": "Temperature unit: 'celsius' or 'fahrenheit'.",
        },
    },
    required_params=["city"],          # only city is mandatory
    implementation=get_weather,         # callable that executes the tool
)
```

### Tool with no implementation

If you omit `implementation`, calling `execute()` returns a placeholder string.
This is useful when you want the model to *describe* what it would do without
actually executing.

```python
search_tool = Tool(
    name="web_search",
    description="Search the web for a query.",
    parameters={
        "query": {"type": "string", "description": "Search query."},
    },
)
# tool.execute({"query": "python"}) -> "Tool 'web_search' does not have an implementation."
```

---

## Registering Tools

### Option A — at init time (recommended)

Every provider that supports tools accepts a `tools` parameter in its constructor. The tools are registered automatically.

```python
from llm4free.llm.apriel import Apriel
from llm4free.AIbase import Tool

def add(a: int, b: int) -> int:
    return a + b

calculator = Tool(
    name="add",
    description="Add two numbers.",
    parameters={
        "a": {"type": "integer", "description": "First number."},
        "b": {"type": "integer", "description": "Second number."},
    },
    implementation=add,
)

ai = Apriel(tools=[calculator])   # tools are registered here
```

### Option B — call `register_tools()` manually

```python
ai = Apriel()
ai.register_tools([calculator])
```

Both approaches populate `ai.available_tools`, a `dict[str, Tool]` that the
provider uses during the tool-calling loop.

> [!NOTE]
> The OpenAI-compatible provider constructor signature is
> `OpenAICompatibleProvider(api_key=None, tools=None, proxies=None, **kwargs)`.
> Tools passed at init time are stored but, for OpenAI-compatible providers, tool
> execution is driven per-request via `chat.completions.create(tools=[...])`.

---

## Using Tools in Chat

For OpenAI-compatible providers (`llm4free.llm.*`), pass tools to
`chat.completions.create()`. The provider injects the tool definitions into the
prompt, calls the model, and — when the model emits `<invoke>` blocks — returns a
`ChatCompletion` whose `choices[0].message.tool_calls` are populated in the
native OpenAI format. Your code is then responsible for executing the tools and
making a follow-up call with the results (standard OpenAI pattern).

```python
from llm4free.llm.apriel import Apriel
from llm4free.AIbase import Tool

def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 25C"

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city.",
    parameters={
        "city": {"type": "string", "description": "City name."},
    },
    implementation=get_weather,
)

ai = Apriel()
response = ai.chat.completions.create(
    model="Apriel-1.6-15B-Thinker",
    messages=[{"role": "user", "content": "What is the weather in London?"}],
    tools=[weather_tool],
)
print(response.choices[0].message.tool_calls)
```

### Passing tools per-request

You can also pass tools directly to `chat.completions.create()` without registering them at init time (as shown above). The provider accepts both `Tool` instances and plain OpenAI-style `dict` definitions:

```python
ai = Apriel()
response = ai.chat.completions.create(
    model="Apriel-1.6-15B-Thinker",
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name."}
                    },
                    "required": ["city"],
                },
            },
        }
    ],
)
```

> [!TIP]
> For non-OpenAI providers (subclasses of `llm4free.AIbase.Provider`), you can use the convenience `chat()` loop instead:
> `ai.chat("What is the weather in London?", tools=[weather_tool])`. This runs the entire
> invoke → execute → feed-back loop automatically and returns the final text. This method
> is **not** available on `llm4free.llm` OpenAI-compatible providers.

> [!WARNING]
> Streaming is disabled when tools are present on OpenAI-compatible providers. If you pass
> `stream=True` together with `tools=`, the provider silently ignores `stream` and returns a
> single `ChatCompletion` so it can inspect the full response before continuing the loop.

---

## How It Works Internally

For OpenAI-compatible providers, the tool loop in `BaseCompletions._run_non_native_tool_loop()` follows these steps:

```
1. Parse the supplied tools into Tool objects (accepts both Tool and dict).
2. Inject tool definitions as an XML instruction block into the (first) system message.
3. Call provider.create() once with stream disabled, tools=None.
4. Parse the response for <invoke> blocks.
5. If no <invoke> found -> return the response verbatim as the final answer.
6. If <invoke> found -> build native ToolCall objects in a ChatCompletion
   with finish_reason="tool_calls" and return it for you to execute + follow up.
```

### XML format the model uses

**Tool invocation** (model output):

```xml
<invoke>
  <tool_name>get_weather</tool_name>
  <parameters>{"city": "London"}</parameters>
</invoke>
```

**Tool result** (fed back to model on the follow-up call):

```xml
<tool_result>
  <tool_name>get_weather</tool_name>
  <result>Weather in London: Sunny, 25C</result>
</tool_result>
```

The model sees these in its conversation context and can chain multiple tool
calls before producing a final text answer.

For non-OpenAI providers, `Provider.run_tool_loop()` performs the equivalent loop internally and returns the final text directly.

---

## Tool Definition Reference

### `Tool` dataclass

| Field              | Type                                | Required | Description                                           |
| ------------------ | ----------------------------------- | -------- | ----------------------------------------------------- |
| `name`             | `str`                               | Yes      | Unique function name the model will invoke.           |
| `description`      | `str`                               | Yes      | What the tool does.                                   |
| `parameters`       | `Dict[str, Dict[str, Any]]`         | No       | Parameter schema (name → info dict). Default: `{}`.   |
| `required_params`  | `Optional[List[str]]`               | No       | Required parameter names. Default: all parameters.    |
| `implementation`   | `Optional[Callable[..., Any]]`      | No       | Python function that executes the tool.               |

> [!NOTE]
> `Tool` is defined in `llm4free/AIbase.py` (the `llm4free.AIbase.Provider` base class also uses it). A duplicate but identical `Tool` dataclass exists in `llm4free/llm/base.py` and is re-exported from `llm4free/llm`. Both are interchangeable; prefer `from llm4free.AIbase import Tool`.

### Parameter schema format

Each entry in `parameters` is a dict with at least `type` and `description`:

```python
{
    "city": {
        "type": "string",
        "description": "City name.",
    },
    "unit": {
        "type": "string",
        "description": "Temperature unit.",
        "enum": ["celsius", "fahrenheit"],   # optional constraint
    },
}
```

### `Tool` methods

| Method      | Signature                          | Returns          |
| ----------- | ---------------------------------- | ---------------- |
| `to_dict()` | `() -> Dict[str, Any]`             | OpenAI-compatible tool definition dict |
| `execute()` | `(arguments: Dict[str, Any]) -> Any` | Result of calling `implementation(**arguments)` |

---

## Examples

### Example 1 — Calculator (OpenAI-compatible provider)

```python
import ast
import operator

from llm4free.llm.llmchat import LLMChat
from llm4free.AIbase import Tool

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return SAFE_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safe calculator using AST parsing (no eval)."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

calc_tool = Tool(
    name="calculate",
    description="Evaluate a mathematical expression.",
    parameters={
        "expression": {
            "type": "string",
            "description": "Math expression, e.g. '2 + 3 * 4'.",
        },
    },
    implementation=calculate,
)

ai = LLMChat()
response = ai.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What is 142 * 37?"}],
    tools=[calc_tool],
)
print(response.choices[0].message.tool_calls)
```

### Example 2 — Multiple tools (OpenAI-compatible provider)

```python
from llm4free.llm.toolbaz import Toolbaz
from llm4free.AIbase import Tool
import datetime

def get_time() -> str:
    return datetime.datetime.now().isoformat()

def get_random_number(min_val: int = 1, max_val: int = 100) -> int:
    import random
    return random.randint(min_val, max_val)

time_tool = Tool(
    name="get_time",
    description="Get the current date and time.",
    parameters={},
    implementation=get_time,
)

random_tool = Tool(
    name="get_random_number",
    description="Generate a random integer in a range.",
    parameters={
        "min_val": {"type": "integer", "description": "Minimum value (inclusive)."},
        "max_val": {"type": "integer", "description": "Maximum value (inclusive)."},
    },
    required_params=[],
    implementation=get_random_number,
)

ai = Toolbaz()
response = ai.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What time is it? And give me a random number between 1 and 50."}],
    tools=[time_tool, random_tool],
)
print(response.choices[0].message.tool_calls)
```

### Example 3 — Web search stub (non-OpenAI provider loop)

When you use a provider that subclasses `llm4free.AIbase.Provider`, you can let the
`chat()` convenience method run the entire loop for you and return the final text:

```python
from llm4free.AIbase import Provider, Tool

def web_search(query: str, num_results: int = 3) -> str:
    """Stub that simulates a web search."""
    return f"Found {num_results} results for '{query}': [result links would appear here]"

search_tool = Tool(
    name="web_search",
    description="Search the web and return top results.",
    parameters={
        "query": {"type": "string", "description": "The search query."},
        "num_results": {
            "type": "integer",
            "description": "Number of results to return.",
        },
    },
    required_params=["query"],
    implementation=web_search,
)

# NOTE: only subclasses of llm4free.AIbase.Provider expose chat(prompt, tools=...).
# For llm4free.llm OpenAI-compatible providers, pass tools to
# chat.completions.create(...) instead (see Examples 1 and 2).
```

### Example 4 — Streaming without tool calls

When no tools are registered (or the model answers directly without calling a tool),
streaming works normally on OpenAI-compatible providers:

```python
from llm4free.llm.apriel import Apriel

# No tools -> streaming behaves as usual
ai = Apriel()
for chunk in ai.chat.completions.create(
    model="Apriel-1.6-15B-Thinker",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True,
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```
