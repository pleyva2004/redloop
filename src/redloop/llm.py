"""Chat clients. Everything talks to OpenAI-compatible endpoints (Ollama,
llama.cpp server, vLLM, OpenAI), so swapping the model under test is config only.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from openai import AsyncOpenAI

from redloop.types import ChatResult, ToolCall


@dataclass
class ModelConfig:
    model: str
    base_url: str = "http://localhost:11434/v1"
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 4
    timeout_s: float = 180.0
    # Provider-specific knobs, e.g. {"reasoning_effort": "none"} for Ollama.
    extra_body: dict[str, Any] = field(default_factory=dict)


class ChatClient(Protocol):
    name: str

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult: ...


class OpenAICompatClient:
    def __init__(self, cfg: ModelConfig, retries: int = 3):
        self.cfg = cfg
        self.name = cfg.model
        self.retries = retries
        key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else "unused"
        self._client = AsyncOpenAI(base_url=cfg.base_url, api_key=key, timeout=cfg.timeout_s)
        self._sem = asyncio.Semaphore(cfg.concurrency)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body

        last_err = ""
        for attempt in range(self.retries):
            async with self._sem:
                t0 = time.perf_counter()
                try:
                    resp = await self._client.chat.completions.create(**kwargs)
                except Exception as e:  # network / server errors are retried
                    last_err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2**attempt)
                    continue
                latency = time.perf_counter() - t0
            msg = resp.choices[0].message
            calls = []
            for tc in msg.tool_calls or []:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
            return ChatResult(text=msg.content or "", tool_calls=calls, latency_s=latency)
        return ChatResult(text="", error=last_err)


class FakeClient:
    """Deterministic client for tests: `fn(messages, tools) -> ChatResult | str`."""

    def __init__(self, fn: Callable[[list[dict], list[dict] | None], ChatResult | str], name: str = "fake"):
        self.fn = fn
        self.name = name
        self.calls: list[list[dict]] = []

    async def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> ChatResult:
        self.calls.append(messages)
        out = self.fn(messages, tools)
        return ChatResult(text=out) if isinstance(out, str) else out


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply (tolerates code fences / prose)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
