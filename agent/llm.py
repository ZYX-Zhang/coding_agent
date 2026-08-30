"""LLM 客户端封装

Agent 只依赖 chat(messages, tools) -> dict 这个接口，

统一返回格式：
    {
        "role": "assistant",
        "content": str,
        "tool_calls": [{"id": str, "type": "function",
                        "function": {"name": str, "arguments": str}}],
        "usage": {"prompt_tokens": int, "completion_tokens": int} | None,
    }
"""
from __future__ import annotations

import os
import time
from .config import api_key, model, url

def _emit_delta(on_delta, text: str, kind: str = "content") -> None:
    """安全地把流式增量上抛给 on_delta。

    - 优先按 (text, kind) 双参调用：kind ∈ {"content", "reasoning"}，
      便于上层（如 Web 前端）区分「正文」与「思考链」分别渲染。
    - 若回调只接受单参（如 CLI 的打字机输出 on_delta(text)），自动降级为
      on_delta(text)，保证旧签名不被破坏、真实流式也不会因 TypeError 崩溃。
    """
    try:
        on_delta(text, kind)
    except TypeError:
        on_delta(text)


class LLMError(Exception):
    """重试耗尽后的最终错误，由调用方（CLI/Agent 外层）兜底。"""


class LLMClient:
    def __init__(self,
                 timeout: float = 300.0,
                 max_retries: int = 3):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError(
                "缺少依赖 openai：请先 pip install openai") from e

        self.api_key = api_key

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=url,
            timeout=timeout,
        )
        self.model = model
        self.max_retries = max_retries

    # ---------------------------------------------------------------- #
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.2,
             on_delta=None) -> dict:
        """一次对话调用。

        :param on_delta: 可选回调 on_delta(text)，传入则启用流式输出，
                         逐段收到 content 时实时上抛（CLI 用来打字机渲染）
        """
        kwargs = dict(model=self.model, messages=messages,
                      temperature=temperature)
        if tools:
            kwargs["tools"] = tools

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if on_delta is not None:
                    return self._chat_stream(kwargs, on_delta)
                return self._chat_once(kwargs)
            except LLMError:
                raise  # 不可重试的配置错误
            except Exception as e:  # 网络 / 429 / 5xx / 超时
                last_err = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt  # 2s, 4s, 8s 退避
                    if on_delta:
                        _emit_delta(on_delta,
                                    f"\n[网络波动，{wait}s 后重试 ({attempt}/"
                                    f"{self.max_retries})…]", "content")
                    time.sleep(wait)
        raise LLMError(f"调用 {self.model} 连续失败 {self.max_retries} 次: "
                       f"{last_err}")

    # ---------------------------------------------------------------- #
    def _chat_once(self, kwargs) -> dict:
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        return {
            "role": "assistant",
            "content": msg.content or "",
            "reasoning": getattr(msg, "reasoning_content", None) or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments or "{}"}}
                for tc in (msg.tool_calls or [])
            ],
            "usage": ({"prompt_tokens": usage.prompt_tokens,
                       "completion_tokens": usage.completion_tokens}
                      if usage else None),
        }

    def _chat_stream(self, kwargs, on_delta) -> dict:
        """流式调用：增量合并 content 与分片的 tool_calls。

        tool_calls 在流式响应里是按 index 分片下发的：
        第一个分片带 id/name，后续分片只带 arguments 的增量，
        必须按 index 聚合拼接，否则拿到的是破碎 JSON。
        """
        kwargs = {**kwargs, "stream": True,
                  "stream_options": {"include_usage": True}}
        stream = self.client.chat.completions.create(**kwargs)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []      # 思考链（部分模型流式下发）
        tc_acc: dict[int, dict] = {}       # index -> {id, name, arguments}
        usage = None

        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            if delta.content:
                content_parts.append(delta.content)
                _emit_delta(on_delta, delta.content, "content")

            # 智谱部分模型（如 GLM-4-AirX/GLM-Z1）把思考链放在 reasoning_content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                _emit_delta(on_delta, reasoning, "reasoning")

            for tc in (delta.tool_calls or []):
                acc = tc_acc.setdefault(tc.index, {"id": "", "name": "",
                                                   "arguments": ""})
                if tc.id:
                    acc["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function.arguments:
                        acc["arguments"] += tc.function.arguments

        return {
            "role": "assistant",
            "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts),
            "tool_calls": [
                {"id": acc["id"] or f"call_{i}", "type": "function",
                 "function": {"name": acc["name"],
                              "arguments": acc["arguments"] or "{}"}}
                for i, acc in sorted(tc_acc.items())
            ],
            "usage": ({"prompt_tokens": usage.prompt_tokens,
                       "completion_tokens": usage.completion_tokens}
                      if usage else None),
        }
