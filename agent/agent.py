"""Agent 主循环

循环逻辑：
    user 输入 → 调 LLM → 解析 tool_calls → 逐个执行 → 结果回填历史
    → 模型调用 finish / 达到 max_turns → 退出
"""
from __future__ import annotations

import json

from .context import Context
from .tools import FINISH_MARKER

# 模型只说话、不调用工具时，回填的提醒（连续超过该次数则放弃）
MAX_IDLE_TURNS = 3
IDLE_NUDGE = (
    "你上一条回复没有调用任何工具。请继续使用工具推进任务，"
    "或调用 finish 总结并结束。"
)


class Agent:
    def __init__(self, llm, tools, context: Context | None = None,
                 safety=None, max_turns: int = 40, on_event=None,
                 on_delta=None):
        """
        :param llm:     任意实现了 chat(messages, tools=None) -> dict 的对象
        :param tools:   ToolRegistry
        :param context: Context；None 时自动创建（默认 100K 预算）
        :param safety:  可调用对象 check(tool, args) -> bool，None 表示全放行
        :param max_turns: 轮数熔断上限
        :param on_event: 事件回调 on_event(kind, payload)，用于 CLI 渲染/日志，
                         测试时用它断言循环内部行为。kind ∈ {
                             turn_start, assistant, tool_start, tool_end, finish}
        :param on_delta: 可选流式回调 on_delta(text)。设置时会把 on_delta
                         透传给 llm.chat
        """
        self.llm = llm
        self.tools = tools
        self.ctx = context or Context()
        self.safety = safety
        self.max_turns = max_turns
        self.on_event = on_event or (lambda kind, payload: None)
        self.on_delta = on_delta

    @property
    def messages(self) -> list[dict]:
        """直接暴露底层历史（append-only 原始记录）。"""
        return self.ctx.messages

    # ------------------------------------------------------------------ #
    def run(self, user_input: str, system_prompt: str = "") -> str:
        """执行一个任务，返回 finish 的 summary 或超限说明。"""
        if system_prompt and not any(m.get("role") == "system"
                                     for m in self.ctx.messages):
            self.ctx.messages.insert(0, {"role": "system", "content": system_prompt})
        self.ctx.add_user(user_input)

        idle = 0  # 连续"只说话不干活"计数
        for turn in range(1, self.max_turns + 1):
            self.on_event("turn_start", {"turn": turn})

            sent = self.ctx.messages_for_llm()  # 裁剪后的视图
            extra = {"on_delta": self.on_delta} if self.on_delta else {}
            resp = self.llm.chat(sent, tools=self.tools.schemas(), **extra)

            # 用真实 usage 校准 token 估算（MockLLM 不带 usage，自动跳过）
            usage = resp.get("usage") if isinstance(resp, dict) else None
            if usage and usage.get("prompt_tokens"):
                self.ctx.report_usage(usage["prompt_tokens"], sent)

            self.ctx.add_assistant(resp)
            self.on_event("assistant", resp)

            tool_calls = resp.get("tool_calls") or []

            # ---- 分支 1：模型只说话不调工具 ----
            if not tool_calls:
                idle += 1
                if idle >= MAX_IDLE_TURNS:
                    return "模型连续多轮未调用工具，任务终止（可能已卡住）。"
                self.ctx.add_user(IDLE_NUDGE)
                continue
            idle = 0

            # ---- 分支 2：逐个执行工具，结果回填 ----
            finished_summary = None
            for tc in tool_calls:
                result = self._dispatch(tc)
                self.ctx.add_tool_result(tc["id"], result,
                                         tool_name=tc["function"]["name"])
                # 终止条件之：finish
                if result.startswith(FINISH_MARKER):
                    finished_summary = result[len(FINISH_MARKER):].strip()

            if finished_summary is not None:
                self.on_event("finish", {"summary": finished_summary})
                return finished_summary

        # ---- 分支 3：轮数熔断 ----
        return f"已达到最大轮数上限（{self.max_turns}），任务未明确完成。"

    # ------------------------------------------------------------------ #
    def _dispatch(self, tc: dict) -> str:
        """执行单个 tool call。任何失败都返回 [error] 文本，绝不抛异常。"""
        name = tc["function"]["name"]

        # 幻觉工具
        tool = self.tools.get(name)
        if tool is None:
            return (f"[error] 未知工具: {name}，可用工具: {self.tools.names()}")

        # 参数是 JSON 字符串，解析失败要回给模型
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError as e:
            return f"[error] 参数 JSON 解析失败: {e}"

        self.on_event("tool_start", {"name": name, "args": args})

        # 安全检查（用户拒绝 => blocked；同样发事件，供 CLI 渲染/审计）
        if self.safety is not None and not self.safety.check(tool, args):
            result = "[blocked] 用户拒绝了该操作，请换一种方式或询问用户。"
            self.on_event("tool_end", {"name": name, "result": result})
            return result

        try:
            result = tool.execute(**args)
        except TypeError as e:  # 参数签名不匹配（模型给错参数）
            result = f"[error] 参数不合法: {e}"
        except Exception as e:  # 工具内部异常兜底
            result = f"[error] 工具执行异常: {type(e).__name__}: {e}"
        self.on_event("tool_end", {"name": name, "result": result})

        return result
