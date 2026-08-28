"""上下文与对话历史管理

1. 维护完整历史
2. token估算
3. messages_for_llm()：超预算时返回"裁剪后的副本"——原始历史不动
   裁剪分三级，逐级加压：
   L1 省略旧工具输出  →  L2 压缩旧轮次为摘要  →  L3 硬截断最旧消息
"""
from __future__ import annotations

import copy
import re

# ---------------------------------------------------------------- #
# token 估算
# ---------------------------------------------------------------- #
# 粗略覆盖 CJK 表意文字与全角符号
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")


def estimate_text_tokens(text: str) -> int:
    """粗估一段文本的 token 数：中文约 1 字 1 token，英文约 4 字符 1 token。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return int(cjk + other / 4) + 1


class Context:
    def __init__(self, system_prompt: str = "", max_tokens: int = 100_000,
                 keep_recent_turns: int = 8, elide_over_chars: int = 200):
        """
        :param max_tokens:        上下文预算
        :param keep_recent_turns: 最近 N 轮完整保留
        :param elide_over_chars:  旧工具结果超过该长度即省略
        """
        self.max_tokens = max_tokens
        self.keep_recent_turns = keep_recent_turns
        self.elide_over_chars = elide_over_chars

        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        # 校准系数：真实 usage / 估算值 的滑动平均，初始 1.0
        self._calib = 1.0
        # 可选的 LLM 摘要器：callable(old_messages) -> str。
        # 不设置时用启发式摘要（零成本、离线可用）
        self.summarizer = None

    # ---------------------------------------------------------------- #
    # 历史写入（append-only）
    # ---------------------------------------------------------------- #
    def add_user(self, content) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, resp: dict) -> None:
        """resp 为 llm.chat() 的返回：{role, content, tool_calls}"""
        self.messages.append({
            "role": "assistant",
            "content": resp.get("content") or "",
            **({"tool_calls": resp["tool_calls"]} if resp.get("tool_calls") else {}),
        })

    def add_tool_result(self, tool_call_id: str, result: str,
                        tool_name: str = "") -> None:
        """tool_name 仅用于摘要统计，不进入消息体。"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })

    def add_message(self, msg: dict) -> None:
        """通用追加（恢复会话/注入提醒等）。"""
        self.messages.append(msg)

    # ---------------------------------------------------------------- #
    # token 估算与校准
    # ---------------------------------------------------------------- #
    def estimate_message(self, msg: dict) -> int:
        """估算单条消息 token"""
        content = msg.get("content")
        if isinstance(content, str):
            n = estimate_text_tokens(content)
        elif isinstance(content, list):
            n = 0
            for part in content:
                if isinstance(part, dict):
                    n += estimate_text_tokens(part.get("text", ""))
        else:
            n = 0
        # tool_calls 的函数名+参数也要计入
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            n += estimate_text_tokens(fn.get("name", ""))
            n += estimate_text_tokens(fn.get("arguments", ""))
        n += 4  # role/结构开销
        return max(1, int(n * self._calib))

    def estimate_total(self, msgs: list[dict] | None = None) -> int:
        return sum(self.estimate_message(m) for m in (msgs if msgs is not None
                                                      else self.messages))

    def report_usage(self, prompt_tokens: int, sent_messages: list[dict]) -> None:
        """用 API 返回的真实 prompt_tokens 校准估算系数（EMA 平滑）。

        :param prompt_tokens:  resp.usage.prompt_tokens
        :param sent_messages:  本次实际发出的消息列表（裁剪后的那份）
        """
        est = sum(self.estimate_message(m) for m in sent_messages)
        if est <= 0 or prompt_tokens <= 0:
            return
        ratio = prompt_tokens / est
        ratio = min(max(ratio, 0.5), 2.0)  # 防单次异常拉偏
        self._calib = round(0.7 * self._calib + 0.3 * ratio, 4)

    # ---------------------------------------------------------------- #
    # 核心：生成发给 LLM 的消息列表（三级裁剪）
    # ---------------------------------------------------------------- #
    def messages_for_llm(self) -> list[dict]:
        """返回（可能已裁剪的）消息副本；self.messages 永不被修改。"""
        msgs = copy.deepcopy(self.messages)

        if self.estimate_total(msgs) <= self.max_tokens:
            return msgs

        # ---- L1：省略旧工具输出 ----
        boundary = self._recent_boundary(msgs)
        msgs = self._elide_old_tool_results(msgs, boundary)
        if self.estimate_total(msgs) <= self.max_tokens:
            return msgs

        # ---- L2：压缩旧轮次为摘要 ----
        msgs = self._compact_old_turns(msgs)
        if self.estimate_total(msgs) <= self.max_tokens:
            return msgs

        # ---- L3：硬截断（保 system + 尽可能新的消息） ----
        return self._hard_truncate(msgs)

    # ---------------------------------------------------------------- #
    # 内部：轮次边界与三级裁剪
    # ---------------------------------------------------------------- #
    def _recent_boundary(self, msgs: list[dict]) -> int:
        """最近 keep_recent_turns 轮的起始下标。

        一"轮"= 一条 assistant 消息（含或不含 tool_calls）。
        切在 assistant 边界上，保证 tool_calls/tool 配对不被拆散。
        """
        assistant_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        if len(assistant_idx) <= self.keep_recent_turns:
            return 0
        return assistant_idx[-self.keep_recent_turns]

    def _elide_old_tool_results(self, msgs: list[dict], boundary: int) -> list[dict]:
        """boundary 之前的超长工具输出替换为省略占位符。"""
        for m in msgs[:boundary]:
            if (m.get("role") == "tool"
                    and isinstance(m.get("content"), str)
                    and len(m["content"]) > self.elide_over_chars):
                m["content"] = (f"[已省略 {len(m['content'])} 字符的工具输出，"
                                f"如需查看请重新调用相应工具]")
        return msgs

    def _compact_old_turns(self, msgs: list[dict]) -> list[dict]:
        """把旧轮次折叠为一条摘要消息。

        保留：system（若有）、第一条 user（原始任务）。
        折叠：二者之后、最近轮边界之前的全部消息。
        """
        boundary = self._recent_boundary(msgs)

        head_end = 0
        if msgs and msgs[0].get("role") == "system":
            head_end = 1
        # 第一条 user 消息 = 原始任务，保留
        first_user = next((i for i, m in enumerate(msgs)
                           if m.get("role") == "user"), None)
        if first_user is not None and first_user >= head_end:
            head_end = first_user + 1

        old = msgs[head_end:boundary]
        if not old:
            return msgs

        summary = (self.summarizer(old) if callable(self.summarizer)
                   else self._heuristic_summary(old))

        return (msgs[:head_end]
                + [{"role": "user",
                    "content": f"[历史摘要｜以下为早期操作的压缩记录]\n{summary}"}]
                + msgs[boundary:])

    def _heuristic_summary(self, old: list[dict]) -> str:
        """零成本启发式摘要：抽取工具调用轨迹 + finish 结论。"""
        lines, calls = [], []
        for m in old:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                try:
                    import json as _json
                    args = _json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                key = args.get("path") or args.get("cmd") or args.get("pattern") or ""
                calls.append(f"{name} {key}".strip())
            c = m.get("content")
            if isinstance(c, str) and c.startswith("__FINISHED__"):
                lines.append(f"早期结论: {c[len('__FINISHED__'):][:200]}")

        lines.append(f"早期共执行 {len(calls)} 次工具调用，轨迹（最多列 30 条）:")
        lines += [f"- {c[:120]}" for c in calls[:30]]
        lines.append("以上旧工具输出已省略；如需细节请重新调用工具查看。")
        return "\n".join(lines)

    def _hard_truncate(self, msgs: list[dict]) -> list[dict]:
        """最后防线：优先丢最旧的整轮（切在 assistant 边界保配对），
        若已无整轮可丢，则直接截断超长工具输出——宁可丢内容也不爆窗。
        """
        while self.estimate_total(msgs) > self.max_tokens and len(msgs) > 2:
            cut = next((i for i, m in enumerate(msgs)
                        if m.get("role") == "assistant"), None)
            if cut is None:
                break
            head = [msgs[0]] if msgs[0].get("role") == "system" else []
            new = head + msgs[cut:]
            if len(new) >= len(msgs):  # 无进展：第一条 assistant 已紧贴头部
                break
            msgs = new

        # 仍超预算：截断超长工具输出（最后兜底）
        if self.estimate_total(msgs) > self.max_tokens:
            per = max(200, self.max_tokens // 4)
            for m in msgs:
                if (m.get("role") == "tool"
                        and isinstance(m.get("content"), str)
                        and len(m["content"]) > per):
                    m["content"] = (m["content"][:per]
                                    + "\n[error] 输出过长被截断")
        return msgs
