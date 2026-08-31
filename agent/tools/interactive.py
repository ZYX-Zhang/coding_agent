"""交互与任务管理工具：ask_user / plan

这三个工具解决的都是"长任务里的信息管理"：
- ask_user：让模型在信息不足时主动提问，而不是瞎猜
- plan：复杂任务先列计划、随执行更新，避免"做到哪忘到哪"
"""
from __future__ import annotations

from .base import Tool


def _default_ask(question: str) -> str:
    return "（当前环境无交互通道，请基于已有信息自行决策并在 finish 中说明）"


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "向用户提一个明确的问题并等待回答。\n"
        "适用场景：任务要求有歧义（改哪个文件/采用哪种方案/删除是否安全）"
        "且猜错代价高时。\n"
        "注意：一次只问一个问题、给出候选项更好；能用工具查到的事实不要问用户。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string",
                         "description": "要问的问题，写清楚背景与候选项"},
        },
        "required": ["question"],
    }

    def __init__(self, ask=None):
        self.ask = ask or _default_ask

    def execute(self, question: str = "", **_) -> str:
        try:
            if not question.strip():
                return "[error] question 不能为空"
            answer = self.ask(question.strip())
            answer = (answer or "").strip()
            if not answer:
                return "[用户回答] （用户未作答，请自行决策并说明）"
            return f"[用户回答] {answer}"
        except Exception as e:
            return f"[error] 提问失败: {type(e).__name__}: {e}"


class PlanTool(Tool):
    name = "plan"
    description = (
        "制定或更新当前任务的执行计划，支持逐步勾选（完成多少勾多少，不留 0/x）。\n"
        "action: set(整体设步骤, 需 steps) / done(标记某步完成, 需 index) / "
        "list(查看当前清单与进度) / clear(清空)。\n"
        "index 取 list/set 返回的步骤序号（从 1 开始）。\n"
        "典型循环：开局 plan(action='set', steps=[...])；每完成一步就 "
        "plan(action='done', index=N) 把它勾掉，前端会实时刷新 n/m 进度。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["set", "done", "list", "clear"],
                       "description": "操作类型"},
            "steps": {"type": "array", "items": {"type": "string"},
                      "description": "action=set 时的完整步骤列表"},
            "index": {"type": "integer",
                      "description": "action=done 时的步骤序号（list 返回的编号，从 1 开始）"},
        },
        "required": ["action"],
    }

    def __init__(self, on_event=None):
        # on_event(kind, payload) 用于把结构化计划推给前端实时渲染；
        # 不传（CLI/测试）则只返回文本，不影响功能。
        self.on_event = on_event
        self.items: list[dict] = []       # 每个注册表一个实例 => 会话隔离
        # 每项 {"text": str, "done": bool}

    def _emit(self) -> None:
        """变更后把结构化计划推给前端（若注入 on_event）。"""
        if not self.on_event:
            return
        done = sum(1 for it in self.items if it.get("done"))
        self.on_event("plan", {
            "items": [{"text": it.get("text", ""), "done": bool(it.get("done"))}
                      for it in self.items],
            "total": len(self.items),
            "done": done,
        })

    def _render(self) -> str:
        if not self.items:
            return "[ok] 计划为空（共完成 0 步）。任务似乎已接近完成，请收尾并 finish。"
        done = sum(1 for it in self.items if it["done"])
        lines = [f"[当前计划] 已完成 {done}/{len(self.items)} 步:"]
        for i, it in enumerate(self.items, 1):
            mark = "x" if it["done"] else " "
            lines.append(f"  {i}. [{'x' if it['done'] else ' '}] {it['text']}")
        return "\n".join(lines)

    def execute(self, action: str = "list", steps=None,
                index: int = 0, **_) -> str:
        action = (action or "list").lower()
        try:
            if action == "set":
                if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
                    return "[error] steps 必须是字符串数组"
                self.items = [{"text": s.strip(), "done": False}
                              for s in steps if s.strip()]
                self._emit()
                if not self.items:
                    return "[ok] 计划已清空（0 步）。任务似乎已接近完成，请 finish。"
                return (f"[ok] 已设定 {len(self.items)} 步计划。\n" + self._render()
                        + "\n完成一步后用 plan(action='done', index=N) 把对应步骤勾掉。")
            if action == "list":
                return self._render()
            if action == "clear":
                self.items = []
                self._emit()
                return "[ok] 计划已清空。"
            if action == "done":
                if index < 1 or index > len(self.items):
                    return (f"[error] index 越界，当前共 {len(self.items)} 步。"
                            f"\n{self._render()}")
                self.items[index - 1]["done"] = True
                self._emit()
                return f"[ok] 已勾掉第 {index} 步。\n" + self._render()
            return "[error] 未知 action（set/done/list/clear）"
        except Exception as e:
            return f"[error] 计划更新失败: {type(e).__name__}: {e}"
