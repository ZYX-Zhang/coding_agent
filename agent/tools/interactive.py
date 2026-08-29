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
        "制定或更新当前任务的执行计划：传入完整的步骤列表（覆盖旧计划）。\n"
        "用法：复杂任务（多文件/多阶段）开始前先 plan 一次；"
        "每完成一步就传入剩余步骤更新计划，再继续执行。\n"
        "返回带勾选框的清单，帮你（模型）自己保持进度条清晰。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "完整步骤列表（已完成的不用再传）",
            },
        },
        "required": ["steps"],
    }

    def __init__(self):
        self.steps: list[str] = []  # 每个注册表一个实例 => 会话隔离
        self.done: int = 0  # 历史累计已完成步数

    def execute(self, steps=None, **_) -> str:
        try:
            if steps is None:
                steps = []
            if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
                return "[error] steps 必须是字符串数组"
            self.done += max(0, len(self.steps) - len(steps))
            self.steps = [s.strip() for s in steps if s.strip()]

            if not self.steps:
                return (f"[ok] 计划已清空（此前共完成 {self.done} 步）。"
                        f"任务似乎已接近完成，请收尾并 finish。")
            lines = [f"[当前计划] 已完成 {self.done} 步，剩余 {len(self.steps)} 步:"]
            lines += [f"  {' '.join(('[ ]', s))}" for s in self.steps]
            lines.append("执行完当前步骤后，请用剩余步骤再次调用 plan 更新进度。")
            return "\n".join(lines)
        except Exception as e:
            return f"[error] 计划更新失败: {type(e).__name__}: {e}"
