"""终止工具：finish。

模型认为任务完成（或确认无法完成）时主动调用它。循环检测到本次 tool_calls 里
包含 finish 即退出，并把 summary 作为最终答复返回给用户。
"""
from __future__ import annotations

from .base import Tool

FINISH_MARKER = "__FINISHED__"


class FinishTool(Tool):
    name = "finish"
    description = (
        "任务完成时调用本工具结束循环。\n"
        "两种情况都应调用：\n"
        "1. 任务已成功完成——总结你做了什么、改了哪些文件、验证结果如何；\n"
        "2. 确认无法完成——说明卡在哪里、已尝试什么、建议用户怎么办。\n"
        "不调用 finish 而只是输出文字，会被视为任务未完成。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "给用户的最终总结：做了什么、关键改动、验证情况",
            },
        },
        "required": ["summary"],
    }

    def execute(self, summary: str = "", **_) -> str:
        # 返回固定标记，Agent 循环扫描工具结果中的 FINISH_MARKER 判定终止
        return f"{FINISH_MARKER} {summary or '(无总结)'}"
