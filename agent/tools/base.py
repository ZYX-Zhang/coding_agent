"""工具基类与注册表。

约定：
- 每个工具 = name + description + JSON Schema + execute()
- execute() 永远返回字符串（错误也返回 "[error] ..." 文本，绝不抛异常）
- 工具本身无状态、不认识 Agent 循环，只接收参数、返回结果
"""
from __future__ import annotations


class Tool:
    #: 工具名，模型在 tool_calls 中引用的名字
    name: str = ""
    #: 给模型看的使用说明：何时用、参数含义、返回格式。写越清楚，成功率越高
    description: str = ""
    #: 参数的 JSON Schema（OpenAI tool calling 格式）
    parameters: dict = {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs) -> str:
        raise NotImplementedError

    def to_schema(self) -> dict:
        """转换为 Chat Completions tools 参数中的条目。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：Agent 循环只认识它，不认识任何具体工具。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]
