"""工具包入口：注册表 + 默认工具集装配。

用法：
    from agent.tools import build_registry

    registry = build_registry(workspace="/path/to/project")
    registry.schemas()          # -> 传给 LLM 的 tools 参数
    tool = registry.get("read_file")
    result = tool.execute(path="src/main.py")
"""
from .base import Tool, ToolRegistry
from .finish import FinishTool, FINISH_MARKER
from .fs import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
    _safe_path,
)
from .shell import RunCommandTool
from .interactive import AskUserTool, PlanTool
from .net import FetchUrlTool
from .search import SearchCodeTool

__all__ = [
    "Tool", "ToolRegistry",
    "ReadFileTool", "WriteFileTool", "EditFileTool",
    "ListDirTool", "SearchFilesTool", "AskUserTool", "PlanTool", "FetchUrlTool",
    "SearchCodeTool", "RunCommandTool", "FinishTool", "FINISH_MARKER",
    "build_registry", "_safe_path",
]


def build_registry(workspace: str, llm=None, ask=None,
                   should_cancel=None) -> ToolRegistry:
    """装配默认工具集。Agent 循环只拿这个注册表，加新工具不改循环。"""
    reg = ToolRegistry()
    # 顺序即下发给模型的 tools 顺序，finish 放最后
    reg.register(ReadFileTool(workspace))  # 1. 读文件
    reg.register(WriteFileTool(workspace))  # 2. 写文件
    reg.register(EditFileTool(workspace))  # 3. 精确编辑
    reg.register(ListDirTool(workspace))  # 4. 看目录
    reg.register(SearchFilesTool(workspace))  # 5. 找文件
    reg.register(RunCommandTool(workspace, should_cancel))  # 6. 执行命令
    reg.register(FinishTool())  # 7. 终止（不属于"手脚"，是出口）
    reg.register(PlanTool())  # 8. 任务计划管理
    reg.register(AskUserTool(ask))  # 9. 向用户提问
    reg.register(FetchUrlTool())  # 10. 获取文档
    reg.register(SearchCodeTool(workspace))  # 11.  按内容找（正则）
    return reg
