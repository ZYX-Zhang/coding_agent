"""代码内容搜索工具：search_code。

与 search_files（按文件名 glob）互补：search_code 按【文件内容】搜，
模型问"哪里定义了 add 函数""谁调用了 parse_config"时用它，
输出 文件:行号:匹配行，让模型直接跳到 read_file 的对应行。
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from .base import Tool
from .fs import ListDirTool, _safe_path

MAX_HITS = 50
MAX_LINE_SHOWN = 200          # 单行匹配最长展示字符
MAX_FILE_BYTES = 512 * 1024   # 超过 512KB 的文件跳过


class SearchCodeTool(Tool):
    name = "search_code"
    description = (
        "在工作区内按正则表达式搜索【文件内容】，返回 文件路径:行号:匹配行。\n"
        "用途：查找函数/类/变量的定义与调用位置（如 \"def add\"、\"parse_config\"）。\n"
        "与 search_files 的区别：search_files 按文件名找文件，本工具按内容找位置。\n"
        "参数：pattern 为正则（Python 语法）；glob 可限定文件类型（如 \"*.py\"）；"
        "默认大小写不敏感，case_sensitive=true 开启敏感。最多返回 50 条命中。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "正则表达式"},
            "glob": {"type": "string",
                     "description": "文件名过滤 glob，如 *.py、src/**/*.ts，可选"},
            "case_sensitive": {"type": "boolean",
                               "description": "是否区分大小写，默认 false"},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: str):
        self.workspace = str(Path(workspace).resolve())

    def execute(self, pattern: str = "", glob: str = "**/*",
                case_sensitive: bool = False, **_) -> str:
        try:
            if not pattern:
                return "[error] pattern 不能为空"
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                rex = re.compile(pattern, flags)
            except re.error as e:
                return f"[error] 正则表达式不合法: {e}"

            ws = Path(self.workspace)
            hits: list[str] = []
            scanned = 0

            for p in ws.glob(glob or "**/*"):
                if not p.is_file():
                    continue
                rel_parts = p.relative_to(ws).parts
                # 跳过忽略目录里的文件
                if any(part in ListDirTool.IGNORE or
                       any(fnmatch.fnmatch(part, pat) for pat in ListDirTool.IGNORE)
                       for part in rel_parts[:-1]):
                    continue
                try:
                    if p.stat().st_size > MAX_FILE_BYTES:
                        continue
                    text = p.read_text(encoding="utf-8", errors="strict")
                except (UnicodeDecodeError, OSError, PermissionError):
                    continue    # 二进制或不可读，直接跳过
                scanned += 1
                for i, line in enumerate(text.splitlines(), start=1):
                    if rex.search(line):
                        shown = line.strip()
                        if len(shown) > MAX_LINE_SHOWN:
                            shown = shown[:MAX_LINE_SHOWN] + "…"
                        hits.append(f"{p.relative_to(ws).as_posix()}:{i}: {shown}")
                        if len(hits) >= MAX_HITS:
                            hits.append(f"...（命中超过 {MAX_HITS} 条已截断，"
                                        f"请用更精确的 pattern 或 glob 缩小范围）")
                            return (f"搜索 /{pattern}/（{scanned} 个文件，"
                                    f"{len(hits)} 条命中）:\n" + "\n".join(hits))
                        break   # 每个文件只记首个命中行，够定位了

            if not hits:
                return (f"在 {scanned} 个文件中未找到匹配 /{pattern}/ 的内容。"
                        f"可尝试：放宽大小写、换更短的关键词、或检查 glob 过滤。")
            return f"搜索 /{pattern}/（{scanned} 个文件，{len(hits)} 条命中）:\n" \
                + "\n".join(hits)
        except Exception as e:
            return f"[error] 内容搜索失败: {type(e).__name__}: {e}"