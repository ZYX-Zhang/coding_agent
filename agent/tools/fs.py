"""文件系统类工具：read_file / write_file / edit_file / list_dir / search_files。
"""
from __future__ import annotations

import difflib
import fnmatch
import os
from pathlib import Path

from .base import Tool

READ_MAX_LINES = 2000  # 超过则只保留头尾各 500 行
READ_TAIL_LINES = 500
MAX_FILE_BYTES = 2 * 1024 * 1024  # 超过 2MB 的文件拒绝整读


def _safe_path(workspace: str, rel_path: str) -> Path:
    """把模型给的路径解析为 workspace 内的绝对路径；越界直接抛 ValueError。

    ValueError 会被 execute 外层的包装捕获并转成 [error] 文本。
    """
    ws = Path(workspace).resolve()
    p = Path(rel_path)
    # 相对路径基于 workspace；绝对路径也强制收编到 workspace 下校验
    if not p.is_absolute():
        p = ws / p
    p = Path(os.path.realpath(p))
    if not (p == ws or ws in p.parents):
        raise ValueError(
            f"路径越界：{rel_path} 解析后为 {p}，不在工作区 {ws} 内"
        )
    return p


class FileSystemTool(Tool):
    """公共基类：持有 workspace 引用。"""

    def __init__(self, workspace: str):
        self.workspace = str(Path(workspace).resolve())


class ReadFileTool(FileSystemTool):
    name = "read_file"
    description = (
        "读取工作区内一个文本文件的内容，返回带行号的文本。\n"
        "用途：在修改任何文件之前先读它，确认现状，不要凭记忆猜测文件内容。\n"
        "超过 2000 行的文件只返回开头和结尾各 500 行，中间用 ... 省略。\n"
        "可选参数 start_line / end_line（从 1 开始计数，含端点）读取指定范围。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作区的文件路径"},
            "start_line": {"type": "integer", "description": "起始行号（从 1 开始，可选）"},
            "end_line": {"type": "integer", "description": "结束行号（含，可选）"},
        },
        "required": ["path"],
    }

    def execute(self, path: str = "", start_line: int | None = None,
                end_line: int | None = None, **_) -> str:
        try:
            p = _safe_path(self.workspace, path)
            if not p.exists():
                return f"[error] 文件不存在: {path}"
            if p.is_dir():
                return f"[error] {path} 是目录，请用 list_dir"
            if p.stat().st_size > MAX_FILE_BYTES:
                return (f"[error] 文件过大（{p.stat().st_size} 字节），"
                        f"请用 start_line/end_line 分段读取")

            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            total = len(lines)

            lo, hi = 1, total
            if start_line is not None:
                lo = max(1, start_line)
            if end_line is not None:
                hi = min(total, end_line)
            sel = lines[lo - 1:hi]

            if len(sel) > READ_MAX_LINES:
                head, tail = sel[:READ_TAIL_LINES], sel[-READ_TAIL_LINES:]
                omitted = len(sel) - READ_TAIL_LINES * 2
                body = (
                        [f"{i:>5} | {s}" for i, s in enumerate(head, start=lo)]
                        + [f"       ...（中间省略 {omitted} 行，如需查看请用 start_line/end_line）"]
                        + [f"{i:>5} | {s}" for i, s in enumerate(tail, start=lo + len(head) + omitted)]
                )
            else:
                body = [f"{i:>5} | {s}" for i, s in enumerate(sel, start=lo)]

            return f"{path}（共 {total} 行，显示 {lo}-{min(hi, total)}）\n" + "\n".join(body)
        except ValueError as e:  # 路径越界
            return f"[error] {e}"
        except UnicodeDecodeError:
            return f"[error] 非 UTF-8 文本文件（可能是二进制）: {path}"
        except Exception as e:  # 兜底：任何异常都转为文本
            return f"[error] 读取失败: {type(e).__name__}: {e}"


class WriteFileTool(FileSystemTool):
    name = "write_file"
    description = (
        "创建或整体覆写一个文件（自动创建父目录）。\n"
        "用途：新建文件，或对已有文件做大改动时直接全量写入。\n"
        "注意：会覆盖原内容；小幅修改已有文件优先用 edit_file。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作区的文件路径"},
            "content": {"type": "string", "description": "完整的文件内容"},
        },
        "required": ["path", "content"],
    }

    def execute(self, path: str = "", content: str = "", **_) -> str:
        try:
            p = _safe_path(self.workspace, path)
            existed = p.exists()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            n = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            tag = "覆写" if existed else "新建"
            # 覆写类操作是否放行由上层 safety 模块决定，工具本身只管执行
            return f"[ok] 已{tag} {path}（{max(n, 0)} 行，{len(content.encode('utf-8'))} 字节）"
        except ValueError as e:
            return f"[error] {e}"
        except Exception as e:
            return f"[error] 写入失败: {type(e).__name__}: {e}"


class EditFileTool(FileSystemTool):
    name = "edit_file"
    description = (
        "对已有文件做精确字符串替换：把 old_str 替换为 new_str。\n"
        "要求 old_str 在文件中【唯一】；若不唯一，请扩大 old_str 范围（多带几行上下文）"
        "直到唯一。\n"
        "new_str 传空字符串即为删除 old_str。"
        "修改前请先 read_file 确认文件现状，不要凭记忆修改。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作区的文件路径"},
            "old_str": {"type": "string", "description": "要被替换的唯一原文片段"},
            "new_str": {"type": "string", "description": "替换后的内容"},
        },
        "required": ["path", "old_str", "new_str"],
    }

    def execute(self, path: str = "", old_str: str = "", new_str: str = "", **_) -> str:
        try:
            if not old_str:
                return "[error] old_str 不能为空"
            p = _safe_path(self.workspace, path)
            if not p.exists():
                return f"[error] 文件不存在: {path}"

            text = p.read_text(encoding="utf-8", errors="replace")
            count = text.count(old_str)

            if count == 0:
                # 给出最相近的片段，帮助模型自我纠正
                hint = self._nearby_hint(text, old_str)
                return (f"[error] 未找到 old_str。{hint}")
            if count > 1:
                first = text.find(old_str)
                line_no = text[:first].count("\n") + 1
                return (f"[error] old_str 出现了 {count} 次（首次在第 {line_no} 行），"
                        f"不唯一。请扩大 old_str 的上下文范围使其唯一。")

            new_text = text.replace(old_str, new_str, 1)
            p.write_text(new_text, encoding="utf-8")
            return self._diff_summary(path, text, new_text)
        except ValueError as e:
            return f"[error] {e}"
        except Exception as e:
            return f"[error] 编辑失败: {type(e).__name__}: {e}"

    @staticmethod
    def _nearby_hint(text: str, old_str: str) -> str:
        """old_str 没匹配上时，用相似度找出最像的行，提示模型可能是哪里的差异。"""
        target_lines = [l.strip() for l in old_str.splitlines() if l.strip()]
        if not target_lines:
            return ""
        best = difflib.get_close_matches(
            target_lines[0], [l.strip() for l in text.splitlines()], n=1, cutoff=0.6
        )
        if best:
            return f"文件中存在相似行：{best[0]!r}，请重新 read_file 核对（常见原因：缩进/空格不一致）。"
        return "请重新 read_file 核对原文。"

    @staticmethod
    def _diff_summary(path: str, old_text: str, new_text: str) -> str:
        diff = list(difflib.unified_diff(
            old_text.splitlines(), new_text.splitlines(),
            fromfile=f"{path}(旧)", tofile=f"{path}(新)", lineterm="",
        ))
        body = "\n".join(diff[:40])
        if len(diff) > 40:
            body += f"\n...（diff 共 {len(diff)} 行，已截断）"
        return f"[ok] 已修改 {path}\n{body}"


class ListDirTool(FileSystemTool):
    name = "list_dir"
    description = (
        "列出目录内容，返回带类型标记（文件/目录）的清单。\n"
        "用途：了解项目结构、查找文件位置。自动忽略 .git、__pycache__、node_modules 等。\n"
        "recursive=true 时递归列出子目录（最多 3 层，文件数上限 500）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作区的目录路径，默认为工作区根目录"},
            "recursive": {"type": "boolean", "description": "是否递归子目录，默认 false"},
        },
        "required": [],
    }

    IGNORE = {".git", "__pycache__", "node_modules", ".venv", "venv",
              ".idea", ".vscode", ".DS_Store", "*.pyc"}

    def execute(self, path: str = ".", recursive: bool = False, **_) -> str:
        try:
            p = _safe_path(self.workspace, path)
            if not p.exists():
                return f"[error] 目录不存在: {path}"
            if not p.is_dir():
                return f"[error] {path} 不是目录"

            rows, count = [], [0]

            def walk(d: Path, depth: int):
                if depth > 3 or count[0] > 500:
                    return
                try:
                    entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name))
                except PermissionError:
                    rows.append("  " * depth + "[无权限]")
                    return
                for ent in entries:
                    if any(fnmatch.fnmatch(ent.name, pat) for pat in self.IGNORE):
                        continue
                    if count[0] > 500:
                        rows.append("  " * depth + "...（条目过多已截断）")
                        return
                    count[0] += 1
                    mark = "d " if ent.is_dir() else "f "
                    rows.append("  " * depth + mark + ent.name)
                    if ent.is_dir() and recursive:
                        walk(ent, depth + 1)

            walk(p, 0)
            header = f"{path}（{'递归' if recursive else '单层'}，共 {count[0]} 项）"
            return header + "\n" + "\n".join(rows)
        except ValueError as e:
            return f"[error] {e}"
        except Exception as e:
            return f"[error] 列目录失败: {type(e).__name__}: {e}"


class SearchFilesTool(FileSystemTool):
    name = "search_files"
    description = (
        "在工作区内按 glob 模式查找文件路径（如 \"**/*.py\"、\"src/**/*.ts\"、\"*.json\"）。\n"
        "用途：按文件名找文件，返回相对路径清单（上限 200 条）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "glob 模式，如 **/*.py"},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str = "**/*", **_) -> str:
        try:
            ws = Path(self.workspace)
            hits = []
            for p in ws.glob(pattern):
                rel = p.relative_to(ws).as_posix()
                if any(part in ListDirTool.IGNORE for part in rel.split("/")[:-1]):
                    continue
                hits.append(("d " if p.is_dir() else "f ") + rel)
                if len(hits) >= 200:
                    hits.append("...（超过 200 条已截断，请缩小模式范围）")
                    break
            if not hits:
                return f"未找到匹配 {pattern} 的文件"
            return f"匹配 {pattern} 的结果（{len(hits)} 条）:\n" + "\n".join(sorted(hits))
        except Exception as e:
            return f"[error] 搜索失败: {type(e).__name__}: {e}"
