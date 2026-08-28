"""命令执行工具：run_command。
"""
from __future__ import annotations

import subprocess

from .base import Tool

OUT_LIMIT = 8000  # stdout 保留末尾 8000 字符
ERR_LIMIT = 4000  # stderr 保留末尾 4000 字符
DEFAULT_TIMEOUT = 60

# 明确禁止的交互式/高危命令前缀（真正的白名单放行逻辑在 safety 模块）
BANNED_PREFIXES = ("sudo ", "vim ", "vi ", "top ", "less ", "more ", "man ")


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "在工作区根目录执行 shell 命令并返回结果。\n"
        "返回格式：[exit=N] + stdout + stderr（各自只保留末尾若干字符）。\n"
        "用途：运行测试/编译/脚本、安装依赖、git status/diff 等查看类操作。\n"
        "注意：命令失败时请仔细阅读 stderr 修复后重试；长命令设置 timeout。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "要执行的完整命令"},
            "timeout": {"type": "integer",
                        "description": "超时秒数，默认 60，长任务（如安装依赖）可调大"},
        },
        "required": ["cmd"],
    }

    def __init__(self, workspace: str):
        self.workspace = str(workspace)

    def execute(self, cmd: str = "", timeout: int = DEFAULT_TIMEOUT, **_) -> str:
        if not cmd or not cmd.strip():
            return "[error] cmd 不能为空"
        stripped = cmd.strip()
        if stripped.startswith(BANNED_PREFIXES):
            return (f"[error] 禁止执行交互式/提权命令: {stripped.split()[0]}。"
                    f"请改用非交互替代（如 sudo 用绝对路径、vim 用 sed）。")

        try:
            r = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout)),
                cwd=self.workspace,
            )
            out = (r.stdout or "")[-OUT_LIMIT:]
            err = (r.stderr or "")[-ERR_LIMIT:]

            parts = [f"[exit={r.returncode}]"]
            if out.strip():
                parts.append(f"stdout: \n{out}")
            if err.strip():
                parts.append(f"stderr: \n{err}")
            if not out.strip() and not err.strip():
                parts.append("（无输出）")
            return "\n".join(parts)

        except subprocess.TimeoutExpired:
            return f"[error] 命令超时（{timeout}s）已终止。请优化命令或增大 timeout。"
        except Exception as e:
            return f"[error] 命令执行异常: {type(e).__name__}: {e}"
