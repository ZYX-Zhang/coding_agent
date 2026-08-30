"""命令执行工具：run_command。
"""
from __future__ import annotations

import os
import signal
import subprocess
import time

from .base import Tool

OUT_LIMIT = 8000  # stdout 保留末尾 8000 字符
ERR_LIMIT = 4000  # stderr 保留末尾 4000 字符
DEFAULT_TIMEOUT = 60


def _kill_tree(proc: "subprocess.Popen") -> None:
    """杀掉整个进程组（含子进程），避免孤儿进程。Windows 用 taskkill。"""
    try:
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_cancellable(cmd, cwd, timeout, should_cancel, *, shell=False,
                    out_limit=OUT_LIMIT, err_limit=ERR_LIMIT):
    """可被取消的子进程运行器。

    通过 start_new_session 建独立进程组；poll 循环每 0.1s 检查 should_cancel
    （用户点停止）。命中即杀进程组并返回 ("cancelled", out, err, -1)；超时同理返回
    ("timeout", out, err, -1)。返回 (tag, stdout, stderr, rc)，tag ∈ ok|cancelled|timeout。
    """
    timeout = max(1, int(timeout)) if timeout else None
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=(os.name != "nt"),
    )
    deadline = time.monotonic() + timeout if timeout else None
    while proc.poll() is None:
        if should_cancel and should_cancel():
            _kill_tree(proc)
            out, err = proc.communicate()
            return "cancelled", out or "", "[cancelled] 任务已被用户停止。", -1
        if deadline is not None and time.monotonic() >= deadline:
            _kill_tree(proc)
            out, err = proc.communicate()
            return "timeout", out or "", f"[error] 命令超时（{timeout}s）已终止。", -1
        time.sleep(0.1)
    out, err = proc.communicate()
    return "ok", out or "", err or "", proc.returncode

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

    def __init__(self, workspace: str, should_cancel=None):
        self.workspace = str(workspace)
        self.should_cancel = should_cancel   # 停止时立即杀掉正在跑的命令

    def execute(self, cmd: str = "", timeout: int = DEFAULT_TIMEOUT, **_) -> str:
        if not cmd or not cmd.strip():
            return "[error] cmd 不能为空"
        stripped = cmd.strip()
        if stripped.startswith(BANNED_PREFIXES):
            return (f"[error] 禁止执行交互式/提权命令: {stripped.split()[0]}。"
                    f"请改用非交互替代（如 sudo 用绝对路径、vim 用 sed）。")

        tag, out, err, rc = run_cancellable(
            cmd, self.workspace, timeout, self.should_cancel, shell=True)
        if tag == "cancelled":
            return "[exit=-1] " + err
        if tag == "timeout":
            return err
        out = out[-OUT_LIMIT:]
        err = err[-ERR_LIMIT:]
        parts = [f"[exit={rc}]"]
        if out.strip():
            parts.append(f"stdout:\n{out}")
        if err.strip():
            parts.append(f"stderr:\n{err}")
        if not out.strip() and not err.strip():
            parts.append("（无输出）")
        return "\n".join(parts)
