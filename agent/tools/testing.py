"""测试运行工具：run_tests。

增量价值：结构化失败信息——模型拿到"哪几个套件/用例挂了 + 摘要"，
而不是从几屏乱码里自己捞。

检测顺序（在 workspace 内）：
1. 优先运行项目统一测试入口 tests/run_all.py（若存在）。这是本项目的标准入口
   （编译全部源码 + 跑所有 test_*.py + 汇总），能正确反映"全绿 / 失败"。
2. 否则回退到 pytest（兼容使用 pytest 的项目）。

两者都支持 should_cancel（停止任务时立即杀掉正在跑的测试）与超时。
"""
from __future__ import annotations

import sys
from pathlib import Path

from .base import Tool
from .shell import run_cancellable

DEFAULT_TIMEOUT = 300
RAW_TAIL = 3000          # 原始输出保留末尾字符数


class RunTestsTool(Tool):
    name = "run_tests"
    description = (
        "运行工作区的测试并回传结构化结果：统计 + 失败清单 + 原始输出末尾。\n"
        "检测顺序：① 若工作区存在 tests/run_all.py（本项目标准入口，会编译全部源码并跑"
        "所有 test_*.py 后汇总），优先运行它；② 否则回退到 pytest（需已安装："
        "pip install pytest）。\n"
        "用途：改动代码后验证。比 run_command 直接跑测试的优势是失败信息已提取好。"
        "path 省略时跑整个工作区（run_all 模式下始终跑 tests/run_all.py，忽略 path）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "测试目标（目录或文件），默认整个工作区"},
            "extra_args": {"type": "string",
                           "description": "追加给测试命令的参数，可选"},
            "timeout": {"type": "integer", "description": "超时秒数，默认 300"},
        },
        "required": [],
    }

    def __init__(self, workspace: str, should_cancel=None):
        self.workspace = str(Path(workspace).resolve())
        self.should_cancel = should_cancel   # 停止时立即杀掉正在跑的测试

    def execute(self, path: str = ".", extra_args: str = "",
                timeout: int = DEFAULT_TIMEOUT, **_) -> str:
        try:
            target = Path(self.workspace) / (path or ".")
            if not target.exists():
                return f"[error] 测试目标不存在: {path}"

            # 优先：项目统一测试入口 tests/run_all.py
            run_all = Path(self.workspace) / "tests" / "run_all.py"
            if run_all.exists():
                return self._run_unified(run_all, extra_args, timeout)

            # 回退：pytest
            cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-rA"]
            if extra_args:
                import shlex
                cmd += shlex.split(extra_args)
            cmd.append(str(target))
            tag, out, err, rc = run_cancellable(
                cmd, self.workspace, timeout, self.should_cancel, shell=False)
            if tag == "cancelled":
                return "[exit=-1] " + err
            if tag == "timeout":
                return (f"[error] 测试超时（{timeout}s）。"
                        f"可用 extra_args 加 -x 或 -k 缩小范围。")
            out = (out or "") + (err or "")
            if "No module named pytest" in out:
                return ("[error] 当前 Python 环境未安装 pytest。"
                        "请先运行: pip install pytest")
            stats, failed = self._parse_pytest(out)
            parts = [f"[exit={rc}] {stats}"]
            if failed:
                parts.append("失败用例:")
                parts += [f"  - {f}" for f in failed[:20]]
                if len(failed) > 20:
                    parts.append(f"  ...（共 {len(failed)} 个失败，已截断）")
            parts.append("--- 原始输出末尾 ---")
            parts.append(out[-RAW_TAIL:].strip() or "（无输出）")
            return "\n".join(parts)
        except Exception as e:
            return f"[error] 运行测试失败: {type(e).__name__}: {e}"

    # ---------------------------------------------------------------- #
    def _run_unified(self, run_all: Path, extra_args: str, timeout: int) -> str:
        cmd = [sys.executable, str(run_all)]
        if extra_args:
            import shlex
            cmd += shlex.split(extra_args)
        tag, out, err, rc = run_cancellable(
            cmd, self.workspace, timeout, self.should_cancel, shell=False)
        if tag == "cancelled":
            return "[exit=-1] " + err
        if tag == "timeout":
            return (f"[error] 测试超时（{timeout}s）。"
                    f"可在 tests/run_all.py 中缩小范围或加 timeout。")
        out = (out or "") + (err or "")
        return self._format_unified(rc, out)

    @staticmethod
    def _format_unified(rc: int, out: str) -> str:
        """解析 tests/run_all.py 的输出：提取汇总行与失败套件/用例。"""
        failed = [l.strip() for l in out.splitlines()
                  if l.strip().startswith("[ FAIL ]")
                  or "套件失败" in l
                  or l.strip().startswith("FAIL ")
                  or " 失败 " in l]
        stats_line = ""
        for l in out.splitlines():
            s = l.strip()
            if s.startswith("汇总"):
                stats_line = s
        verdict = "全部通过 ✓" if rc == 0 else "存在失败 ✗"
        parts = [f"[exit={rc}] {stats_line or verdict}"]
        if rc != 0:
            parts.append("失败套件/用例:")
            parts += [f"  - {f}" for f in failed[:20]]
            if len(failed) > 20:
                parts.append(f"  ...（共 {len(failed)} 项失败，已截断）")
        parts.append("--- 原始输出末尾 ---")
        parts.append(out[-RAW_TAIL:].strip() or "（无输出）")
        return "\n".join(parts)

    @staticmethod
    def _parse_pytest(out: str) -> tuple[str, list[str]]:
        """从 pytest -q 输出中提取统计行与失败用例行。"""
        stats = ""
        for line in out.splitlines():
            s = line.strip()
            if re_match_stats(s):
                stats = s
        failed = [l.strip() for l in out.splitlines()
                  if l.strip().startswith(("FAILED", "ERROR"))]
        if not stats:
            stats = f"退出码非零，详见原始输出（解析到 {len(failed)} 个失败项）" \
                if failed else "未能解析统计行，详见原始输出"
        return stats, failed


    # 兼容旧测试/调用方：_parse 即 pytest 解析实现
    _parse = _parse_pytest

def re_match_stats(s: str) -> bool:
    """是否为 pytest 的结果统计行。"""
    import re
    return bool(re.search(r"\d+ (passed|failed|error)", s)) and " in " in s
