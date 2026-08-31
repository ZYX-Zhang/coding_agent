"""agent CLI

用法：
    python -m myagent.cli [workspace]                     # 交互模式
    python -m myagent.cli . --task "修复 bug 并跑测试"     # 单任务模式（跑完即退）
    python -m myagent.cli . --resume ~/.agent/sessions/xxx.jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path

from .agent import Agent
from .context import Context
from .llm import LLMClient, LLMError
from .safety import AutoSafety, ConfirmSafety
from .session import Session
from .tools import build_registry


# ---------------------------------------------------------------- #
# 终端颜色（自动探测；NO_COLOR / 非 tty 时关闭）
# ---------------------------------------------------------------- #
class _C:
    enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    DIM, BOLD = "\033[2m", "\033[1m"
    CYAN, GREEN, YELLOW, RED, BLUE, MAGENTA = ("\033[36m", "\033[32m",
                                               "\033[33m", "\033[31m",
                                               "\033[34m", "\033[35m")
    RESET = "\033[0m"


def paint(code: str, s: str) -> str:
    return f"{code}{s}{_C.RESET}" if _C.enabled else s


# ---------------------------------------------------------------- #
# 工作区目录树（注入 system prompt，让模型开局有地图）
# ---------------------------------------------------------------- #
_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
                ".mypy_cache", ".pytest_cache", ".myagent", ".idea", ".vscode"}


def dir_tree(root: str, max_depth: int = 2, max_entries: int = 40) -> str:
    """生成目录树文本（限深度/条数），用于 system prompt 注入。"""
    lines, count = [f"{root}/"], 0

    def walk(d: Path, prefix: str, depth: int):
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return
        try:
            entries = sorted(d.iterdir(),
                             key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for p in entries:
            if count >= max_entries:
                lines.append(prefix + "…（已截断）")
                return
            if p.name in _IGNORE_DIRS or p.name.startswith("."):
                continue
            count += 1
            branch = "├── " if count < max_entries else "└── "
            name = p.name + ("/" if p.is_dir() else "")
            lines.append(prefix + branch + name)
            if p.is_dir():
                walk(p, prefix + ("│   " if count < max_entries else "    "),
                     depth + 1)

    walk(Path(root), "", 1)
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """你是运行在用户本机终端里的编程智能体，通过调用工具自主完成编程任务。

# 工作区
当前工作目录：{workspace}
目录结构（可能不完整，可用 list_dir / search_files 探索）：
{tree}

# 工作规则
1. 动手前先看：修改文件前必须先 read_file 了解现状；不熟悉的目录先 list_dir。
2. 自测闭环（强制）：任何代码改动（新建/修改 .py 或前端逻辑）完成后，必须调用
   run_tests 验证——它会在本项目自动优先运行 tests/run_all.py 统一入口（编译全部
   源码 + 跑所有测试并汇总）。规则：全部通过才可继续或 finish；若失败，必须读取
   失败信息、定位并修复代码，再次调用 run_tests，循环修复。单轮任务内最多重试 5 次，
   仍失败则向用户说明实情并停止——绝不在测试未通过时谎报"已完成"。
3. 诚实汇报：工具返回 [error] 时先分析原因再重试，禁止假装成功。
4. 结束条件：只有 run_tests 全部通过后，才可调用 finish 汇报完成
   （做了什么、验证结果、遗留问题）；任务确认无法继续时也要 finish 并如实说明。
   不要用普通文字代替 finish。
5. 只在工作区内操作，不要触碰工作区外的文件。
6. 任务计划（强制）：凡涉及写代码、改代码、多文件改动或明显多步骤的任务，开局必须先用
   plan 工具列出完整步骤（action="set"），再逐步执行；每完成一步立即调用 plan(done) 勾选，
   让进度面板实时反映真实进度（不能等到写完才补列）。纯问答、单条查询、只读探索可不列计划。"""


def build_system_prompt(workspace: str) -> str:
    try:
        tree = dir_tree(workspace)
    except OSError:
        tree = "（目录树生成失败，请用 list_dir 自行探索）"
    return SYSTEM_PROMPT_TEMPLATE.format(workspace=workspace, tree=tree)


# ---------------------------------------------------------------- #
# CLI 应用主体
# ---------------------------------------------------------------- #
HELP_TEXT = """命令：
  /help              显示本帮助
  /new               开始新会话（当前会话已自动保存）
  /save [file]       手动保存会话（默认自动保存到会话文件）
  /resume <file>     加载历史会话继续聊
  /stats             统计：轮次 / token 估算 / 工具调用
  /ctx               当前上下文占用 vs 预算
  /tree              重新查看工作区目录树
  /auto              切换 自动确认模式（写操作不再逐个询问）
  /exit  /quit       退出（Ctrl+D 同效）
其它任何输入都会作为任务交给 agent 执行。"""


class CliApp:
    def __init__(self, llm, workspace: str, *,
                 session_dir: str | None = None,
                 auto: bool = False,
                 max_turns: int = 40,
                 input_fn=input,
                 out_fn=None,
                 stream: bool = True):
        self.llm = llm
        self.workspace = str(Path(workspace).resolve())
        self.session_dir = Path(session_dir or
                                Path(self.workspace) / ".agent" / "sessions")
        self.auto = auto
        self.max_turns = max_turns
        self.input_fn = input_fn
        self._out_fn = out_fn
        self.stream = stream

        self.safety: ConfirmSafety | AutoSafety | None = None
        self.session: Session | None = None
        self._saved_len = 0            # 已落盘的消息数（增量追加）
        self._streamed = False         # 本轮是否已有流式输出
        self._tool_calls_total = 0
        self._setup_safety()
        self._new_session()

    # ---------------------------------------------------------------- #
    # 装配
    # ---------------------------------------------------------------- #
    def out(self, text: str = "") -> None:
        (self._out_fn if self._out_fn else print)(text)

    def _setup_safety(self) -> None:
        def ask(prompt: str) -> str:
            self.out(paint(_C.YELLOW, prompt))
            try:
                return self.input_fn("   > ")
            except EOFError:
                return "n"

        if self.auto:
            self.safety = AutoSafety()
        else:
            self.safety = ConfirmSafety(ask=ask)

    def _new_session(self) -> None:
        if self.session:
            self.session.close()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session = Session(self.session_dir / f"{ts}.jsonl")
        self.ctx = Context(system_prompt=build_system_prompt(self.workspace))
        self.agent = Agent(self.llm, build_registry(self.workspace),
                           context=self.ctx, safety=self.safety,
                           max_turns=self.max_turns,
                           on_event=self._on_event,
                           on_delta=self._on_delta if self.stream else None)
        self._saved_len = 0

    def _ask_user(self, question: str) -> str:
        """ask_user 工具的终端实现：渲染问题并读一行回答。"""
        self.out(paint(_C.CYAN, f"\n❓ {question}"))
        try:
            return self.input_fn(paint(_C.CYAN, "回答 > ")).strip()
        except EOFError:
            return ""

    # ---------------------------------------------------------------- #
    # 事件渲染（Agent 循环回调 -> 终端输出）
    # ---------------------------------------------------------------- #
    def _on_delta(self, text: str) -> None:
        self._streamed = True
        self.out_fn_raw(text)

    def out_fn_raw(self, text: str) -> None:
        """流式片段：不换行直写（测试时注入的 out_fn 也要能收）。"""
        if self._out_fn:
            self._out_fn(text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _on_event(self, kind: str, payload: dict) -> None:
        if kind == "turn_start":
            bar = "─" * 34
            self.out(paint(_C.DIM, f"── 轮次 {payload['turn']} {bar}"))
        elif kind == "assistant":
            content = payload.get("content") or ""
            if self._streamed:
                self.out()  # 流式已打印，补个换行
                self._streamed = False
            elif content.strip():
                self.out(paint(_C.BLUE, content))
        elif kind == "tool_start":
            name, args = payload["name"], payload["args"]
            key = (args.get("path") or args.get("cmd")
                   or args.get("pattern") or "")
            self.out(paint(_C.CYAN, f"  ⚒ {name}  {str(key)[:72]}"))
        elif kind == "tool_end":
            result = payload["result"]
            first = result.splitlines()[0] if result else "(空)"
            if len(first) > 78:
                first = first[:75] + "…"
            if first.startswith("[error]"):
                color = _C.RED
            elif first.startswith("[blocked]"):
                color = _C.YELLOW
            elif first.startswith("[ok]") or first.startswith("[exit=0]"):
                color = _C.GREEN
            else:
                color = _C.DIM
            self.out(paint(color, "  │  " + first))
            self._tool_calls_total += 1
        elif kind == "finish":
            self.out()
            self.out(paint(_C.GREEN + _C.BOLD, f"✔ 完成：{payload['summary']}"))
            self.out()

    # ---------------------------------------------------------------- #
    # 会话持久化（增量追加）
    # ---------------------------------------------------------------- #
    def _flush_session(self) -> None:
        if not self.session:
            return
        fresh = self.ctx.messages[self._saved_len:]
        if fresh:
            self.session.extend(fresh)
            self._saved_len = len(self.ctx.messages)

    # ---------------------------------------------------------------- #
    # REPL 主循环
    # ---------------------------------------------------------------- #
    def repl(self) -> int:
        self._banner()
        while True:
            try:
                line = self.input_fn(paint(_C.GREEN + _C.BOLD, "❯ ")).strip()
            except (EOFError, KeyboardInterrupt):
                self.out()
                break
            if not line:
                continue
            if line.startswith("/"):
                if self.handle_command(line):
                    break
                continue
            self._run_task(line)
        self._flush_session()
        if self.session:
            self.session.close()
        self.out(paint(_C.DIM, "会话已保存，再见。"))
        return 0

    def _run_task(self, task: str) -> None:
        self.out()  # 与提示符空一行
        try:
            summary = self.agent.run(task)
        except KeyboardInterrupt:
            self.out()
            self.out(paint(_C.YELLOW,
                           "⏹ 任务被中断（历史已保留，可继续对话让它接着干）。"))
        except LLMError as e:
            self.out(paint(_C.RED, f"✘ 模型调用失败：{e}"))
            self.out(paint(_C.DIM, "  任务未执行。可检查网络/API key 后重试。"))
        else:
            if not summary.startswith(("已达到最大轮数", "模型连续")):
                pass  # finish 摘要已由事件打印
            else:
                self.out(paint(_C.YELLOW, f"⚠ {summary}"))
        finally:
            self._flush_session()

    # ---------------------------------------------------------------- #
    # 斜杠命令
    # ---------------------------------------------------------------- #
    def handle_command(self, line: str) -> bool:
        """返回 True 表示请求退出 REPL。"""
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")

        if cmd in ("/exit", "/quit", "/q"):
            return True
        if cmd == "/help":
            self.out(HELP_TEXT)
        elif cmd == "/new":
            self._flush_session()
            self._new_session()
            self.out(paint(_C.GREEN, f"✔ 新会话：{self.session.path.name}"))
        elif cmd == "/save":
            self._flush_session()
            self.out(paint(_C.GREEN, f"✔ 已保存到 {self.session.path}"))
        elif cmd == "/resume":
            if not arg:
                self.out(paint(_C.YELLOW, "用法: /resume <会话文件.jsonl>"))
            elif not Path(arg).exists():
                self.out(paint(_C.YELLOW, f"文件不存在: {arg}"))
            else:
                self._flush_session()
                self._new_session()
                from .session import resume_context
                before = len(self.ctx.messages)
                resume_context(arg, self.ctx)
                loaded = len(self.ctx.messages) - before
                self._saved_len = len(self.ctx.messages)
                self.out(paint(_C.GREEN,
                               f"✔ 已恢复 {loaded} 条消息，继续。"))
        elif cmd == "/stats":
            est = self.ctx.estimate_total()
            self.out(f"轮次内工具调用: {self._tool_calls_total}")
            self.out(f"上下文估算: {est:,} tokens / 预算 {self.ctx.max_tokens:,}")
            self.out(f"校准系数: {self.ctx._calib}")
            self.out(f"会话文件: {self.session.path}")
            denied = getattr(self.safety, "denied", [])
            if denied:
                self.out(paint(_C.YELLOW,
                               f"被拒绝操作: {len(denied)} 次（含高危拦截）"))
        elif cmd == "/ctx":
            est = self.ctx.estimate_total()
            pct = est * 100 // max(self.ctx.max_tokens, 1)
            self.out(f"上下文占用: {est:,} / {self.ctx.max_tokens:,} "
                     f"({pct}%)，历史 {len(self.ctx.messages)} 条消息")
        elif cmd == "/tree":
            self.out(dir_tree(self.workspace))
        elif cmd == "/auto":
            self.auto = not self.auto
            self._setup_safety()
            self.agent.safety = self.safety
            mode = "自动确认（写操作不询问）" if self.auto else "逐项确认"
            self.out(paint(_C.YELLOW, f"安全模式切换为: {mode}"))
        else:
            self.out(paint(_C.YELLOW, f"未知命令 {cmd}，/help 查看可用命令"))
        return False

    # ---------------------------------------------------------------- #
    def _banner(self) -> None:
        model = getattr(self.llm, "model", "mock")
        mode = "自动确认" if self.auto else "写操作需确认"
        self.out(paint(_C.BOLD + _C.MAGENTA,
                       "CodingAgent — 编程智能体 (CLI)"))
        self.out(paint(_C.DIM,
                       f"模型: {model} | 工作区: {self.workspace}"))
        self.out(paint(_C.DIM,
                       f"会话: {self.session.path} | 安全: {mode}"))
        self.out(paint(_C.DIM, "输入任务开始，/help 查看命令。"))


# ---------------------------------------------------------------- #
# 入口
# ---------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="CodingAgent",
        description="编程智能体 CLI")
    ap.add_argument("workspace", nargs="?", default="./workspace",
                    help="工作区目录（默认当前目录）")
    ap.add_argument("--api-key", default=None,
                    help="默认读环境变量 ZHIPUAI_API_KEY")
    ap.add_argument("--task", default=None,
                    help="单任务模式：执行完该任务后退出（适合跑基准）")
    ap.add_argument("--resume", default=None, metavar="FILE",
                    help="恢复历史会话")
    ap.add_argument("--session-dir", default=None)
    ap.add_argument("-y", "--yes", action="store_true",
                    help="自动确认所有写操作（高危命令仍拦截）")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args(argv)

    workspace = str(Path(args.workspace).resolve())
    if not Path(workspace).is_dir():
        os.mkdir(workspace)
        os.chmod(workspace, 0o777)
        # print(f"错误: 工作区不存在 {workspace}", file=sys.stderr)
        # return 2

    try:
        llm = LLMClient()
    except LLMError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2

    app = CliApp(llm, workspace, session_dir=args.session_dir,
                 auto=args.yes, max_turns=args.max_turns,
                 stream=not args.no_stream)

    if args.resume:
        app.handle_command(f"/resume {args.resume}")

    if args.task:
        app._run_task(args.task)
        app._flush_session()
        return 0
    return app.repl()


if __name__ == "__main__":
    sys.exit(main())
