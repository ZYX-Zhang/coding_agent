"""安全策略与操作确认

分层策略（从宽到严）：
    allow   只读操作，直接放行
    confirm 写文件 / 执行命令，需要用户确认（y / n / a=本次会话全放行）
    deny    高危命令，无条件拒绝（用户确认也不行）

"""
from __future__ import annotations

import re
import shlex

# ---------------------------------------------------------------- #
# 工具分级
# ---------------------------------------------------------------- #
READONLY_TOOLS = {"read_file", "list_dir", "search_files", "finish"}
CONFIRM_TOOLS = {"write_file", "edit_file", "run_command"}

# ---------------------------------------------------------------- #
# 高危命令模式（deny：无论怎么确认都不执行）
# ---------------------------------------------------------------- #
_DENY_PATTERNS = [
    (r"rm\s+(-[a-z]*\s+)*-?[rf]{1,2}[a-z]*\s+(/|~|\$HOME)", "递归删除系统目录"),
    (r"rm\s+-[a-z]*[rf]", "递归/强制删除"),
    (r"\bmkfs(\.\w+)?\b", "格式化磁盘"),
    (r"\bdd\s+.*\bof=/dev/", "裸写磁盘设备"),
    (r":\(\)\s*\{.*\};\s*:", "fork 炸弹"),
    (r">\s*/dev/sd[a-z]", "覆写磁盘设备"),
    (r"chmod\s+-R\s+777\s+/", "全盘放开权限"),
    (r"curl[^|]*\|\s*(ba)?sh", "管道执行远程脚本"),
    (r"wget[^|]*\|\s*(ba)?sh", "管道执行远程脚本"),
    (r"\bgit\s+push\s+.*--force", "强推远端"),
    (r"\bshutdown\b|\breboot\b|\bhalt\b", "关机/重启"),
]
_DENY_RE = [(re.compile(p, re.S), why) for p, why in _DENY_PATTERNS]

# 写文件目标里同样危险的路径
_DENY_PATHS = ("/etc/", "/dev/", "/sys/", "/proc/", "/bin/", "/sbin/",
               "/usr/bin/", "/usr/sbin/", "id_rsa", ".ssh/authorized_keys")


def _split_cmd(cmd: str) -> list[str]:
    """尽量安全地切分命令首词；失败就按空白切。"""
    try:
        return shlex.split(cmd)
    except ValueError:
        return cmd.split()


class SafetyPolicy:
    """纯决策：无 IO、无状态，随便单测。"""

    def classify(self, tool_name: str, args: dict) -> tuple[str, str]:
        """返回 (决策, 原因)。决策 ∈ {"allow", "confirm", "deny"}。"""
        if tool_name in READONLY_TOOLS:
            return "allow", "只读操作"

        # ---- 高危路径检查（对写类工具的 path 参数） ----
        path = str(args.get("path", ""))
        if tool_name in ("write_file", "edit_file"):
            for bad in _DENY_PATHS:
                if bad in path:
                    return "deny", f"写入敏感路径 {path}"
            return "confirm", f"写文件 {path}"

        # ---- 命令检查 ----
        if tool_name == "run_command":
            cmd = str(args.get("cmd", ""))
            low = cmd.lower()
            for rex, why in _DENY_RE:
                if rex.search(cmd):
                    return "deny", why
            argv = _split_cmd(cmd)
            if argv and argv[0] in ("sudo", "su"):
                return "deny", "提权命令"
            return "confirm", f"执行命令: {cmd[:60]}"

        # 未知工具（将来新增的）默认要确认，白名单才会 allow
        return "confirm", f"未分级工具 {tool_name}"


class ConfirmSafety:
    """
    Agent 只调用 check(tool, args) -> bool。
    """

    def __init__(self, ask=None, policy: SafetyPolicy | None = None):
        """
        :param ask: ask(prompt) -> str，返回 "y"/"n"/"a"（大小写不敏感）。
                    CLI 传基于 input 的实现；测试传脚本。
                    None 表示用内置 input()。
        """
        self.policy = policy or SafetyPolicy()
        self.ask = ask or (lambda prompt: input(prompt))
        self.auto_allowed = False  # 用户按过 "a" 后本会话全放行
        self.denied: list[tuple[str, str]] = []  # 审计记录
        self.confirmed: list[str] = []

    def check(self, tool, args: dict) -> bool:
        name = getattr(tool, "name", str(tool))
        decision, reason = self.policy.classify(name, args or {})

        if decision == "allow":
            return True
        if decision == "deny":
            self.denied.append((name, reason))
            return False

        # auto 模式（--yes 或用户按过 a）：写操作直接放行
        if self.auto_allowed:
            self.confirmed.append(f"{name}: {reason} (auto)")
            return True

        answer = self.ask(f"⚠ {reason}\n   允许执行? [y/n/a(本次会话全部允许)] "
                          ).strip().lower()
        if answer in ("a", "always"):
            self.auto_allowed = True
            self.confirmed.append(f"{name}: {reason} (all)")
            return True
        if answer in ("y", "yes"):
            self.confirmed.append(f"{name}: {reason}")
            return True
        self.denied.append((name, reason))
        return False


class AutoSafety:
    """全自动（--yes 模式）：deny 依旧拦截，其余全放行。"""

    def __init__(self, policy: SafetyPolicy | None = None):
        self.policy = policy or SafetyPolicy()
        self.denied: list[tuple[str, str]] = []

    def check(self, tool, args: dict) -> bool:
        name = getattr(tool, "name", str(tool))
        decision, reason = self.policy.classify(name, args or {})
        if decision == "deny":
            self.denied.append((name, reason))
            return False
        return True
