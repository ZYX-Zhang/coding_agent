"""测试运行工具：run_tests。

支持多语言：自动探测工作区使用的语言/测试框架，运行对应测试命令，
并提取结构化的统计与失败清单（模型无需从几屏乱码里自己捞）。

检测顺序：
1. 若目标是整个工作区且存在 tests/run_all.py（本项目标准入口），优先运行它。
2. 若目标是单个文件，按扩展名选择对应语言的 Runner，并生成针对该文件的
   测试命令（如 Python 脚本直接执行、Go 测试所在包等）。
3. 若目标是目录/整个工作区，按工作区实际语言自动探测并运行对应测试：
   - Python   ：pytest（若已安装）→ unittest discover（回退）
   - Node/JS  ：package.json 的 test 脚本
   - Go       ：go test ./...
   - Rust     ：cargo test
   - Java     ：Maven（mvn test）/ Gradle（gradlew test）
   - C#       ：dotnet test
   - Ruby     ：bundle exec rspec
   - PHP      ：phpunit
   - C/C++    ：ctest
   检测到多种语言时，依次运行并各自汇总。
4. 都没检测到已知框架：给出明确提示，不再误导安装 pytest。

全部支持 should_cancel（停止任务时立即杀掉正在跑的测试）与超时。
"""
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

from .base import Tool
from .shell import run_cancellable

DEFAULT_TIMEOUT = 300
RAW_TAIL = 3000


# ---------------------------------------------------------------- #
# Runner：每一种语言/框架一个实例
# ---------------------------------------------------------------- #
class _Runner:
    key: str = ""
    label: str = ""
    file_exts: tuple[str, ...] = ()

    def detect(self, ws: Path) -> bool:
        raise NotImplementedError

    def detect_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.file_exts

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        raise NotImplementedError

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        raise NotImplementedError


def _is_unittest_file(path: Path) -> bool:
    """启发式判断一个 .py 是否使用 unittest 框架。"""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return ("import unittest" in text
            or "from unittest" in text
            or "unittest.main()" in text)


class _PyRunner(_Runner):
    key, label = "python", "Python (pytest / unittest)"
    file_exts = (".py",)

    def detect(self, ws: Path) -> bool:
        if any((ws / m).exists() for m in
               ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")):
            return True
        if any(ws.rglob("test_*.py")) or any(ws.rglob("*_test.py")):
            return True
        return any(ws.rglob("*.py"))     # 兜底：有任意 py 文件即按 Python 处理

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        if target.is_file():
            # 单 Python 文件：unittest 风格走 unittest discover，否则直接执行
            if _is_unittest_file(target):
                c = [sys.executable, "-m", "unittest", "discover",
                     "-s", str(target.parent), "-p", target.name]
            else:
                c = [sys.executable, str(target)]
            if extra:
                c += shlex.split(extra)
            return c

        # 目录 / workspace
        have_pytest = False
        try:
            import importlib.util
            have_pytest = importlib.util.find_spec("pytest") is not None
        except Exception:
            have_pytest = False
        if have_pytest:
            c = [sys.executable, "-m", "pytest", "-q", "--no-header", "-rA",
                 str(target)]
        else:
            # 无 pytest 时回退 unittest discover，不再提示安装 pytest
            c = [sys.executable, "-m", "unittest", "discover",
                 "-s", str(target), "-p", "test_*.py"]
        if extra:
            c += shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        failed: list[str] = []
        stats = ""
        for line in out.splitlines():
            s = line.strip()
            if re.search(r"\d+ (passed|failed|error)", s) and " in " in s:
                stats = s
                break
        failed = [l.strip() for l in out.splitlines()
                  if l.strip().startswith(("FAILED", "ERROR"))]
        # unittest 兜底
        if not stats and ("Ran " in out or "OK" in out or "FAILED" in out):
            m = re.search(r"Ran (\d+) tests?", out)
            n = m.group(1) if m else "?"
            if "FAILED" in out:
                fm = re.search(r"Failures=(\d+), Errors=(\d+)", out)
                stats = (f"{n} run, failures={fm.group(1)}, errors={fm.group(2)}"
                         if fm else f"{n} run, FAILED")
            else:
                stats = f"Ran {n} tests, OK"
            failed += [l.strip().lstrip("FAIL: ").strip()
                       for l in out.splitlines() if l.strip().startswith("FAIL: ")]
        if not stats:
            if failed:
                stats = f"退出码 {rc}（解析到 {len(failed)} 个失败项）"
            elif rc == 0:
                stats = "全部通过 ✓（直接执行，无结构化统计）"
            else:
                stats = "存在失败 ✗（退出码非零，无结构化统计）"
        return stats, failed


def _pkg_manager(ws: Path) -> str:
    if (ws / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (ws / "yarn.lock").exists():
        return "yarn"
    return "npm"


class _NodeRunner(_Runner):
    key, label = "node", "Node/JS (npm/pnpm/yarn test)"
    file_exts = (".js", ".mjs", ".ts", ".jsx", ".tsx")

    def detect(self, ws: Path) -> bool:
        return (ws / "package.json").exists()

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        if target.is_file():
            # Node 18+ 原生测试 runner 可直接跑单文件
            return ["node", "--test", str(target)]
        pm = _pkg_manager(ws)
        c = [pm, "test"]
        if extra:
            c += ["--"] + shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        failed, stats = [], ""
        for line in out.splitlines():
            s = line.strip()
            if s.startswith(("×", "✗", "✘", "●")) or s.startswith("failed "):
                failed.append(s)
            elif re.search(r"^FAIL ", s):
                failed.append(s)
            elif re.search(r"Tests:\s*\d+ failed", s):
                stats = s
            elif re.search(r"Test Files\s+\d+ failed", s):
                stats = s
            elif re.search(r"failing:\s*\d+", s) and "passing" in s:
                stats = s
            elif re.search(r"failed,?\s*\d+ passed", s):
                stats = s
        if not stats:
            stats = "存在失败 ✗" if (rc != 0 or failed) else "通过 ✓"
        return stats, failed[:20]


class _GoRunner(_Runner):
    key, label = "go", "Go (go test)"
    file_exts = (".go",)

    def detect(self, ws: Path) -> bool:
        return (ws / "go.mod").exists()

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        if target.is_file():
            try:
                rel = target.parent.relative_to(ws)
            except ValueError:
                rel = Path(".")
            pkg = "." if rel == Path(".") else f"./{rel.as_posix()}"
            c = ["go", "test", pkg]
        else:
            c = ["go", "test", "./..."]
        if extra:
            c += shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        failed = [l.strip() for l in out.splitlines()
                  if l.strip().startswith("--- FAIL:")]
        n = len(failed)
        stats = (f"{n} 个测试失败 ✗" if n else "全部通过 ✓")
        if not failed and rc != 0:
            stats = "存在失败 ✗（详见原始输出）"
        return stats, failed


class _RustRunner(_Runner):
    key, label = "rust", "Rust (cargo test)"
    file_exts = (".rs",)

    def detect(self, ws: Path) -> bool:
        return (ws / "Cargo.toml").exists()

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        if target.is_file():
            return None          # cargo test 不支持单文件
        c = ["cargo", "test"]
        if extra:
            c += shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        failed = [l.strip() for l in out.splitlines()
                  if re.search(r"\.\.\. FAILED$", l.strip())
                  or l.strip().startswith("test result: FAILED")]
        stats = ""
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("test result:"):
                stats = s
                break
        if not stats:
            stats = "存在失败 ✗" if rc != 0 else "全部通过 ✓"
        return stats, failed


class _MavenRunner(_Runner):
    key, label = "maven", "Java (mvn test)"
    file_exts = (".java",)

    def detect(self, ws: Path) -> bool:
        return (ws / "pom.xml").exists()

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        if target.is_file():
            return None          # Maven 不支持单文件
        c = ["mvn", "-q", "test"]
        if extra:
            c += shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        stats, failed = "", []
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("[INFO] Tests run:") or s.startswith("Tests run:"):
                if "Failures: 0" not in s or "Errors: 0" not in s:
                    failed.append(s)
                if not stats:
                    stats = s
        if not stats:
            stats = "存在失败 ✗" if rc != 0 else "全部通过 ✓"
        return stats, failed


class _GradleRunner(_Runner):
    key, label = "gradle", "Java (gradlew test)"
    file_exts = (".java",)

    def detect(self, ws: Path) -> bool:
        return ((ws / "build.gradle").exists()
                or (ws / "build.gradle.kts").exists())

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        if target.is_file():
            return None          # Gradle 不支持单文件
        gw = "gradlew" if (ws / "gradlew").exists() else "gradle"
        c = [gw, "test"]
        if extra:
            c += shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        failed = [l.strip() for l in out.splitlines()
                  if l.strip().startswith("FAILED ")]
        stats = ""
        for line in out.splitlines():
            s = line.strip()
            if re.search(r"\d+ tests completed, \d+ failed", s):
                stats = s
                break
            if s == "BUILD FAILED":
                stats = "BUILD FAILED"
        if not stats:
            stats = "存在失败 ✗" if rc != 0 else "全部通过 ✓"
        return stats, failed


class _DotnetRunner(_Runner):
    key, label = "dotnet", "C# (dotnet test)"
    file_exts = (".cs",)

    def detect(self, ws: Path) -> bool:
        return any(ws.rglob("*.csproj")) or any(ws.rglob("*.sln"))

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        if target.is_file():
            return None          # dotnet test 以项目为单位
        c = ["dotnet", "test"]
        if extra:
            c += shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        stats, failed = "", []
        for line in out.splitlines():
            s = line.strip()
            if re.search(r"Failed:\s*\d+, Passed:\s*\d+", s):
                stats = s
            elif s.startswith("Failed "):
                failed.append(s)
        if not stats:
            stats = "存在失败 ✗" if rc != 0 else "全部通过 ✓"
        return stats, failed


class _RubyRunner(_Runner):
    key, label = "ruby", "Ruby (rspec)"
    file_exts = (".rb",)

    def detect(self, ws: Path) -> bool:
        return (ws / "Gemfile").exists()

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        if target.is_file():
            c = ["bundle", "exec", "rspec", str(target)]
        else:
            c = ["bundle", "exec", "rspec"]
        if extra:
            c += shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        stats, failed = "", []
        for line in out.splitlines():
            s = line.strip()
            if re.search(r"\d+ examples?, \d+ failure", s):
                stats = s
            elif s.startswith("rspec "):
                failed.append(s)
        if not stats:
            stats = "存在失败 ✗" if rc != 0 else "全部通过 ✓"
        return stats, failed


class _PhpRunner(_Runner):
    key, label = "php", "PHP (phpunit)"
    file_exts = (".php",)

    def detect(self, ws: Path) -> bool:
        return (ws / "composer.json").exists()

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        c = ["phpunit"]
        if str(target) != str(ws):
            c.append(str(target))
        if extra:
            c += shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        stats, failed = "", []
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Tests:") or s.startswith("FAILURES!"):
                stats = s
            elif re.match(r"^\d+\) ", s):
                failed.append(s)
        if not stats:
            stats = "存在失败 ✗" if rc != 0 else "全部通过 ✓"
        return stats, failed


class _CtestRunner(_Runner):
    key, label = "ctest", "C/C++ (ctest)"
    file_exts = (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")

    def detect(self, ws: Path) -> bool:
        return (ws / "CMakeLists.txt").exists()

    def cmd(self, ws: Path, target: Path, extra: str) -> list[str] | None:
        if target.is_file():
            return None          # ctest 不支持单文件
        c = ["ctest", "--output-on-failure"]
        if extra:
            c += shlex.split(extra)
        return c

    def parse(self, out: str, rc: int) -> tuple[str, list[str]]:
        failed = [l.strip() for l in out.splitlines()
                  if "***Failed" in l]
        stats = "100% tests passed ✓" if "100% tests passed" in out else ""
        if not stats:
            stats = "存在失败 ✗" if (rc != 0 or failed) else "全部通过 ✓"
        return stats, failed


RUNNERS: list[_Runner] = [
    _PyRunner(), _NodeRunner(), _GoRunner(), _RustRunner(),
    _MavenRunner(), _GradleRunner(), _DotnetRunner(),
    _RubyRunner(), _PhpRunner(), _CtestRunner(),
]


def detect_runners(ws: str | Path) -> list[_Runner]:
    """返回工作区里检测到的所有适用 Runner（按 RUNNERS 顺序）。"""
    ws = Path(ws)
    return [r for r in RUNNERS if r.detect(ws)]


# ---------------------------------------------------------------- #
# 工具主体
# ---------------------------------------------------------------- #
class RunTestsTool(Tool):
    name = "run_tests"
    description = (
        "运行工作区或单个文件的测试并回传结构化结果：统计 + 失败清单 + 原始输出末尾。\n"
        "自动适配多种语言：\n"
        "① 若目标是整个工作区且存在 tests/run_all.py（本项目标准入口），优先运行它；\n"
        "② 若目标是单个文件，按扩展名选择对应 Runner 并生成针对该文件的命令"
        "（如 .py 直接执行或用 unittest，.go 测试所在包）；\n"
        "③ 若目标是目录/工作区，按工作区实际语言自动探测并运行对应测试："
        "Python(pytest/unittest)、Node(npm/pnpm/yarn test)、Go(go test)、"
        "Rust(cargo test)、Java(Maven/Gradle)、C#(dotnet test)、Ruby(rspec)、"
        "PHP(phpunit)、C/C++(ctest)；检测到多种语言时依次运行并各自汇总；\n"
        "④ 都无则给出明确提示，不再误导安装 pytest。\n"
        "用途：改动代码后验证。path 省略时跑整个工作区。"
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
            ws = Path(self.workspace).resolve()
            target = (ws / (path or ".")).resolve()
            if not target.exists():
                return f"[error] 测试目标不存在: {path}"

            # 1) 项目统一测试入口（仅针对整个工作区）
            run_all = ws / "tests" / "run_all.py"
            if run_all.exists() and str(target) == str(ws):
                return self._run_unified(run_all, extra_args, timeout)

            # 2) 单文件 vs 目录/工作区
            if target.is_file():
                return self._run_single(ws, target, extra_args, timeout)
            return self._run_directory(ws, target, extra_args, timeout)
        except Exception as e:
            return f"[error] 运行测试失败: {type(e).__name__}: {e}"

    def _runner_for_file(self, target: Path) -> _Runner | None:
        for r in RUNNERS:
            if not r.detect_file(target):
                continue
            # 需要 workspace 级配置文件的 runner，单文件也要确认配置存在
            if r.key in ("node", "go", "dotnet", "ruby", "php"):
                if not r.detect(Path(self.workspace)):
                    return None
            # 这些 runner 本身不支持单文件测试
            if r.key in ("rust", "maven", "gradle", "ctest"):
                return None
            return r
        return None

    def _run_single(self, ws: Path, target: Path, extra: str,
                    timeout: int) -> str:
        runner = self._runner_for_file(target)
        if runner is None:
            exts = sorted({e for r in RUNNERS for e in r.file_exts})
            return (f'[error] 不支持的测试文件类型：{target.suffix or "（无扩展名）"}。'
                    f'支持的扩展名：{", ".join(exts)}；'
                    f'或对工作区运行 run_tests 以按项目配置探测。')

        cmd = runner.cmd(ws, target, extra)
        if cmd is None:
            return (f'[error] {runner.label} 不支持针对单个文件运行测试：'
                    f'{target.name}。请指定目录或整个工作区。')

        tag, out, err, rc = run_cancellable(
            cmd, str(ws), timeout, self.should_cancel, shell=False)
        if tag == "cancelled":
            return f"[exit=-1] 已取消\n{err or ''}"
        if tag == "timeout":
            return f"[error] 测试超时（{timeout}s）"
        o = (out or "") + (err or "")
        stats, failed = runner.parse(o, rc)
        lines = [f"=== {runner.label} ===", f"[exit={rc}] {stats}"]
        if failed:
            lines.append("失败:")
            lines += [f"  - {f}" for f in failed[:20]]
            if len(failed) > 20:
                lines.append(f"  ...（共 {len(failed)} 项，已截断）")
        lines.append("--- 原始输出末尾 ---")
        lines.append(o[-RAW_TAIL:].strip() or "（无输出）")
        return "\n".join(lines)

    def _run_directory(self, ws: Path, target: Path, extra: str,
                       timeout: int) -> str:
        matched = detect_runners(ws)
        if not matched:
            return ("[error] 未检测到已知测试框架（Python/Node/Go/Rust/"
                    "Maven/Gradle/C#/Ruby/PHP/C++）。请确认工作区含 "
                    "package.json / go.mod / Cargo.toml / pom.xml / "
                    "build.gradle / *.csproj / Gemfile / composer.json / "
                    "CMakeLists.txt 之一，或为本项目添加 tests/run_all.py。")

        sections, any_fail = [], False
        for r in matched:
            cmd = r.cmd(ws, target, extra)
            if not cmd:
                continue
            tag, out, err, rc = run_cancellable(
                cmd, str(ws), timeout, self.should_cancel, shell=False)
            if tag == "cancelled":
                sections.append(f"=== {r.label} ===\n[exit=-1] 已取消\n"
                                + (err or ""))
                any_fail = True
                continue
            if tag == "timeout":
                sections.append(f"=== {r.label} ===\n[error] 测试超时（{timeout}s）")
                any_fail = True
                continue
            o = (out or "") + (err or "")
            stats, failed = r.parse(o, rc)
            lines = [f"=== {r.label} ===", f"[exit={rc}] {stats}"]
            if failed:
                any_fail = True
                lines.append("失败:")
                lines += [f"  - {f}" for f in failed[:20]]
                if len(failed) > 20:
                    lines.append(f"  ...（共 {len(failed)} 项，已截断）")
            lines.append("--- 原始输出末尾 ---")
            lines.append(o[-RAW_TAIL:].strip() or "（无输出）")
            sections.append("\n".join(lines))

        verdict = "全部通过 ✓" if not any_fail else "存在失败 ✗"
        header = (f"[多语言测试] 检测到 {len(matched)} 类: "
                  f"{', '.join(r.label for r in matched)}\n总判定: {verdict}\n")
        return header + "\n\n".join(sections)

    # ---------------------------------------------------------------- #
    def _run_unified(self, run_all: Path, extra_args: str, timeout: int) -> str:
        cmd = [sys.executable, str(run_all)]
        if extra_args:
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

    # 兼容旧调用方：_parse 即 pytest 解析实现
    @staticmethod
    def _parse_pytest(out: str) -> tuple[str, list[str]]:
        stats = ""
        for line in out.splitlines():
            s = line.strip()
            if re.search(r"\d+ (passed|failed|error)", s) and " in " in s:
                stats = s
                break
        failed = [l.strip() for l in out.splitlines()
                  if l.strip().startswith(("FAILED", "ERROR"))]
        if not stats:
            stats = (f"退出码非零，详见原始输出（解析到 {len(failed)} 个失败项）"
                     if failed else "未能解析统计行，详见原始输出")
        return stats, failed

    _parse = _parse_pytest
