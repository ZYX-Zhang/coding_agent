"""工作区快照与撤销恢复（纯标准库实现）。

用于实现「任务级代码快照 / 一键撤销」：
- take_snapshot(workspace, dest)：在任务开始前把 workspace 全量复制到 dest，
  忽略 .git / node_modules / .agent_snapshots 等大目录与 *.pyc 等噪音文件。
- restore_snapshot(workspace, src)：把 workspace 恢复到 src 所记录的「任务开始前」
  状态——
    * 快照里有的文件 → 覆盖写回（恢复被改动的）；
    * workspace 里有但快照里没有的文件 → 删除（任务新建的）；
    * workspace 里被删、快照里有的文件 → 恢复（任务删除的）。
  恢复全程只触及 workspace 内部，且跳过 .agent_snapshots / .git / .agent 等
  受保护目录，不会误删用户的版本库或快照自身。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# 快照与恢复时一律跳过的目录 / 文件（避免把依赖、版本库、快照自身卷进来）
IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".agent", ".agent_snapshots", ".idea", ".vscode", ".mypy_cache",
    ".pytest_cache", "dist", "build",
}
IGNORE_FILES_SUFFIX = (".pyc",)


def _dir_ignored(name: str) -> bool:
    return name in IGNORE_DIRS


def _file_ignored(name: str) -> bool:
    return name.endswith(IGNORE_FILES_SUFFIX)


def take_snapshot(workspace: str, dest: str) -> int:
    """把 workspace 全量快照到 dest（覆盖式）。返回快照文件数。"""
    ws = Path(workspace)
    d = Path(dest)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

    count = 0
    for root, dirs, files in os.walk(ws):
        # 原地裁剪，避免进入被忽略目录（也覆盖 symlink 目录的误入）
        dirs[:] = [dn for dn in dirs
                   if not _dir_ignored(dn) and not os.path.islink(os.path.join(root, dn))]
        rel = Path(root).relative_to(ws)
        for fn in files:
            if _file_ignored(fn):
                continue
            src = Path(root) / fn
            if src.is_symlink() or src.is_dir():
                continue
            tgt = d / rel / fn
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tgt)
            count += 1
    return count


def restore_snapshot(workspace: str, src: str) -> dict:
    """把 workspace 恢复到 src 快照所记录的「任务开始前」状态。

    返回 {"restored": int, "removed": int, "restored_dirs": int}。
    """
    ws = Path(workspace)
    s = Path(src)
    if not s.exists():
        raise FileNotFoundError(f"快照不存在: {src}")

    # 1) 把快照文件覆盖写回 workspace（恢复改动 / 恢复被删文件）
    restored = 0
    for root, dirs, files in os.walk(s):
        dirs[:] = [dn for dn in dirs if not _dir_ignored(dn)]
        rel = Path(root).relative_to(s)
        for fn in files:
            sp = Path(root) / fn
            if sp.is_symlink() or sp.is_dir():
                continue
            tgt = ws / rel / fn
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, tgt)
            restored += 1

    # 2) 删除 workspace 中存在但快照里没有的文件（任务新建的），并清理空目录
    removed = 0
    for root, dirs, files in os.walk(ws):
        # 受保护目录不进入、不删除
        dirs[:] = [dn for dn in dirs
                   if not _dir_ignored(dn)
                   and dn != ".agent_snapshots"
                   and not os.path.islink(os.path.join(root, dn))]
        rel = Path(root).relative_to(ws)
        for fn in files:
            if _file_ignored(fn) or _dir_ignored(fn):
                continue
            if not (s / rel / fn).exists():
                try:
                    (Path(root) / fn).unlink()
                    removed += 1
                except OSError:
                    pass

    # 3) 清理被丢弃后残留的空目录（从深到浅）
    restored_dirs = 0
    for root, dirs, files in os.walk(ws, topdown=False):
        if _dir_ignored(Path(root).name) or Path(root).name == ".agent_snapshots":
            continue
        try:
            if not os.listdir(root):
                os.rmdir(root)
                restored_dirs += 1
        except OSError:
            pass

    return {"restored": restored, "removed": removed, "restored_dirs": restored_dirs}
