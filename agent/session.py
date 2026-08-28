"""会话持久化

"""
from __future__ import annotations

import json
from pathlib import Path


class Session:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 追加模式打开并保持句柄，避免每条消息开关文件
        self._fh = open(self.path, "a", encoding="utf-8")

    # ---------------------------------------------------------------- #
    def append(self, msg: dict) -> None:
        """追加一条消息（user/assistant/tool/system 均可）。"""
        self._fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._fh.flush()

    def extend(self, messages: list[dict]) -> None:
        for m in messages:
            self.append(m)

    # ---------------------------------------------------------------- #
    @staticmethod
    def load(path: str | Path) -> list[dict]:
        """读取整个会话为消息列表（跳过损坏行而不是整体失败）。"""
        msgs, bad = [], 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
        if bad:
            msgs.append({"role": "system",
                         "content": f"[警告] 会话文件有 {bad} 行损坏已跳过"})
        return msgs

    def close(self) -> None:
        self._fh.close()


def resume_context(path: str | Path, ctx) -> Context:
    """从 JSONL 恢复 Context：把历史消息原样灌回。"""
    for m in Session.load(path):
        ctx.add_message(m)
    return ctx
