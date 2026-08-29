"""网络抓取工具：fetch_url。
"""
from __future__ import annotations

import urllib.parse
import urllib.request

from .base import Tool

_MAX_BYTES = 1_000_000  # 最多下载 1MB
_UA = "agent/0.1 (+local coding agent)"


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "抓取一个网页 / 接口的文本正文（只读，不改工作区）。\n"
        "coding agent 查 API 文档、报错原因、RFC 时常需它。\n"
        "仅支持 http/https，禁止 file://；返回解码后的前 max_chars 个字符"
        "（默认 8000，HTML 标签未剥离，便于模型按需提取）。\n"
        "返回 [ok] 正文 或 [error] 原因。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string",
                    "description": "要抓取的 http/https 地址"},
            "max_chars": {"type": "integer",
                          "description": "最多返回的字符数，默认 8000"},
            "timeout": {"type": "integer",
                        "description": "读取超时秒数，默认 15"},
        },
        "required": ["url"],
    }

    def execute(self, url: str, max_chars: int = 8000,
                timeout: int = 15, **_) -> str:
        if not url:
            return "[error] url 不能为空"
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"[error] 仅支持 http/https，拒绝 scheme: {parsed.scheme}"
        max_chars = max(0, min(int(max_chars), 200_000))
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(_MAX_BYTES)
                ctype = resp.headers.get("Content-Type", "")
            text = raw.decode("utf-8", errors="ignore")
        except Exception as e:
            return f"[error] 抓取失败: {type(e).__name__}: {e}"

        # 截断到 max_chars（优先保留开头）
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n…[已截断，原文 {len(text)} 字符]"
        head = f"[ok] 已抓取 {url}（Content-Type: {ctype or '未知'}）：\n"
        return head + text
