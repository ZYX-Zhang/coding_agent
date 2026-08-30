"""CodingAgent Web 客户端 — 本地 HTTP 服务 + 浏览器界面。

用法：
    python -m CodingAgent.server [workspace]              # 打开 http://127.0.0.1:8765
    python -m CodingAgent.server . --port 9000 -y         # 指定端口 + 自动确认

"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from .agent import Agent
from .cli import build_system_prompt
from .context import Context
from .llm import LLMClient, LLMError
from .safety import AutoSafety, SafetyPolicy
from .session import Session
from .tools import build_registry

WEB_DIR = Path(__file__).parent / "web"


# ---------------------------------------------------------------- #
# 事件总线：Agent 工作线程 publish，SSE 连接线程消费
# ---------------------------------------------------------------- #
class EventBus:
    def __init__(self):
        self.events: list[dict] = []
        self.cond = threading.Condition()

    def publish(self, kind: str, payload: dict) -> None:
        with self.cond:
            self.events.append({"seq": len(self.events),
                                "kind": kind, "payload": payload})
            self.cond.notify_all()

    def wait_batch(self, after: int, timeout: float = 15.0):
        """等待 after 之后的新事件；超时返回空批（SSE 用作 keepalive）。"""
        with self.cond:
            if after >= len(self.events) - 1:
                if not self.cond.wait(timeout):
                    return after, []
            batch = self.events[after + 1:]
            return (batch[-1]["seq"] if batch else after), batch


# ---------------------------------------------------------------- #
# Web 版安全器：确认请求抛给浏览器，阻塞等待决策
# ---------------------------------------------------------------- #
def _brief_args(args: dict) -> str:
    try:
        s = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(args)
    return s if len(s) <= 300 else s[:297] + "…"


class WebSafety:
    """Agent 循环调用 check(tool, args) -> bool（协议与 CLI 的 ConfirmSafety 一致）。

    confirm 级操作 → publish confirm_request 事件 → 阻塞等浏览器 POST /api/decision
    超时（默认 10 分钟）视为拒绝，Agent 循环继续，绝不死锁。
    """

    def __init__(self, bus: EventBus, policy: SafetyPolicy | None = None,
                 timeout: float = 600.0):
        self.policy = policy or SafetyPolicy()
        self.bus = bus
        self.timeout = timeout
        self.auto_allowed = False
        self.denied: list[tuple[str, str]] = []
        self.confirmed: list[str] = []
        self._pending: dict[int, dict] = {}
        self._next_id = 0

    def check(self, tool, args: dict) -> bool:
        name = getattr(tool, "name", str(tool))
        decision, reason = self.policy.classify(name, args or {})

        if decision == "allow":
            return True
        if decision == "deny":
            self.denied.append((name, reason))
            self.bus.publish("denied", {"tool": name, "reason": reason})
            return False

        if self.auto_allowed:                    # 用户在浏览器按过"全部允许"
            self.confirmed.append(f"{name}: {reason} (auto)")
            return True

        cid = self._next_id
        self._next_id += 1
        req = {"id": cid, "tool": name, "reason": reason,
               "args": _brief_args(args), "answer": None,
               "event": threading.Event()}
        self._pending[cid] = req
        self.bus.publish("confirm_request",
                         {"id": cid, "tool": name, "reason": reason,
                          "args": req["args"]})
        ok = req["event"].wait(self.timeout)
        self._pending.pop(cid, None)
        answer = req["answer"] if ok else "n"    # 超时 = 拒绝

        self.bus.publish("confirm_result",
                         {"id": cid, "answer": answer,
                          "approved": answer in ("y", "a")})
        if answer == "a":
            self.auto_allowed = True
        if answer in ("y", "a"):
            self.confirmed.append(f"{name}: {reason}")
            return True
        self.denied.append((name, reason))
        return False

    def decide(self, cid: int, answer: str) -> bool:
        """HTTP 线程调用：应答一个挂起的确认。"""
        req = self._pending.get(cid)
        if req is None:
            return False
        req["answer"] = answer
        req["event"].set()
        return True

    def pending(self) -> list[dict]:
        return [{"id": r["id"], "tool": r["tool"], "reason": r["reason"],
                 "args": r["args"]} for r in self._pending.values()]


class WebAsk:
    """ask_user 工具的 Web 实现（协议与 WebSafety 同款的事件等待模式）。

    ask(question) → publish ask_request 事件 → 阻塞等浏览器 POST /api/answer
    → 返回用户输入的自由文本；超时（10 分钟）返回空串，循环不卡死。
    """

    def __init__(self, bus: EventBus, timeout: float = 600.0):
        self.bus = bus
        self.timeout = timeout
        self._pending: dict[int, dict] = {}
        self._next_id = 0

    def ask(self, question: str) -> str:
        qid = self._next_id
        self._next_id += 1
        req = {"id": qid, "question": question, "answer": None,
               "event": threading.Event()}
        self._pending[qid] = req
        self.bus.publish("ask_request", {"id": qid, "question": question})
        ok = req["event"].wait(self.timeout)
        self._pending.pop(qid, None)
        answer = (req["answer"] if ok else "") or ""
        self.bus.publish("ask_result", {"id": qid, "answer": answer})
        return answer.strip()

    def decide(self, qid: int, answer: str) -> bool:
        """HTTP 线程调用：应答一个挂起的提问。"""
        req = self._pending.get(qid)
        if req is None:
            return False
        req["answer"] = answer
        req["event"].set()
        return True

    def pending(self) -> list[dict]:
        return [{"id": r["id"], "question": r["question"]}
                for r in self._pending.values()]


# ---------------------------------------------------------------- #
# 应用主体（对应 CLI 的 CliApp）
# ---------------------------------------------------------------- #
class WebApp:
    def __init__(self, llm, workspace: str, *, auto: bool = False,
                 max_turns: int = 40, session_dir: str | None = None):
        self.llm = llm
        self.workspace = str(Path(workspace).resolve())
        self.session_dir = Path(session_dir or
                                Path(self.workspace) / ".CodingAgent" / "sessions")
        self.auto = auto
        self.max_turns = max_turns

        self.bus = EventBus()
        self.running = False
        self.cancelled = False              # 协作式取消标志（stop 置位）
        self._run_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None  # 当前后台任务线程
        self.safety: WebSafety | AutoSafety | None = None
        self.session: Session | None = None
        self._saved_len = 0
        self._task_t0: float | None = None       # 当前任务开始时刻
        self._last_elapsed_ms: int = 0           # 上一任务总耗时
        self._last_summary: str = ""             # 上一任务 finish 摘要
        self._new_session()

    # ---------------------------------------------------------------- #
    def _new_session(self) -> None:
        if self.session:
            self.session.close()
        self.bus = EventBus()               # 事件流随会话重置
        self.session_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.session = Session(self.session_dir / f"{ts}.jsonl")
        self.ctx = Context(system_prompt=build_system_prompt(self.workspace))
        self.safety = AutoSafety() if self.auto else WebSafety(self.bus)
        # llm -> 启用 research 子智能体；web_ask -> ask_user 浏览器问答
        self.web_ask = WebAsk(self.bus)
        self.agent = Agent(self.llm,
                           build_registry(self.workspace, llm=self.llm,
                                          ask=self.web_ask.ask,
                                          should_cancel=lambda: self.cancelled),
                           context=self.ctx, safety=self.safety,
                           max_turns=self.max_turns,
                           on_event=lambda k, p: self.bus.publish(k, p),
                           on_delta=lambda t, k="content":
                               self.bus.publish(
                                   "reasoning" if k == "reasoning" else "delta",
                                   {"text": t}),
                           should_cancel=lambda: self.cancelled)
        self._saved_len = 0

    # ---------------------------------------------------------------- #
    def _replay_to_bus(self, msgs: list[dict]) -> None:
        """把历史消息序列转成前端事件回放，使切换会话时 UI 能重建对话。"""
        for m in msgs:
            role = m.get("role")
            if role == "user":
                self.bus.publish("user", {"message": m.get("content", "")})
            elif role == "assistant":
                if m.get("content"):
                    self.bus.publish("assistant", {"content": m["content"]})
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    self.bus.publish("tool_start", {
                        "name": fn.get("name"),
                        "args": fn.get("arguments"),
                        "call_id": tc.get("id"),
                    })
            elif role == "tool":
                self.bus.publish("tool_end", {
                    "name": m.get("name"),
                    "call_id": m.get("tool_call_id"),
                    "result": m.get("content", ""),
                    "duration_ms": 0,
                })

    def _switch_session(self, session_id: str) -> bool:
        """切换到历史会话并恢复上下文，可继续对话（续跑）。"""
        path = self.session_dir / session_id
        if not path.exists() and not str(path).endswith(".jsonl"):
            path = self.session_dir / (session_id + ".jsonl")
        if not path.exists():
            return False
        self._flush_session()
        if self.session:
            self.session.close()
        self.bus = EventBus()           # 事件流随会话重置
        msgs = Session.load(path)
        self.ctx = Context(system_prompt=build_system_prompt(self.workspace))
        for m in msgs:
            self.ctx.add_message(m)
        self.safety = AutoSafety() if self.auto else WebSafety(self.bus)
        self.web_ask = WebAsk(self.bus)
        self.agent = Agent(self.llm,
                           build_registry(self.workspace, llm=self.llm,
                                          ask=self.web_ask.ask,
                                          should_cancel=lambda: self.cancelled),
                           context=self.ctx, safety=self.safety,
                           max_turns=self.max_turns,
                           on_event=lambda k, p: self.bus.publish(k, p),
                           on_delta=lambda t, k="content":
                               self.bus.publish(
                                   "reasoning" if k == "reasoning" else "delta",
                                   {"text": t}),
                           should_cancel=lambda: self.cancelled)
        self.session = Session(path)    # 继续追加到同一文件（续跑落盘）
        self._saved_len = len(self.ctx.messages)
        self._replay_to_bus(msgs)
        self.bus.publish("session",
                         {"message": f"已切换到会话 {path.stem}（可继续对话）"})
        return True

    def list_sessions(self) -> list[dict]:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for p in sorted(self.session_dir.glob("*.jsonl"),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                msgs = Session.load(p)
            except Exception:
                msgs = []
            title = p.stem
            for m in msgs:
                if m.get("role") == "user":
                    title = (m.get("content") or "").strip().split("\n")[0][:40]
                    break
            items.append({
                "id": p.name,
                "title": title,
                "mtime": int(p.stat().st_mtime * 1000),
                "messages": len(msgs),
                "current": (self.session is not None
                            and str(p.resolve()) == str(self.session.path.resolve())),
            })
        return items

    def view_session(self, session_id: str) -> dict | None:
        """只读查看某个历史会话的内容（不切换、不修改任何运行状态）。

        即使当前有任务在跑也安全：它不触碰 self.ctx / self.agent / self.bus，
        仅读取目标会话文件并返回消息列表，供前端以「只读快照」形式渲染，
        从而支持「一个会话执行时查看其他历史会话」。
        """
        path = self.session_dir / session_id
        if not path.exists() and not str(path).endswith(".jsonl"):
            path = self.session_dir / (session_id + ".jsonl")
        if not path.exists():
            return None
        try:
            msgs = Session.load(path)
        except Exception:
            msgs = []
        title = path.stem
        for m in msgs:
            if m.get("role") == "user":
                title = (m.get("content") or "").strip().split("\n")[0][:40]
                break
        return {"id": path.name, "title": title, "messages": msgs}

    def delete_session(self, session_id: str) -> dict:
        path = self.session_dir / session_id
        if not path.exists() and not str(path).endswith(".jsonl"):
            path = self.session_dir / (session_id + ".jsonl")
        if not path.exists():
            return {"ok": False, "error": "会话不存在"}
        is_current = (self.session is not None
                      and str(path.resolve()) == str(self.session.path.resolve()))
        try:
            path.unlink()
        except OSError as e:
            return {"ok": False, "error": str(e)}
        if is_current:
            self._flush_session()
            self._new_session()
            self.bus.publish("session",
                             {"message": "已删除当前会话，已新建空白会话"})
        return {"ok": True, "current_deleted": is_current}

    def set_auto(self, on: bool) -> None:
        """运行时切换自动确认（下一轮生效，pending 不受影响）。"""
        self.auto = on
        self.agent.safety = AutoSafety() if on else WebSafety(self.bus)

    # ---------------------------------------------------------------- #
    def start_task(self, message: str) -> tuple[bool, str]:
        """提交任务（单任务模型，参照 WorkBuddy 的停止语义）。

        只有一个后台 worker 在跑。只要上一任 worker 线程还活着——无论是正常执行，
        还是刚被停止、正在退场——都拒绝新提交（返回 409）。前端据此退避重试；
        新任务真正开始的前提是上一任 agent.run 已彻底退出，因此绝不会有两个
        agent.run 并发写同一份 ctx，从根上杜绝会话历史错乱，也无需任何"在后台
        join 旧任务"的阻塞（那正是之前卡顿/不显示任务的根因）。
        """
        import time as _time
        old = self._worker_thread
        if old is not None and old.is_alive():
            return False, "上一个任务仍在停止中，请稍候"
        with self._run_lock:
            self.running = True
            self.cancelled = False          # 新任务从干净状态开始
        # 立刻广播 user 事件：前端马上显示任务气泡（stop 已提前释放输入）
        self.bus.publish("user", {"message": message})
        self._task_t0 = _time.monotonic()
        t = threading.Thread(target=self._worker, args=(message,),
                             daemon=True)
        self._worker_thread = t
        t.start()
        return True, "已开始"

    def cancel_task(self) -> bool:
        """停止当前任务（协作式取消）。

        仅置位 cancelled 标志，由 Agent 主循环在「下一轮模型调用之前」与
        「每一个工具执行之前」检查而主动终止——当前正在进行的模型生成 / 工具
        会先跑完，再于下一个步骤边界干净收尾。取消前的文件改动与会话历史全部
        保留，取消后用户可立即输入新任务从该会话续跑。

        注意：这里不把 running 置 False——改由 worker 的 finally 收尾。这样在退场
        期间 start_task 的「worker 存活」守卫持续返回 409，新任务会等旧 worker 真正
        退出后再开始（无卡顿、无并发写），停止后用户可立即输入并续跑。
        """
        if not self.running:
            return False
        self.cancelled = True
        return True

    def _worker(self, message: str) -> None:
        import time as _time
        summary = ""
        try:
            summary = self.agent.run(message) or ""
            self._last_summary = summary
            if summary.startswith(("已达到最大轮数", "模型连续")):
                self.bus.publish("warning", {"message": summary})
        except LLMError as e:
            self.bus.publish("error", {"message": f"模型调用失败: {e}"})
        except Exception as e:                       # 兜底
            self.bus.publish("error",
                             {"message": f"{type(e).__name__}: {e}"})
        finally:
            # 落盘保留改动（即便被取消也保留，便于续跑）
            self._flush_session()
            elapsed = int((_time.monotonic() - (self._task_t0 or _time.monotonic()))
                          * 1000)
            self._last_elapsed_ms = elapsed
            # 收尾：worker 线程此刻已彻底退出（is_alive() 随之变 False），
            # 故 start_task 的「worker 存活」守卫在真正结束后才放行新任务，
            # 绝不会与下一个 agent.run 并发写同一份 ctx。
            with self._run_lock:
                self.running = False
            self.bus.publish("task_end", {"elapsed_ms": elapsed,
                                          "summary": summary})
            self._task_t0 = None

    def _flush_session(self) -> None:
        if not self.session:
            return
        fresh = self.ctx.messages[self._saved_len:]
        if fresh:
            self.session.extend(fresh)
            self._saved_len = len(self.ctx.messages)

    # ---------------------------------------------------------------- #
    def state(self) -> dict:
        tool_calls = sum(1 for e in self.bus.events
                         if e["kind"] == "tool_end")
        turns = sum(1 for e in self.bus.events
                    if e["kind"] == "turn_start")
        import time as _time
        elapsed = (self._last_elapsed_ms if not self.running and self._last_elapsed_ms
                   else int((_time.monotonic() - self._task_t0) * 1000)
                   if self.running and self._task_t0 else 0)
        safety = self.safety
        return {
            "running": self.running,
            "workspace": self.workspace,
            "model": getattr(self.llm, "model", "mock"),
            "auto": self.auto,
            "events": self.bus.events[-1000:],
            "pending": (safety.pending()
                        if isinstance(safety, WebSafety) else []),
            "stats": {
                "messages": len(self.ctx.messages),
                "tool_calls": tool_calls,
                "turns": turns,
                "elapsed_ms": elapsed,
                "last_summary": self._last_summary,
                "tokens_estimated": self.ctx.estimate_total(),
                "token_budget": self.ctx.max_tokens,
                "session_file": str(self.session.path) if self.session else "",
                "denied": len(getattr(safety, "denied", [])),
            },
        }


# ---------------------------------------------------------------- #
# HTTP 服务
# ---------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    app: WebApp            # 类属性，serve() 时注入

    def log_message(self, *args):                   # 静默访问日志
        pass

    # ---- 工具方法 ---- #
    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # ---- GET ---- #
    def do_GET(self):
        url = urlsplit(self.path)
        qs = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            self._serve_index()
        elif url.path == "/api/state":
            self._json(self.app.state())
        elif url.path == "/api/sessions":
            self._json(self.app.list_sessions())
        elif url.path == "/api/session/view":
            sid = qs.get("id", [""])[0]
            data = self.app.view_session(sid)
            if data is None:
                self._json({"error": "会话不存在"}, 404)
            else:
                self._json(data)
        elif url.path == "/api/events":
            self._serve_sse(url)
        else:
            self._json({"error": "not found"}, 404)

    def _serve_index(self) -> None:
        index = WEB_DIR / "index.html"
        if not index.exists():
            self._json({"error": "index.html missing"}, 500)
            return
        body = index.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self, url) -> None:
        qs = parse_qs(url.query)
        after = int(qs.get("after", ["-1"])[0])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                after, batch = self.app.bus.wait_batch(after)
                if not batch:                       # keepalive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                for e in batch:
                    data = json.dumps(e, ensure_ascii=False)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                    # 浏览器关掉了页面

    # ---- POST ---- #
    def do_POST(self):
        url = urlsplit(self.path)
        body = self._body()
        if url.path == "/api/task":
            message = str(body.get("message", "")).strip()
            if not message:
                self._json({"error": "message 为空"}, 400)
                return
            ok, msg = self.app.start_task(message)
            self._json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif url.path == "/api/decision":
            safety = self.app.safety
            if not isinstance(safety, WebSafety):
                self._json({"ok": False, "error": "当前为自动确认模式"}, 400)
                return
            ok = safety.decide(int(body.get("id", -1)),
                               str(body.get("answer", "n")).lower())
            self._json({"ok": ok}, 200 if ok else 404)
        elif url.path == "/api/answer":
            ok = self.app.web_ask.decide(int(body.get("id", -1)),
                                         str(body.get("answer", "")))
            self._json({"ok": ok}, 200 if ok else 404)
        elif url.path == "/api/session/switch":
            if self.app.running:
                self._json({"ok": False, "error": "任务运行中，不能切换"}, 409)
                return
            sid = str(body.get("id", ""))
            ok = self.app._switch_session(sid)
            self._json({"ok": ok}, 200 if ok else 404)
        elif url.path == "/api/session/delete":
            # 放行：运行中也能删除「其他」历史会话（只读文件，不影响正在跑的任务）；
            # 只有删除「当前正在运行」的会话会被 delete_session 内部拒绝。
            sid = str(body.get("id", ""))
            self._json(self.app.delete_session(sid))
        elif url.path == "/api/new":
            if self.app.running:
                self._json({"ok": False, "error": "任务运行中，不能新建"}, 409)
                return
            self.app._flush_session()
            self.app._new_session()
            self.app.bus.publish("session", {"message": "已开始新会话"})
            self._json({"ok": True, "session_file":
                        str(self.app.session.path)})
        elif url.path == "/api/auto":
            self.app.set_auto(bool(body.get("on")))
            self.app.bus.publish("session", {
                "message": "已切换为自动确认（写操作不再询问）"
                if self.app.auto else "已切换为逐项确认"})
            self._json({"ok": True, "auto": self.app.auto})
        elif url.path == "/api/stop":
            ok = self.app.cancel_task()
            self._json({"ok": ok,
                        "message": "已请求停止，Agent 将在下一步骤前终止"
                        if ok else "当前没有运行中的任务"},
                       200 if ok else 409)
        else:
            self._json({"error": "not found"}, 404)


def serve(app: WebApp, port: int = 8765, host: str = "127.0.0.1") -> None:
    Handler.app = app
    httpd = ThreadingHTTPServer((host, port), Handler)
    actual = httpd.server_address[1]
    print(f"CodingAgent Web 客户端已启动: http://{host}:{actual}")
    print(f"工作区: {app.workspace} | 模型: {getattr(app.llm, 'model', '?')}")
    print("Ctrl+C 停止服务。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        app._flush_session()
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="CodingAgent-server",
        description="编程智能体 Web 客户端")
    ap.add_argument("workspace", nargs="?", default="./workspace")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--session-dir", default=None)
    ap.add_argument("-y", "--yes", action="store_true",
                    help="自动确认所有写操作（高危命令仍拦截）")
    ap.add_argument("--max-turns", type=int, default=40)
    args = ap.parse_args(argv)

    workspace = str(Path(args.workspace).resolve())
    if not Path(workspace).is_dir():
        print(f"错误: 工作区不存在 {workspace}", file=sys.stderr)
        return 2
    try:
        llm = LLMClient()
    except LLMError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2

    app = WebApp(llm, workspace, auto=args.yes,
                 max_turns=args.max_turns, session_dir=args.session_dir)
    serve(app, port=args.port, host=args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
