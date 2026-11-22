"""FastAPI HTTP 服务：把 Agent 暴露为 Web UI、同步接口与 SSE 流式接口。

启动::

    python -m agent serve --open          # 自动打开 Web UI
    python -m agent serve --port 8077
    # 或
    uvicorn agent.server:app --host 127.0.0.1 --port 8077

接口：
- GET  /                    Web UI（web/index.html）
- GET  /static/*            前端静态资源
- GET  /healthz             组件健康检查
- GET  /v1/tools            工具目录
- GET  /v1/config           生效配置（已脱敏）
- GET  /v1/traces           最近运行记录
- POST /v1/agent/run        同步执行（支持 history）
- POST /v1/agent/stream     SSE 事件流（event=类型，data=payload）
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# 本地 OCR/上传依赖；放在工作区，避免污染全局环境
_here = Path(__file__).resolve().parents
_candidates = [_here[1] / ".agent" / "ocrlibs"] if len(_here) > 1 else []
if len(_here) > 4:
    _candidates.append(_here[4] / ".agent" / "ocrlibs")
for _candidate in _candidates:
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))
        break

try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - 可选依赖
    raise RuntimeError(
        "HTTP 服务需要 fastapi/uvicorn：pip install fastapi uvicorn"
    ) from exc

from .agent import Agent
from .config import load_config, load_env_file
from .llm import create_llm
from .policy import AgentPolicy
from .trace import list_traces
from .types import Message
from .vision import QwenVLClient

logger = logging.getLogger(__name__)


WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class RunRequest(BaseModel):
    """同步运行请求。"""

    task: str = Field(..., min_length=1, description="用户任务/问题")
    session_id: Optional[str] = Field(default=None, description="会话 ID，用于多轮记忆")
    mode: Optional[Literal["fast", "normal", "auto"]] = Field(default=None, description="回答模式")
    fast: bool = Field(default=False, description="兼容旧客户端：true 等价于 fast")
    attachment_text: Optional[str] = Field(
        default=None,
        max_length=4000,
        description="图片 OCR 文字；由 /v1/attachments/ocr 生成",
    )
    history: Optional[List[HistoryMessage]] = Field(
        default=None, description="可选的多轮历史（前端持久化时传入）"
    )


class StreamRequest(BaseModel):
    """流式运行请求。"""

    task: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    mode: Optional[Literal["fast", "normal", "auto"]] = None
    history: Optional[List[HistoryMessage]] = None


def _resolve_mode(request: Any) -> str:
    mode = getattr(request, "mode", None)
    if mode in ("fast", "normal"):
        return mode
    if mode == "auto":
        task = str(getattr(request, "task", ""))
        high_risk = (
            "电池", "充电", "固件", "返航", "坠机", "摔", "进水",
            "安全", "限高", "避障", "无法开机", "开不了机",
        )
        return "normal" if any(term in task for term in high_risk) else "fast"
    return "fast" if getattr(request, "fast", False) else "normal"


def _to_history(request: Any) -> Optional[List[Message]]:
    history = getattr(request, "history", None)
    if not history:
        return None
    return [Message(role=item.role, content=item.content) for item in history]


def _task_with_attachment(request: Any) -> str:
    task = request.task.strip()
    text = (getattr(request, "attachment_text", None) or "").strip()
    if not text:
        return task
    return (
        task
        + "\n\n[用户上传图片的 OCR 文字]\n"
        + text[:3000]
        + "\n\n[附件说明] 你只能依据上述图片文字和视觉观察判断，不能声称看过图片本身；"
        "视觉观察标注为“非手册”，不能替代手册事实；"
        "如果文字不足以判断外观损伤，请明确说明并询问用户。"
    )


def _safe_config(agent: Agent) -> Dict[str, Any]:
    data = agent.config.to_dict()
    if isinstance(data.get("llm"), dict):
        data["llm"]["api_key"] = "***" if data["llm"].get("api_key") else None
    data["runtime"] = {
        "provider": agent.llm.name,
        "tools": agent.runtime.enabled_tools,
        "kb": agent.kb.description if agent.kb is not None else None,
        "config_path": agent.config.config_path,
    }
    return data


def create_app(
    config_path: Optional[str] = None,
    *,
    agent: Optional[Agent] = None,
) -> FastAPI:
    """创建 FastAPI 应用。Agent 延迟到首次请求时才初始化。"""
    app = FastAPI(
        title="DJI Manual Agent",
        version="0.1.0",
        description="大疆手册多步检索 Agent：工具调用、记忆、引用与追踪",
    )
    holder: Dict[str, Optional[Agent]] = {"agent": agent, "fast": None}
    lock = threading.Lock()
    jobs: Dict[str, Dict[str, Any]] = {}
    jobs_lock = threading.Lock()
    rate_hits: Dict[str, list] = defaultdict(list)
    rate_lock = threading.Lock()
    ocr_holder: Dict[str, Any] = {"engine": None}
    ocr_lock = threading.Lock()
    load_env_file()
    vision_client = QwenVLClient()

    def _rate_limit(http_request: Request) -> None:
        try:
            limit = int(os.environ.get("AGENT_RATE_LIMIT_PER_MINUTE", "30"))
        except ValueError:
            limit = 30
        if limit <= 0:
            return
        ip = http_request.client.host if http_request.client else "unknown"
        key = _user_of(http_request) or ip
        now = time.time()
        with rate_lock:
            hits = [stamp for stamp in rate_hits[key] if now - stamp < 60]
            if len(hits) >= limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"请求过于频繁，请稍后再试（{limit}/分钟）",
                    headers={"Retry-After": "30"},
                )
            hits.append(now)
            rate_hits[key] = hits

    def _auth_users() -> Dict[str, str]:
        raw = os.environ.get("AGENT_USERS", "").strip()
        users: Dict[str, str] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    for name, password in parsed.items():
                        name = str(name).strip()
                        if name and ":" not in name and str(password):
                            users[name] = str(password)
            except Exception:
                logger.warning("AGENT_USERS 不是合法 JSON，忽略")
        token = os.environ.get("AGENT_ACCESS_TOKEN", "").strip()
        if token and "dji" not in users:
            users["dji"] = token
        return users

    def _user_of(request: Request) -> str:
        return str(getattr(request.state, "user_id", "") or "")

    def _scoped_session(user: str, session_id: Optional[str]) -> Optional[str]:
        if not user or not session_id:
            return session_id
        return f"u:{user}:{session_id}"

    @app.middleware("http")
    async def _basic_auth(request: Request, call_next):
        # 健康检查不暴露业务数据，允许无认证探活
        if request.url.path == "/healthz":
            return await call_next(request)
        users = _auth_users()
        if not users:
            return await call_next(request)
        authorization = request.headers.get("authorization", "")
        user_id = ""
        password_used = ""
        if authorization.startswith("Basic "):
            try:
                decoded = base64.b64decode(authorization[6:]).decode("utf-8")
                username, password = decoded.split(":", 1)
                expected = users.get(username, "")
                if expected and hmac.compare_digest(password, expected):
                    user_id, password_used = username, password
            except Exception:
                user_id = ""
        elif authorization.startswith("Bearer "):
            supplied = authorization[7:].strip()
            for name, password in users.items():
                if hmac.compare_digest(supplied, password):
                    user_id, password_used = name, password
                    break
        if not user_id:
            cookie = request.cookies.get("agent_access", "")
            if ":" in cookie:
                name, password = cookie.split(":", 1)
                expected = users.get(name, "")
                if expected and hmac.compare_digest(password, expected):
                    user_id, password_used = name, password
            elif cookie:
                for name, password in users.items():
                    if hmac.compare_digest(cookie, password):
                        user_id, password_used = name, password
                        break
        # 不再接受 URL query 里的 access_token，避免 token 进入日志/历史记录
        if not user_id:
            return Response(
                status_code=401,
                content="Authentication required",
                headers={"WWW-Authenticate": 'Basic realm="DJI Manual Agent"'},
            )
        request.state.user_id = user_id
        response = await call_next(request)
        if password_used:
            response.set_cookie(
                "agent_access",
                f"{user_id}:{password_used}",
                max_age=7 * 24 * 3600,
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="lax",
            )
            # 同源 fetch 使用的 UI token；页面本身必须已通过认证
            response.set_cookie(
                "agent_ui",
                password_used,
                max_age=7 * 24 * 3600,
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="lax",
            )
            response.set_cookie(
                "agent_user",
                user_id,
                max_age=7 * 24 * 3600,
                httponly=False,
                secure=request.url.scheme == "https",
                samesite="lax",
            )
        return response

    def get_agent(mode: str = "normal") -> Agent:
        with lock:
            key = "fast" if mode == "fast" else "agent"
            if holder[key] is None:
                config = load_config(config_path)
                if mode == "fast":
                    # 快速模式：少检索、少校验、关闭长思考；适合公网交互，质量略降
                    config.tools.enabled = [name for name in config.tools.enabled if name in ("manual_search", "manual_fetch", "calculator")]
                    config.knowledge.top_k = min(3, config.knowledge.top_k)
                    config.agent.max_steps = min(3, config.agent.max_steps)
                    config.agent.max_llm_calls = min(5, config.agent.max_llm_calls)
                    config.agent.claim_verifier = False
                    config.agent.fact_coverage = False
                    config.agent.max_fact_retries = 0
                    config.agent.prefetch_search = True
                    config.memory.history_turns = min(3, config.memory.history_turns)
                    config.llm.thinking = False
                    config.llm.reasoning_effort = "low"
                else:
                    # 标准模式：保留多步检索和高风险校验，但压缩上下文与预检索
                    config.tools.enabled = [name for name in config.tools.enabled if name in ("manual_search", "manual_fetch", "calculator")]
                    config.knowledge.top_k = min(6, config.knowledge.top_k)
                    config.knowledge.max_passage_chars = min(900, config.knowledge.max_passage_chars)
                    config.agent.max_observation_chars = min(5000, config.agent.max_observation_chars)
                    config.agent.prefetch_search = True
                    config.llm.reasoning_effort = "medium"
                holder[key] = Agent(config, policy=AgentPolicy.from_config(config))
            return holder[key]  # type: ignore[return-value]

    @app.on_event("shutdown")
    def _shutdown() -> None:  # pragma: no cover - 生命周期
        with lock:
            for key in ("agent", "fast"):
                if holder.get(key) is not None:
                    holder[key].close()
                    holder[key] = None

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        try:
            return get_agent().health()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/tools")
    def tools() -> Any:
        runtime_agent = get_agent()
        return runtime_agent.tools.describe(runtime_agent.runtime.enabled_tools)

    @app.get("/v1/config")
    def config() -> Any:
        return _safe_config(get_agent())

    @app.get("/v1/traces")
    def traces(limit: int = 20, http_request: Request = None) -> Any:
        user = _user_of(http_request) if http_request is not None else ""
        if user and user != "dji":
            raise HTTPException(status_code=403, detail="admin only")
        return list_traces(get_agent().config.trace.dir, limit=min(max(1, limit), 100))

    def _ocr_bytes(data: bytes, filename: str = "") -> Dict[str, Any]:
        from PIL import Image
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR

        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((2200, 2200))
        with ocr_lock:
            if ocr_holder["engine"] is None:
                ocr_holder["engine"] = RapidOCR()
            engine = ocr_holder["engine"]
        result, _ = engine(np.array(image))
        lines = [str(item[1]).strip() for item in (result or []) if item and item[1]]
        return {
            "filename": filename,
            "text": "\n".join(lines).strip(),
            "lines": lines[:80],
            "engine": "rapidocr",
        }

    @app.post("/v1/attachments/ocr")
    async def ocr_attachment(file: UploadFile = File(...), http_request: Request = None) -> Any:
        if http_request is not None:
            _rate_limit(http_request)
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        data = await file.read()
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="图片不能超过 8MB")
        try:
            return _ocr_bytes(data, file.filename or "")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"OCR 失败：{type(exc).__name__}") from exc

    @app.post("/v1/attachments/analyze")
    async def analyze_attachment(file: UploadFile = File(...), question: str = Form(""), http_request: Request = None) -> Any:
        if http_request is not None:
            _rate_limit(http_request)
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        data = await file.read()
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="图片不能超过 8MB")
        from starlette.concurrency import run_in_threadpool

        try:
            ocr = await run_in_threadpool(_ocr_bytes, data, file.filename or "")
        except Exception as exc:
            ocr = {"filename": file.filename or "", "text": "", "lines": [], "engine": "rapidocr", "error": type(exc).__name__}
        vision = await run_in_threadpool(
            vision_client.analyze,
            data,
            mime=file.content_type or "image/png",
            question=question,
        )
        return {
            "ocr": ocr,
            "vision": vision.to_dict(),
            "vision_error": vision.error,
            "vision_available": vision_client.available,
        }

    @app.get("/v1/sessions")
    def sessions(limit: int = 20, http_request: Request = None) -> Any:
        runtime_agent = get_agent()
        memory = getattr(runtime_agent, "memory", None)
        if memory is None or not getattr(memory, "available", False):
            return {"sessions": []}
        user = _user_of(http_request) if http_request is not None else ""
        rows = memory.list_sessions(limit=min(max(1, limit), 100), prefix="")
        if user:
            own = f"u:{user}:"
            rows = [
                row for row in rows
                if str(row.get("session_id", "")).startswith(own)
                or (user == "dji" and not str(row.get("session_id", "")).startswith("u:"))
            ]
            for row in rows:
                sid = str(row.get("session_id", ""))
                if sid.startswith(own):
                    row["session_id"] = sid[len(own):]
        return {"sessions": rows[: min(max(1, limit), 100)]}

    @app.get("/v1/sessions/{session_id}")
    def session_detail(session_id: str, limit: int = 50, http_request: Request = None) -> Any:
        runtime_agent = get_agent()
        memory = getattr(runtime_agent, "memory", None)
        if memory is None or not getattr(memory, "available", False):
            return {"session_id": session_id, "turns": [], "summary": ""}
        user = _user_of(http_request) if http_request is not None else ""
        scoped = _scoped_session(user, session_id) or session_id
        turns = memory.get_turns(scoped, limit=min(max(1, limit), 100))
        summary = memory.get_summary(scoped)
        # 兼容旧单用户数据：dji 管理员可读未加命名空间的历史
        if user == "dji" and not turns:
            turns = memory.get_turns(session_id, limit=min(max(1, limit), 100))
            summary = memory.get_summary(session_id)
        return {
            "session_id": session_id,
            "turns": turns,
            "summary": summary,
        }

    @app.post("/v1/agent/jobs")
    def create_job(request: RunRequest, http_request: Request) -> Any:
        """异步任务：公网隧道下避免长连接超时，前端轮询结果。"""
        _rate_limit(http_request)
        job_id = uuid.uuid4().hex
        user = _user_of(http_request)
        scoped_session = _scoped_session(user, request.session_id)
        with jobs_lock:
            if len(jobs) > 50:
                for old_id in list(jobs)[:10]:
                    jobs.pop(old_id, None)
            jobs[job_id] = {"status": "running", "created": job_id, "owner": user, "last_event": {"type": "queued"}}

        def on_event(event: Any) -> None:
            with jobs_lock:
                job = jobs.get(job_id)
                if job is not None:
                    job["last_event"] = {"type": event.type, "step": event.step}

        def worker() -> None:
            try:
                result = get_agent(_resolve_mode(request)).ask(
                    _task_with_attachment(request),
                    session_id=scoped_session,
                    history=_to_history(request),
                    on_event=on_event,
                )
                payload = result.to_dict(include_steps=True)
                with jobs_lock:
                    job = jobs.get(job_id, {})
                    job.update({"status": "done", "result": payload})
                    jobs[job_id] = job
            except Exception as exc:  # pragma: no cover
                with jobs_lock:
                    job = jobs.get(job_id, {})
                    job.update({
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    jobs[job_id] = job

        threading.Thread(target=worker, daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    @app.get("/v1/agent/jobs/{job_id}")
    def get_job(job_id: str, http_request: Request) -> Any:
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.get("owner", "") != _user_of(http_request):
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/remote", include_in_schema=False)
    def remote() -> Any:
        html = (WEB_DIR / "remote.html").read_text(encoding="utf-8")
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.post("/v1/agent/run")
    def run(request: RunRequest, http_request: Request) -> Any:
        _rate_limit(http_request)
        try:
            result = get_agent(_resolve_mode(request)).ask(
                _task_with_attachment(request),
                session_id=_scoped_session(_user_of(http_request), request.session_id),
                history=_to_history(request),
            )
            return result.to_dict(include_steps=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/v1/agent/stream")
    def stream(request: StreamRequest, http_request: Request) -> StreamingResponse:
        runtime_agent = get_agent()
        scoped_session = _scoped_session(_user_of(http_request), request.session_id)

        def event_stream():
            try:
                for event in runtime_agent.stream(
                    request.task,
                    session_id=scoped_session,
                    history=_to_history(request),
                ):
                    # SSE 的 event 字段已表示类型，data 只下发业务 payload
                    payload = json.dumps(event.payload, ensure_ascii=False, default=str)
                    yield f"event: {event.type}\ndata: {payload}\n\n"
            except Exception as exc:  # pragma: no cover
                error = json.dumps({"error": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {error}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # Web UI（零构建静态页面，与 API 同源）
    # ------------------------------------------------------------------
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="web-static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            if os.environ.get("AGENT_ACCESS_TOKEN"):
                # 公网隧道下使用异步轮询页，避免长连接被隧道截断
                return RedirectResponse("/remote")
            return FileResponse(WEB_DIR / "index.html")

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon() -> Response:
            return Response(status_code=204)

    return app


# uvicorn agent.server:app
app = create_app()


def serve(
    config_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8077,
    *,
    open_browser: bool = False,
    fallback_offline: bool = True,
) -> None:
    """启动 uvicorn，同时托管 Web UI。"""
    if config_path:
        os.environ["AGENT_CONFIG"] = config_path
    # 公网隧道启动时自动启用 Basic Auth；仅影响 serve，不影响测试/本地 create_app
    token_file = Path(__file__).resolve().parents[1] / ".agent" / "access_token.txt"
    if not os.environ.get("AGENT_ACCESS_TOKEN") and token_file.exists():
        token = token_file.read_text(encoding="ascii", errors="ignore").strip()
        if token:
            os.environ["AGENT_ACCESS_TOKEN"] = token
    # Web UI 默认也要开箱即用：LLM 不可用时降级到离线策略（可用 --no-fallback-offline 关闭）
    if fallback_offline and not os.environ.get("AGENT_LLM_PROVIDER"):
        try:
            config = load_config(config_path)
            if config.llm.provider not in ("offline", "scripted"):
                health = create_llm(config.llm).health()
                if not health.get("healthy"):
                    reason = str(health.get("error") or "健康检查未通过").replace("\n", " ")
                    print(
                        f"[提示] LLM 后端 {config.llm.provider} 不可用（{reason[:160]}），"
                        "Web 服务将使用离线策略。",
                        file=sys.stderr,
                    )
                    os.environ["AGENT_LLM_PROVIDER"] = "offline"
                    os.environ["AGENT_LLM_MODEL"] = "offline-policy"
        except Exception as exc:
            print(f"[提示] LLM 健康检查失败，继续使用当前配置：{exc}", file=sys.stderr)
    if open_browser:
        url = f"http://{host}:{port}/"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    import uvicorn

    uvicorn.run("agent.server:app", host=host, port=port, reload=False, log_level="info")


__all__ = ["app", "create_app", "serve"]
