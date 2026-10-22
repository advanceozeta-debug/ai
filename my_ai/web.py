"""브라우저(태블릿 포함)에서 오미크론과 대화하는 웹 서버. 실행: python -m my_ai.web

환경 변수
  ANTHROPIC_API_KEY  Claude API 키 (필수)
  DATABASE_URL       PostgreSQL 주소. 있으면 기억과 대화 기록을 여기에 저장 (예: Neon)
  APP_PASSWORD       있으면 이 비밀번호를 아는 사람만 들어올 수 있음. 인터넷에 올릴 때는 꼭 설정하세요.
"""

import argparse
import hashlib
import hmac
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Agent, friendly_error
from .personas import DEFAULT_PERSONA, PERSONAS
from .storage import open_store

STATIC_DIR = Path(__file__).parent / "static"
COOKIE = "omicron_auth"
PUBLIC_PATHS = {"/", "/api/auth", "/api/login", "/manifest.webmanifest", "/icon-192.png", "/icon-512.png", "/apple-touch-icon.png"}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SessionRequest(BaseModel):
    persona: str = DEFAULT_PERSONA
    resume: bool = True


class LoginRequest(BaseModel):
    password: str


def create_app(
    data_dir: Path = Path("data"),
    effort: str = "medium",
    web_search: bool = True,
    client=None,
    database_url: str | None = None,
    password: str | None = None,
    locked: bool = False,
) -> FastAPI:
    app = FastAPI(title="오미크론")
    store = open_store(data_dir, database_url)
    # 브라우저 탭마다 대화 세션 하나. 서버가 다시 켜지면 사라지지만, 기록은 저장소에 남아 이어서 열 수 있어요.
    sessions: dict[str, tuple[Agent, threading.Lock]] = {}
    if locked:  # 인터넷에 올렸는데 비밀번호가 없으면 아무도 못 들어오게 잠급니다
        password = None
    token = hmac.new(password.encode(), b"omicron-session-v1", hashlib.sha256).hexdigest() if password else None
    if locked:
        token = uuid.uuid4().hex + uuid.uuid4().hex

    def authed(request: Request) -> bool:
        return token is None or hmac.compare_digest(request.cookies.get(COOKIE, ""), token)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        if request.url.path not in PUBLIC_PATHS and not authed(request):
            return JSONResponse({"detail": "로그인이 필요해요."}, status_code=401)
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/manifest.webmanifest")
    def manifest():
        return FileResponse(STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/{name}.png")
    def icon(name: str):
        if name not in ("icon-192", "icon-512", "apple-touch-icon"):
            raise HTTPException(404)
        return FileResponse(STATIC_DIR / f"{name}.png", media_type="image/png")

    @app.get("/api/auth")
    def auth_state(request: Request):
        return {"required": token is not None, "ok": authed(request), "locked": locked}

    @app.post("/api/login")
    def login(req: LoginRequest, request: Request, response: Response):
        if token is None:
            return {"ok": True}
        if locked:
            raise HTTPException(503, "서버에 APP_PASSWORD 가 아직 설정되지 않았어요. Render 환경 변수에 넣어 주세요.")
        if not hmac.compare_digest(req.password.encode(), password.encode()):
            time.sleep(1)  # 비밀번호 마구 넣어보기를 느리게
            raise HTTPException(401, "비밀번호가 틀렸어요.")
        secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
        response.set_cookie(COOKIE, token, max_age=60 * 60 * 24 * 365, httponly=True, samesite="lax", secure=secure)
        return {"ok": True}

    @app.get("/api/personas")
    def personas():
        return [{"key": k, "name": v["name"], "description": v["description"]} for k, v in PERSONAS.items()]

    @app.post("/api/session")
    def new_session(req: SessionRequest):
        if req.persona not in PERSONAS:
            raise HTTPException(400, "알 수 없는 성격입니다")
        conv_id = (store.latest_conversation(req.persona) if req.resume else None) or store.new_conversation(req.persona)
        agent = Agent(req.persona, data_dir, effort=effort, web_search=web_search, client=client, store=store, conversation_id=conv_id)
        session_id = uuid.uuid4().hex
        sessions[session_id] = (agent, threading.Lock())
        return {"session_id": session_id, "name": PERSONAS[req.persona]["name"], "history": agent.history}

    @app.get("/api/memory")
    def memory():
        return {"facts": store.facts}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if req.session_id not in sessions:
            raise HTTPException(404, "세션이 만료됐어요.")
        agent, lock = sessions[req.session_id]
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "아직 이전 답변을 쓰는 중이에요.")

        events: queue.Queue = queue.Queue()

        def worker():
            try:
                agent.ask(
                    req.message,
                    on_text=lambda t: events.put({"type": "text", "text": t}),
                    on_tool=lambda name, args: events.put({"type": "tool", "name": name, "input": args}),
                )
            except Exception as e:
                events.put({"type": "error", "message": friendly_error(e) or f"알 수 없는 오류: {e}"})
            finally:
                lock.release()
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def stream():
            # 한 줄에 JSON 하나씩 (NDJSON) 보내면 브라우저가 실시간으로 받아 그립니다.
            while (event := events.get()) is not None:
                yield json.dumps(event, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="오미크론 웹 화면")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--no-web", action="store_true", help="웹 검색 끄기")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    password = os.environ.get("APP_PASSWORD") or None
    public = args.host not in ("127.0.0.1", "localhost")
    if public and not password:
        print("⚠️  APP_PASSWORD 가 없어서 잠근 상태로 엽니다. 비밀번호를 설정하면 들어갈 수 있어요.")
    app = create_app(
        args.data_dir, args.effort, not args.no_web,
        database_url=os.environ.get("DATABASE_URL"), password=password, locked=public and not password,
    )
    print(f"🌐 브라우저에서 http://{args.host}:{args.port} 을 여세요")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
