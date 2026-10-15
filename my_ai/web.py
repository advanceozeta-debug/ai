"""브라우저에서 AI와 대화하는 웹 서버. 실행: python -m my_ai.web"""

import argparse
import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Agent, friendly_error
from .memory import Memory
from .personas import DEFAULT_PERSONA, PERSONAS

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SessionRequest(BaseModel):
    persona: str = DEFAULT_PERSONA


def create_app(data_dir: Path = Path("data"), effort: str = "medium", web_search: bool = True, client=None) -> FastAPI:
    app = FastAPI(title="오미크론")
    # 브라우저 탭마다 대화 세션 하나. 기억(memory.json)은 모든 세션이 공유합니다.
    sessions: dict[str, tuple[Agent, threading.Lock]] = {}

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/personas")
    def personas():
        return [{"key": k, "name": v["name"], "description": v["description"]} for k, v in PERSONAS.items()]

    @app.post("/api/session")
    def new_session(req: SessionRequest):
        if req.persona not in PERSONAS:
            raise HTTPException(400, "알 수 없는 성격입니다")
        agent = Agent(req.persona, data_dir, effort=effort, web_search=web_search, client=client)
        session_id = uuid.uuid4().hex
        sessions[session_id] = (agent, threading.Lock())
        return {"session_id": session_id, "name": PERSONAS[req.persona]["name"]}

    @app.get("/api/memory")
    def memory():
        return {"facts": Memory(data_dir / "memory.json").facts}  # 매번 파일에서 최신 내용 읽기

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if req.session_id not in sessions:
            raise HTTPException(404, "세션이 만료됐어요. 새로고침해 주세요.")
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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--no-web", action="store_true", help="웹 검색 끄기")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    app = create_app(args.data_dir, args.effort, not args.no_web)
    print(f"🌐 브라우저에서 http://{args.host}:{args.port} 을 여세요")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
