"""웹 서버 테스트. 가짜 Claude 클라이언트로 API 키 없이 돌아갑니다."""

import json
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from my_ai.web import create_app
from tests.test_my_ai import FakeClient


def test_chat_streams_text_and_tools(tmp_path):
    fake = FakeClient([
        NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t1", name="remember", input={"fact": "고양이를 키움"})]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="고양이 이름이 뭐예요?")]),
    ])
    web = TestClient(create_app(tmp_path, client=fake))

    assert web.get("/").status_code == 200
    assert {p["key"] for p in web.get("/api/personas").json()} >= {"friend", "tutor"}

    session_id = web.post("/api/session", json={"persona": "friend"}).json()["session_id"]
    res = web.post("/api/chat", json={"session_id": session_id, "message": "나 고양이 키워"})
    events = [json.loads(line) for line in res.text.splitlines()]
    assert events[0] == {"type": "tool", "name": "remember", "input": {"fact": "고양이를 키움"}}
    assert {"type": "text", "text": "고양이 이름이 뭐예요?"} in events
    assert events[-1] == {"type": "done"}
    assert web.get("/api/memory").json() == {"facts": ["고양이를 키움"]}


def test_chat_reports_errors(tmp_path):
    class Boom(FakeClient):
        def _stream(self, **kwargs):
            raise TypeError("Could not resolve authentication method")

    web = TestClient(create_app(tmp_path, client=Boom([])))
    session_id = web.post("/api/session", json={"persona": "friend"}).json()["session_id"]
    events = [json.loads(l) for l in web.post("/api/chat", json={"session_id": session_id, "message": "hi"}).text.splitlines()]
    assert events[0]["type"] == "error" and "API 키" in events[0]["message"]
    assert web.post("/api/chat", json={"session_id": "nope", "message": "hi"}).status_code == 404
