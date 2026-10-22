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


def test_password_lock(tmp_path):
    web = TestClient(create_app(tmp_path, client=FakeClient([]), password="열려라참깨"))
    assert web.get("/api/auth").json() == {"required": True, "ok": False, "locked": False}
    assert web.get("/api/memory").status_code == 401
    assert web.get("/manifest.webmanifest").status_code == 200  # 홈 화면 설치용 파일은 로그인 없이
    assert web.post("/api/login", json={"password": "틀림"}).status_code == 401
    assert web.post("/api/login", json={"password": "열려라참깨"}).status_code == 200
    assert web.get("/api/auth").json()["ok"] is True
    assert web.get("/api/memory").status_code == 200


def test_locked_without_password(tmp_path):
    web = TestClient(create_app(tmp_path, client=FakeClient([]), locked=True))
    assert web.get("/api/auth").json()["locked"] is True
    assert web.post("/api/login", json={"password": ""}).status_code == 503
    assert web.post("/api/session", json={"persona": "friend"}).status_code == 401


def test_conversation_resumes_after_restart(tmp_path):
    fake = FakeClient([NS(stop_reason="end_turn", content=[NS(type="text", text="반가워요 지우님!")])])
    web = TestClient(create_app(tmp_path, client=fake))
    sid = web.post("/api/session", json={"persona": "friend"}).json()["session_id"]
    web.post("/api/chat", json={"session_id": sid, "message": "안녕, 나 지우야"})

    # 서버가 다시 켜진 상황: 새 앱, 같은 저장소
    fake2 = FakeClient([NS(stop_reason="end_turn", content=[NS(type="text", text="물론 기억하죠")])])
    web2 = TestClient(create_app(tmp_path, client=fake2))
    data = web2.post("/api/session", json={"persona": "friend"}).json()
    assert data["history"] == [{"role": "user", "text": "안녕, 나 지우야"}, {"role": "assistant", "text": "반가워요 지우님!"}]
    web2.post("/api/chat", json={"session_id": data["session_id"], "message": "내 이름 기억해?"})
    sent = fake2.calls[0]["messages"]
    assert [m["content"] for m in sent] == ["안녕, 나 지우야", "반가워요 지우님!", "내 이름 기억해?"]

    # 새 대화는 빈 기록으로 시작
    assert web2.post("/api/session", json={"persona": "friend", "resume": False}).json()["history"] == []
