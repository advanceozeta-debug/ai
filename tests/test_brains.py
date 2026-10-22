"""무료 두뇌(OpenAI 방식 API) 테스트. 진짜 openai 라이브러리를 가짜 서버에 연결해 확인합니다."""

import json

import httpx2 as httpx
import openai
from fastapi.testclient import TestClient

from my_ai.agent import Agent
from my_ai.brains import brain_from_env
from my_ai.web import create_app


def sse(chunks):
    body = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def chunk(delta, finish=None):
    return {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def fake_server(requests):
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:  # 첫 응답: 도구 호출 두 개, 인자가 조각나서 옴
            return sse([
                chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "calculator", "arguments": '{"expre'}}]}),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ssion": "6*7"}'}}]}),
                chunk({"tool_calls": [{"index": 1, "id": "", "type": "function", "function": {"name": "remember", "arguments": '{"fact": "고양이를 키움"}'}}]}),
                chunk({}, "tool_calls"),
            ])
        return sse([chunk({"role": "assistant", "content": "답은 "}), chunk({"content": "42예요!"}), chunk({}, "stop")])
    return handler


def make_client(requests):
    return openai.OpenAI(api_key="test", base_url="https://free.example/v1", http_client=httpx.Client(transport=httpx.MockTransport(fake_server(requests))))


def test_brain_presets():
    assert brain_from_env({}).kind == "claude"
    b = brain_from_env({"BRAIN": "gemini", "GEMINI_API_KEY": "k"})
    assert (b.kind, b.api_key, b.web_search) == ("openai", "k", False)
    assert brain_from_env({"BRAIN": "groq", "BRAIN_MODEL": "x"}).model == "x"


def test_free_brain_tool_loop(tmp_path):
    requests = []
    brain = brain_from_env({"BRAIN": "groq", "GROQ_API_KEY": "test"})
    agent = Agent("friend", tmp_path, client=make_client(requests), brain=brain)
    out, tools = [], []
    assert agent.ask("6 곱하기 7? 그리고 나 고양이 키워", on_text=out.append, on_tool=lambda n, a: tools.append((n, a))) == "답은 42예요!"
    assert "".join(out) == "답은 42예요!"
    assert tools == [("calculator", {"expression": "6*7"}), ("remember", {"fact": "고양이를 키움"})]
    assert agent.memory.facts == ["고양이를 키움"]

    first, second = requests
    assert first["model"] == "llama-3.3-70b-versatile" and first["messages"][0]["role"] == "system"
    assert "web_search" not in {t["function"]["name"] for t in first["tools"]}
    assert "웹 검색을 쓸 수 없어요" in first["messages"][0]["content"]
    sent = second["messages"]
    assert sent[-3]["tool_calls"][1]["id"].startswith("call_")  # 빈 id 는 새로 만들어 붙임
    assert [(m["role"], m["content"]) for m in sent[-2:]] == [("tool", "42"), ("tool", "기억했습니다: 고양이를 키움")]


def test_web_app_with_free_brain(tmp_path):
    requests = []
    brain = brain_from_env({"BRAIN": "gemini", "GEMINI_API_KEY": "test"})
    web = TestClient(create_app(tmp_path, client=make_client(requests), brain=brain))
    assert web.get("/api/info").json() == {"brain": "Gemini (무료)", "model": "gemini-2.5-flash", "web_search": False}
    sid = web.post("/api/session", json={"persona": "friend"}).json()["session_id"]
    events = [json.loads(l) for l in web.post("/api/chat", json={"session_id": sid, "message": "hi"}).text.splitlines()]
    assert "".join(e["text"] for e in events if e["type"] == "text") == "답은 42예요!"
    # 대화 기록도 저장되어 다시 열면 이어짐
    assert web.post("/api/session", json={"persona": "friend"}).json()["history"][-1]["text"] == "답은 42예요!"
