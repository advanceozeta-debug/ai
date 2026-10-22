"""API 키 없이 돌아가는 테스트. Claude 대신 가짜 클라이언트를 씁니다."""

from types import SimpleNamespace as NS

import pytest

from my_ai.agent import Agent
from my_ai.memory import Memory
from my_ai.tools import run_tool, safe_eval


def test_calculator():
    assert safe_eval("(3+4)*2**3") == 56
    assert abs(safe_eval("sqrt(16) + pi") - (4 + 3.14159265)) < 1e-6
    with pytest.raises(ValueError):
        safe_eval("__import__('os')")


def test_memory_persists(tmp_path):
    m = Memory(tmp_path / "m.json")
    m.add("이름은 민수")
    assert Memory(tmp_path / "m.json").facts == ["이름은 민수"]
    m.forget("민수")
    assert Memory(tmp_path / "m.json").facts == []


def test_read_file_stays_in_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("hello", encoding="utf-8")
    mem = Memory(tmp_path / "m.json")
    assert run_tool("read_file", {"path": "a.txt"}, mem, ws) == ("hello", False)
    result, is_error = run_tool("read_file", {"path": "../m.json"}, mem, ws)
    assert is_error and "workspace" in result


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for b in self.message.content:
            if b.type == "text":
                yield NS(type="text", text=b.text)

    def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return FakeStream(self.replies.pop(0))


def test_agent_tool_loop(tmp_path):
    client = FakeClient([
        NS(stop_reason="tool_use", content=[
            NS(type="tool_use", id="t1", name="calculator", input={"expression": "6*7"}),
            NS(type="tool_use", id="t2", name="remember", input={"fact": "좋아하는 숫자는 42"}),
        ]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="답은 42예요!")]),
    ])
    agent = Agent("tutor", tmp_path, client=client)
    out = []
    assert agent.ask("6 곱하기 7은?", on_text=out.append) == "답은 42예요!"
    assert "".join(out) == "답은 42예요!"

    # 두 도구 결과가 한 메시지로 돌아갔는지
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["content"] for r in results] == ["42", "기억했습니다: 좋아하는 숫자는 42"]
    assert client.calls[0]["model"] == "claude-opus-5-5"
    assert client.calls[0]["fallbacks"] == "default"

    # 새 세션은 저장된 기억을 system 프롬프트에 담고 시작
    assert "좋아하는 숫자는 42" in Agent("friend", tmp_path, client=client).system


def test_agent_rolls_back_on_error(tmp_path):
    class Boom(FakeClient):
        def _stream(self, **kwargs):
            raise RuntimeError("network")

    agent = Agent("friend", tmp_path, client=Boom([]))
    with pytest.raises(RuntimeError):
        agent.ask("안녕", on_text=lambda t: None)
    assert agent.messages == []


def test_credit_error_is_friendly():
    import httpx2 as httpx
    import anthropic
    from my_ai.agent import friendly_error

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    res = httpx.Response(400, request=req)
    err = anthropic.BadRequestError("Your credit balance is too low to access the Anthropic API.", response=res, body=None)
    assert "크레딧" in friendly_error(err)
