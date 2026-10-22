"""AI의 두뇌: Claude API를 호출하고, 도구 사용 루프를 돌립니다."""

from collections.abc import Callable
from pathlib import Path

import anthropic

from .personas import build_system_prompt
from .storage import FileStore
from .tools import TOOLS, WEB_SEARCH_TOOL, run_tool

MODEL = "claude-opus-5-5"
MAX_STEPS = 20  # 한 질문에 도구를 연달아 쓸 수 있는 최대 횟수


def friendly_error(e: Exception) -> str | None:
    """API 오류를 사용자용 한국어 문구로 바꿉니다. 모르는 오류면 None."""
    if isinstance(e, anthropic.AuthenticationError):
        return "API 키가 올바르지 않아요. ANTHROPIC_API_KEY 를 확인해 주세요."
    if isinstance(e, TypeError) and "authentication" in str(e):
        return "API 키가 없어요. export ANTHROPIC_API_KEY=... 로 설정한 뒤 다시 실행해 주세요."
    if isinstance(e, anthropic.RateLimitError):
        return "요청이 너무 많아요. 잠시 후 다시 시도해 주세요."
    if isinstance(e, anthropic.APIStatusError):
        return f"API 오류 ({e.status_code}): {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "네트워크 연결을 확인해 주세요."
    return None


class Agent:
    def __init__(
        self,
        persona: str,
        data_dir: Path = Path("data"),
        effort: str = "medium",
        web_search: bool = True,
        client: anthropic.Anthropic | None = None,
        store=None,
        conversation_id: str | None = None,
    ):
        self.client = client or anthropic.Anthropic()
        # store: 기억 + 대화 기록 저장소 (storage.py). 없으면 data/ 폴더의 파일을 씁니다.
        self.memory = store or FileStore(data_dir)
        self.workspace = data_dir / "workspace"
        self.effort = effort
        self.tools = TOOLS + ([WEB_SEARCH_TOOL] if web_search else [])
        self.conversation_id = conversation_id
        # 이어서 하는 대화는 저장해 둔 질문·답 글로 시작합니다.
        self.history = self.memory.load_messages(conversation_id) if conversation_id else []
        self.messages: list = [{"role": m["role"], "content": m["text"]} for m in self.history if m["text"]]
        # 기억은 세션 시작 시점에 한 번만 넣습니다. 대화 중에 system을 바꾸면 프롬프트 캐시가 깨지기 때문입니다.
        self.system = (
            f"{build_system_prompt(persona)}\n\n"
            f"지금까지 사용자에 대해 기억하는 내용:\n{self.memory.as_text()}"
        )

    def ask(
        self,
        user_text: str,
        on_text: Callable[[str], None] = lambda t: print(t, end="", flush=True),
        on_tool: Callable[[str, dict], None] = lambda n, a: None,
    ) -> str:
        """사용자 메시지를 보내고 최종 답변 텍스트를 돌려줍니다. 답변은 on_text로 실시간 전달됩니다."""
        start = len(self.messages)
        shown: list[str] = []

        def collect(text: str) -> None:
            shown.append(text)
            on_text(text)

        try:
            result = self._run(user_text, collect, on_tool)
        except Exception:
            del self.messages[start:]  # API 오류 시 이번 턴을 통째로 되돌려 다음 질문이 깨끗하게 시작되도록
            raise
        if self.conversation_id:
            turn = [{"role": "user", "text": user_text}, {"role": "assistant", "text": "".join(shown).strip()}]
            self.memory.append_messages(self.conversation_id, turn)
            self.history.extend(turn)
        return result

    def _run(self, user_text: str, on_text: Callable[[str], None], on_tool: Callable[[str, dict], None]) -> str:
        self.messages.append({"role": "user", "content": user_text})
        final_text = ""

        for _ in range(MAX_STEPS):
            with self.client.beta.messages.stream(
                model=MODEL,
                max_tokens=64000,
                system=self.system,
                messages=self.messages,
                tools=self.tools,
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
                # 안전 분류기가 요청을 거절하면 Anthropic 권장 모델로 자동 재시도
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                for event in stream:
                    if event.type == "text":
                        on_text(event.text)
                response = stream.get_final_message()

            # thinking 블록 등을 포함한 응답 전체를 그대로 이어 붙입니다 (히스토리는 덧붙이기만).
            self.messages.append({"role": "assistant", "content": response.content})
            final_text = "".join(b.text for b in response.content if b.type == "text")

            if response.stop_reason == "refusal":
                notice = "\n(이 요청에는 답변할 수 없어요. 다르게 질문해 주세요.)"
                on_text(notice)
                return final_text + notice
            if response.stop_reason == "pause_turn":  # 웹 검색이 길어져 잠시 멈춤 → 이어서 진행
                continue
            if response.stop_reason != "tool_use":
                if response.stop_reason == "max_tokens":
                    on_text("\n(답변이 길어서 중간에 끊겼어요.)")
                return final_text

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                on_tool(block.name, block.input)
                result, is_error = run_tool(block.name, block.input, self.memory, self.workspace)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result, "is_error": is_error}
                )
            # 병렬 도구 호출 결과는 한 메시지로 모아서 보냅니다.
            self.messages.append({"role": "user", "content": tool_results})

        on_text("\n(도구를 너무 많이 써서 여기서 멈췄어요.)")
        return final_text
