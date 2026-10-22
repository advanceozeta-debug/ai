"""오미크론의 '두뇌'(언어 모델) 고르기.

환경 변수 BRAIN 으로 정합니다.
  claude      Anthropic Claude (기본값, ANTHROPIC_API_KEY 필요, 유료)
  gemini      Google Gemini 무료 등급 (GEMINI_API_KEY, aistudio.google.com 에서 발급)
  groq        Groq 무료 등급 (GROQ_API_KEY, console.groq.com 에서 발급)
  openrouter  OpenRouter 무료 모델 (OPENROUTER_API_KEY, openrouter.ai 에서 발급)
BRAIN_MODEL 로 모델 이름을 바꿀 수 있습니다 (무료 모델 목록은 자주 바뀌어요).

claude 외의 두뇌는 모두 OpenAI 방식 API 로 부릅니다. 웹 검색은 claude 두뇌에만 있어요.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Brain:
    kind: str           # "claude" 또는 "openai" (OpenAI 방식 API)
    label: str          # 화면에 보여줄 이름
    model: str
    base_url: str | None = None
    api_key: str | None = None

    @property
    def web_search(self) -> bool:
        return self.kind == "claude"


PRESETS = {
    "gemini": ("Gemini (무료)", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash", "GEMINI_API_KEY"),
    "groq": ("Groq (무료)", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openrouter": ("OpenRouter (무료)", "https://openrouter.ai/api/v1", "openrouter/free", "OPENROUTER_API_KEY"),
}


def brain_from_env(env=os.environ) -> Brain:
    name = env.get("BRAIN", "claude").strip().lower()
    if name == "claude":
        return Brain("claude", "Claude", env.get("BRAIN_MODEL") or "claude-opus-5-5")
    if name not in PRESETS:
        raise ValueError(f"알 수 없는 BRAIN 값: {name} (claude, gemini, groq, openrouter 중 하나)")
    label, base_url, model, key_env = PRESETS[name]
    return Brain("openai", label, env.get("BRAIN_MODEL") or model, base_url, env.get(key_env) or env.get("BRAIN_API_KEY"))
