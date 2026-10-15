"""터미널에서 AI와 대화하는 프로그램."""

import argparse
from pathlib import Path

import anthropic

from .agent import Agent
from .personas import DEFAULT_PERSONA, PERSONAS

HELP = """명령어:
  /persona <이름>  성격 바꾸기 (대화 새로 시작)
  /memory          기억하는 내용 보기
  /reset           대화 새로 시작
  /help            도움말
  /quit            종료"""


def main() -> None:
    parser = argparse.ArgumentParser(description="나만의 AI 비서")
    parser.add_argument("--persona", default=DEFAULT_PERSONA, choices=PERSONAS)
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--no-web", action="store_true", help="웹 검색 끄기")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    def new_agent(persona: str) -> Agent:
        return Agent(persona, args.data_dir, effort=args.effort, web_search=not args.no_web)

    persona = args.persona
    agent = new_agent(persona)
    print(f"🤖 {PERSONAS[persona]['name']} ({PERSONAS[persona]['description']})")
    print("성격 목록: " + ", ".join(PERSONAS) + "   |   /help 로 명령어 보기\n")

    while True:
        try:
            user = input("나 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/help":
            print(HELP)
            continue
        if user == "/memory":
            print(agent.memory.as_text())
            continue
        if user == "/reset":
            agent = new_agent(persona)
            print("(대화를 새로 시작합니다)")
            continue
        if user.startswith("/persona"):
            choice = user.removeprefix("/persona").strip()
            if choice not in PERSONAS:
                print("가능한 성격: " + ", ".join(f"{k}({v['description']})" for k, v in PERSONAS.items()))
                continue
            persona = choice
            agent = new_agent(persona)
            print(f"🤖 {PERSONAS[persona]['name']}(으)로 바뀌었어요.")
            continue

        print(f"{PERSONAS[persona]['name']} > ", end="", flush=True)
        try:
            agent.ask(user, on_tool=lambda name, a: print(f"\n  [도구 사용: {name} {a}]", flush=True))
        except anthropic.AuthenticationError:
            print("\nAPI 키가 올바르지 않아요. ANTHROPIC_API_KEY 를 확인해 주세요.")
        except TypeError as e:
            if "authentication" not in str(e):
                raise
            print("\nAPI 키가 없어요. export ANTHROPIC_API_KEY=... 로 설정한 뒤 다시 실행해 주세요.")
        except anthropic.RateLimitError:
            print("\n요청이 너무 많아요. 잠시 후 다시 시도해 주세요.")
        except anthropic.APIStatusError as e:
            print(f"\nAPI 오류 ({e.status_code}): {e.message}")
        except anthropic.APIConnectionError:
            print("\n네트워크 연결을 확인해 주세요.")
        print("\n")


if __name__ == "__main__":
    main()
