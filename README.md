# 나만의 AI 비서 (my_ai)

Anthropic의 오픈소스 [Python SDK](https://github.com/anthropics/anthropic-sdk-python)로 Claude API 위에 만든 AI 비서입니다.
Claude 모델 가중치는 공개되어 있지 않기 때문에, 모델을 새로 학습하는 대신 **성격 · 기억 · 도구**를 직접 설계해서 나만의 AI를 만듭니다.

## 할 수 있는 것

| 기능 | 설명 |
|---|---|
| 💬 대화 | 실시간 스트리밍으로 답변 |
| 🎭 성격 4종 | `friend`(대화 친구), `tutor`(공부), `coder`(코딩), `support`(고객 상담) |
| 🧠 장기 기억 | 대화 중 알게 된 사실을 `data/memory.json`에 저장, 다음 실행에도 기억 |
| 🧮 도구 | 계산기, 현재 시각, 웹 검색, `data/workspace/` 파일 읽기 |

## 실행 방법

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # https://console.anthropic.com 에서 발급
python -m my_ai                          # 기본: friend
python -m my_ai --persona tutor          # 공부 선생님
python -m my_ai --effort high            # 더 깊이 생각 (느리고 비용 증가)
python -m my_ai --no-web                 # 웹 검색 끄기
```

대화 중 명령어: `/persona <이름>`, `/memory`, `/reset`, `/help`, `/quit`

## 직접 고쳐보기

- **새 성격 추가** → `my_ai/personas.py`의 `PERSONAS`에 항목 하나 추가
- **새 도구 추가** → `my_ai/tools.py`의 `TOOLS`에 스키마, `run_tool`에 실행 코드 추가
- **모델 바꾸기** → `my_ai/agent.py`의 `MODEL` (기본 `claude-opus-5-5`; 저렴하게는 `claude-sonnet-5-5`, `claude-haiku-5-5`)

## 구조

```
my_ai/
  agent.py     Claude 호출 + 도구 사용 루프
  personas.py  성격(시스템 프롬프트)
  tools.py     도구 정의와 실행
  memory.py    장기 기억 저장
  cli.py       터미널 대화 화면
tests/         API 키 없이 돌아가는 테스트 (python -m pytest)
```

안전 분류기가 요청을 거절하면 서버 측 `fallbacks: "default"`로 다른 모델에서 자동 재시도합니다.
