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

### 🌐 웹 화면으로 쓰기

```bash
python -m my_ai.web              # 브라우저에서 http://127.0.0.1:8000 열기
python -m my_ai.web --port 9000  # 포트 바꾸기
```

왼쪽에서 성격을 고르고, AI가 기억하는 내용을 확인할 수 있어요. 휴대폰 화면과 다크 모드도 지원합니다.
기본으로 내 컴퓨터(127.0.0.1)에서만 열립니다. 다른 사람에게 공개하면 내 API 키로 요금이 나가니 주의하세요.

### 🔗 바로 써보기 (설치 없이)

**claude.ai에서:** https://claude.ai/artifact/32zMFeLrj7XvkSQv1WqUTd
로그인한 내 Claude 계정으로 대화합니다 (API 키 필요 없음). claude.ai 설정 → 커넥터에서 **Parallel Search**(무료)를 연결하면
웹 검색도 켜집니다. 기억은 사람마다 비공개로 저장돼요.

**크롬에서 (웹 검색 포함):** `artifact/my-ai.html` 파일을 받아서 크롬으로 엽니다.
(또는 위 링크의 "크롬용 파일로 저장" 버튼). 처음 열면 Anthropic API 키를 물어보고,
그 키로 Claude API를 직접 호출합니다. 키와 기억은 그 브라우저에만 저장돼요.

`artifact/index.html` 은 claude.ai에 올리는 원본이고, `artifact/my-ai.html` 은 같은 내용을 크롬에서 바로 열 수 있게 감싼 파일입니다.

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
  web.py       웹 서버 (FastAPI, 답변을 실시간 스트리밍)
  static/      웹 화면 (index.html 한 파일)
tests/         API 키 없이 돌아가는 테스트 (python -m pytest)
```

안전 분류기가 요청을 거절하면 서버 측 `fallbacks: "default"`로 다른 모델에서 자동 재시도합니다.
