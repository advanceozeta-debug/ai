# 오미크론 — 처음부터 직접 만든 AI

미리 학습된 모델을 하나도 쓰지 않고, **빈 숫자 덩어리에서 시작해서** 직접 읽힌 글로 한국어와 성격을 배운 언어 모델(GPT 구조)이에요.

- 대화하기: https://advanceozeta-debug.github.io/ai/ (태블릿·휴대폰 브라우저에서 바로, 계산은 기기 안에서)
- 크게 키우기: [![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/advanceozeta-debug/ai/blob/main/omicron_colab.ipynb)

## 오미크론 단계

| 단계 | 두뇌 숫자 | 읽은 글 | 사전 | 어디서 학습 |
|---|---|---|---|---|
| 1. 성격만 | 86만 | 성격 대화 70만 글자 | 글자 369개 | CPU 15분 |
| 2. 한국어 (지금 웹페이지) | 222만 | 리뷰·대화 1000만 글자 | 글자 2,185개 | CPU 1시간 반 |
| 3. 큰 오미크론 | 약 2천만 | 위키백과 포함 9000만 글자 | 단어 조각 16,000개 | Colab GPU 약 50분 |

## 큰 오미크론 만들기 (Colab)

1. 위 **Colab에서 열기** 버튼 → 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU**
2. **런타임 → 모두 실행** (약 50분)
3. 마지막에 받은 `model.bin`, `tokenizer.json` 을 이 저장소 `brain/` 폴더에 업로드 (**Add file → Upload files**)
4. 1~2분 뒤 웹페이지를 새로고침하면 새 두뇌로 대화

## 파일 구성

| 파일 | 하는 일 |
|---|---|
| `persona.py` | **성격 정의.** 이름, 말투, 대답들. 여기를 바꾸면 다른 AI가 됨 |
| `model.py` | GPT 본체: 어텐션, 트랜스포머 블록, 문장 생성 — 전부 직접 구현 |
| `tokenizer.py` | 글자 단위 사전 (1·2단계) |
| `make_data.py` | persona.py → 성격 대화 `data.txt` |
| `prepare_korean.py` | 리뷰·일상 대화 데이터 받기 → `pretrain.txt`, `dialog.txt` |
| `train.py`, `chat.py` | 1·2단계 학습과 대화 (CPU) |
| `big_prepare.py` | 위키백과 + 단어 조각 사전(BPE) + 번호로 바꾸기 (3단계) |
| `big_train.py`, `big_chat.py` | 3단계 학습(GPU)과 대화 |
| `export_brain.py` | 학습된 두뇌 → 웹페이지용 `brain/` (8비트 압축) |
| `index.html`, `web/brain_engine.js` | 웹 대화 화면과 브라우저용 두뇌 계산기 |
| `omicron_colab.ipynb` | Colab 노트북 (`make_notebook.py` 로 생성) |

## 직접 실행 (컴퓨터)

```bash
pip install torch tokenizers
# 2단계 (CPU)
python prepare_korean.py
python train.py --data pretrain.txt --iters 3000 --out base.pt --vocab-from pretrain.txt dialog.txt
python train.py --data dialog.txt --init base.pt --iters 1200 --lr 5e-4
python chat.py
python export_brain.py --model model.pt --tokenizer vocab.json      # → brain/
```

## 나만의 성격으로 바꾸기

1. `persona.py` 에서 `NAME` 과 `INTENTS` 의 대답들을 원하는 말투로 고쳐요.
2. 새 주제는 `([사람이 할 말들], [AI 대답들])` 묶음으로 추가해요.
3. 다시 학습하면 새 성격으로 태어나요.

## 할 수 있는 것과 없는 것

- 할 수 있음: 자연스러운 한국어 문장, 감정에 맞춘 짧은 대답, 정해 준 말투와 성격 유지
- 어려움: 정확한 지식, 계산, 긴 논리, 코딩. 이런 걸 잘하는 AI는 이보다 수백~수만 배 크고 수천억 글자를 읽어요.

## 원리 한 줄 요약

모델은 "지금까지 나온 글을 보고 다음 조각 맞히기"만 배워요. 이걸 수천만 번 반복하면 문법과 말투가 숫자 속에 새겨져요.

사용한 데이터: 한국어 위키백과 (wikimedia/wikipedia, CC BY-SA), 한국어 챗봇 데이터 (songys/Chatbot_data, MIT), 네이버 영화 리뷰 NSMC (e9t/nsmc, CC0).
