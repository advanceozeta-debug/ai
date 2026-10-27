# omicron_colab.ipynb 를 만드는 스크립트 (노트북 내용을 고치려면 여기를 고치고 다시 실행)
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# 오미크론 키우기 (Colab GPU)

처음부터 직접 만든 언어 모델 **오미크론**에게 한국어 약 9000만 글자를 읽히는 노트북이에요.

**시작하기 전에 딱 한 번**
1. 위 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** 를 고르고 저장
2. 메뉴 **런타임 → 모두 실행**

전부 끝나는 데 **약 40~60분** 걸려요. 그동안 이 탭을 닫지 말고 화면이 꺼지지 않게 해 주세요.
끝나면 마지막에 `model.bin`, `tokenizer.json` 두 파일이 다운로드돼요.
""")

code("""
# 1. GPU 확인 + 오미크론 코드 받기
import torch
assert torch.cuda.is_available(), "GPU가 꺼져 있어요! 메뉴 런타임 → 런타임 유형 변경 → T4 GPU 를 고르고 다시 '모두 실행' 하세요."
print("GPU:", torch.cuda.get_device_name(0))
!rm -rf /content/omicron && git clone -q --depth 1 https://github.com/advanceozeta-debug/ai /content/omicron
%cd /content/omicron
!pip -q install tokenizers datasets
""")

md("""
## 설정
숫자를 바꾸면 두뇌 크기와 학습량이 달라져요. 처음에는 그대로 두는 걸 추천해요.
- `N_EMBD`, `N_LAYER`: 두뇌 크기 (지금 약 2천만 숫자). 키우면 똑똑해지지만 느려지고 웹페이지도 무거워져요.
- `ITERS_1`: 한국어 학습 횟수. 5000회 ≈ 9000만 글자를 두 번 읽는 양.
""")

code("""
# 2. 설정
WIKI_CHARS = 82_000_000   # 위키백과에서 읽을 글자 수 (+ 리뷰·대화 약 1000만 = 약 9000만)
VOCAB      = 16000        # 단어 조각 개수
N_EMBD, N_LAYER, N_HEAD, BLOCK = 384, 8, 6, 256
ITERS_1    = 5000         # 1단계: 한국어
ITERS_2    = 800          # 2단계: 대화 + 성격
""")

code("""
# 3. 데이터 준비 (약 10분): 위키백과 받기 → 단어 조각 사전 → 번호로 바꾸기
!python big_prepare.py --wiki-chars {WIKI_CHARS} --vocab {VOCAB}
""")

code("""
# 4. 1단계: 한국어 학습 (약 25~35분)
#    500회마다 오차와 오미크론의 대답이 나와요. 점점 말이 되는 걸 볼 수 있어요.
!python big_train.py --data data/pre --iters {ITERS_1} --n-embd {N_EMBD} --n-layer {N_LAYER} --n-head {N_HEAD} --block {BLOCK} --out base_big.pt
""")

code("""
# 5. 2단계: 대화 + 오미크론 성격 (약 5분)
#    대화만 배우면 한국어를 잊어버리니까 위키백과 글을 30% 섞어서 읽어요.
!python big_train.py --data data/dlg --mix data/pre --mix-ratio 0.3 --init base_big.pt --iters {ITERS_2} --lr 3e-4 --dropout 0.1 --eval-every 200 --out omicron_big.pt
""")

code("""
# 6. 처음 보는 말로 시험해 보기
from big_chat import Omicron
bot = Omicron("omicron_big.pt", "data/tokenizer.json")
for q in ["안녕", "너 누구야", "오늘 시험 망쳤어", "주말에 친구랑 놀이공원 가", "요즘 잠이 안 와",
          "강아지 키우고 싶어", "비 오니까 기분이 꿀꿀해", "대한민국의 수도는 어디야"]:
    bot.history = ""
    print(f"나: {q}\\n오미크론: {bot.reply(q)}\\n")
""")

md("""
## 웹페이지에 새 두뇌 넣기
아래 칸이 끝나면 `model.bin`, `tokenizer.json` 이 다운로드돼요. (브라우저가 "여러 파일 다운로드 허용"을 물으면 허용)

그다음 GitHub에서:
1. https://github.com/advanceozeta-debug/ai/tree/main/brain 열기
2. **Add file → Upload files** → 받은 두 파일 선택
3. **Commit changes** 누르기

1~2분 뒤 https://advanceozeta-debug.github.io/ai/ 를 새로고침하면 새 오미크론이에요.
""")

code("""
# 7. 웹페이지용으로 내보내고 다운로드
!python export_brain.py --model omicron_big.pt --tokenizer data/tokenizer.json --out brain_new
from google.colab import files
files.download("brain_new/model.bin")
files.download("brain_new/tokenizer.json")
""")

code("""
# 8. (선택) 여기서 직접 대화하기. '끝' 을 입력하면 멈춰요.
bot.history = ""
while (q := input("나: ").strip()) not in ("끝", ""):
    print("오미크론:", bot.reply(q), "\\n")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("omicron_colab.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("omicron_colab.ipynb 생성")
