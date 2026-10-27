# 한국어 학습 데이터 만들기
#  1) pretrain.txt : 한국어 자체를 배우는 글 (영화 리뷰 20만 + 대화 문장들)
#  2) dialog.txt   : "나: … / 오미크론: …" 형식의 대화 (일상 대화 + persona.py 성격)
#
# 데이터 출처 (둘 다 공개 데이터, GitHub에서 받음)
#  - 한국어 챗봇 데이터 (songys/Chatbot_data, MIT 라이선스)
#  - 네이버 영화 리뷰 NSMC (e9t/nsmc, CC0 퍼블릭 도메인)
# 실행: python prepare_korean.py   (corpus/ 폴더에 두 파일이 있어야 함)
import csv
import os
import random
import re
import subprocess
from persona import NAME, USER

random.seed(0)
CORPUS = "corpus"
SOURCES = {
    "ChatbotData.csv": "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
    "ratings.txt": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt",
}
os.makedirs(CORPUS, exist_ok=True)
for name, url in SOURCES.items():
    path = os.path.join(CORPUS, name)
    if not os.path.exists(path):
        print("받는 중:", url)
        subprocess.run(["curl", "-sSL", "-o", path, url], check=True)


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"([ㅋㅎㅠㅜ.!?~])\1{3,}", r"\1\1\1", s)  # ㅋㅋㅋㅋㅋㅋ → ㅋㅋㅋ
    return s


def polite(a):
    a = re.sub(r"[\s.!?~]+$", "", a)
    return a.endswith(("요", "니다", "니까", "세요", "죠"))


# ---- 대화 데이터 ----
pairs = []
for r in csv.DictReader(open(f"{CORPUS}/ChatbotData.csv", encoding="utf-8")):
    q, a = clean(r["Q"]), clean(r["A"])
    if q and a and polite(a):            # 오미크론은 존댓말만 쓰니까 존댓말 대답만 사용
        pairs.append((q, a))

# ---- 영화 리뷰 ----
reviews = []
for line in open(f"{CORPUS}/ratings.txt", encoding="utf-8").read().splitlines()[1:]:
    parts = line.split("\t")
    if len(parts) == 3:
        t = clean(parts[1])
        if 4 <= len(t) <= 140 and re.search("[가-힣]", t):
            reviews.append(t)
reviews = list(dict.fromkeys(reviews))  # 중복 제거

# ---- 성격 대화 (persona.py → make_data.py 가 만든 data.txt) ----
subprocess.run(["python", "make_data.py"], check=True)
persona_dialogs = open("data.txt", encoding="utf-8").read().split("\n\n")


def dialog_block(n_turns):
    return "".join(f"{USER}: {q}\n{NAME}: {a}\n" for q, a in random.sample(pairs, n_turns))


# 1) 한국어 기초: 리뷰 + 대화 문장 (대화는 형식 그대로 넣어 '주고받기'도 미리 익히게)
pre = reviews[:]
pre += [dialog_block(random.randint(1, 3)).strip() for _ in range(len(pairs))]
random.shuffle(pre)
open("pretrain.txt", "w", encoding="utf-8").write("\n".join(pre))

# 2) 대화 + 성격: 일상 대화 3번 반복분 + 성격 대화 (비율 약 6:4)
dlg = [dialog_block(random.randint(1, 4)) for _ in range(len(pairs) * 3 // 2)]
dlg += persona_dialogs[: len(dlg) * 2 // 3]
random.shuffle(dlg)
open("dialog.txt", "w", encoding="utf-8").write("\n".join(dlg))

print(f"대화 {len(pairs):,}쌍 | 리뷰 {len(reviews):,}개")
print(f"pretrain.txt {sum(map(len, pre)):,}글자 | dialog.txt {sum(map(len, dlg)):,}글자")
