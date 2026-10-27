# persona.py 의 대화 조각들을 섞어서 학습용 대화 텍스트(data.txt)를 만든다.
# 실행: python make_data.py
import random
import re
from persona import NAME, USER, SLOTS, INTENTS

random.seed(0)
N_DIALOGS = 6000      # 만들 대화 개수 (많을수록 성격이 더 단단해짐)
TURNS = (1, 4)        # 대화 하나당 주고받는 횟수 범위


def fill(text, chosen):
    """{food} 같은 빈칸을 채운다. 한 턴 안에서는 같은 단어로 통일."""
    def repl(m):
        key = m.group(1)
        if key not in chosen:
            chosen[key] = random.choice(SLOTS[key])
        return chosen[key]
    return re.sub(r"\{(\w+)\}", repl, text)


def one_turn():
    users, bots = random.choice(INTENTS)
    chosen = {}
    u = fill(random.choice(users), chosen)
    b = fill(random.choice(bots), chosen)
    # 말끝 변형을 조금 줘서 다양한 입력에 익숙해지게
    if random.random() < 0.2:
        u += random.choice(["?", "!", "~", "ㅎㅎ", ".."])
    return f"{USER}: {u}\n{NAME}: {b}\n"


lines = []
for _ in range(N_DIALOGS):
    lines.append("".join(one_turn() for _ in range(random.randint(*TURNS))))

text = "\n".join(lines)
with open("data.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"data.txt 생성: {len(text):,} 글자, 서로 다른 글자 {len(set(text))}개")
