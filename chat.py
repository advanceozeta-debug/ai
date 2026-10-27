# 학습된 모델과 대화하기
# 실행: python chat.py        (끝내려면 '끝' 입력)
import torch
from model import Config, TinyGPT
from tokenizer import CharTokenizer
from persona import NAME, USER

TEMPERATURE = 0.7   # 낮을수록 얌전, 높을수록 엉뚱

tok = CharTokenizer.load("vocab.json")
ckpt = torch.load("model.pt", map_location="cpu")
model = TinyGPT(Config(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.eval()
newline = tok.stoi["\n"]


def reply(history, user_text):
    prompt = history + f"{USER}: {user_text}\n{NAME}: "
    ids = torch.tensor([tok.encode(prompt)])
    out = model.generate(ids, max_new_tokens=120, temperature=TEMPERATURE, stop_id=newline)
    answer = tok.decode(out[0, ids.shape[1]:].tolist()).strip()
    return answer, prompt + answer + "\n"


if __name__ == "__main__":
    print(f"{NAME}와 대화를 시작해요. ('끝' 입력 시 종료)\n")
    history = ""
    while True:
        user_text = input(f"{USER}: ").strip()
        if user_text in ("끝", "quit", "exit"):
            break
        answer, history = reply(history, user_text)
        history = history[-300:]  # 최근 대화만 기억
        print(f"{NAME}: {answer}\n")
