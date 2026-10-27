# 큰 오미크론과 대화하기 (Colab 또는 내 컴퓨터)
# 실행: python big_chat.py --model omicron_big.pt      ('끝' 입력 시 종료)
import argparse
import torch
from tokenizers import Tokenizer
from model import Config, TinyGPT

NAME, USER = "오미크론", "나"


class Omicron:
    def __init__(self, model_path="omicron_big.pt", tokenizer_path="data/tokenizer.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = Tokenizer.from_file(tokenizer_path)
        ck = torch.load(model_path, map_location="cpu")
        self.model = TinyGPT(Config(**ck["config"])).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.nl = self.tok.token_to_id("\n")
        self.history = ""

    def decode(self, ids):
        return "".join(self.tok.id_to_token(i) for i in ids).replace("▁", " ")

    @torch.no_grad()
    def reply(self, text, temperature=0.8, top_k=40):
        prompt = self.history + f"{USER}: {text}\n{NAME}: "
        ids = self.tok.encode(prompt).ids[-(self.model.cfg.block_size - 64):]
        x = torch.tensor([ids], device=self.device)
        out = self.model.generate(x, 60, temperature=temperature, top_k=top_k, stop_id=self.nl)
        answer = self.decode(out[0, x.shape[1]:].tolist()).strip()
        self.history = (prompt + answer + "\n")[-400:]
        return answer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="omicron_big.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    a = ap.parse_args()
    bot = Omicron(a.model, a.tokenizer)
    print(f"{NAME}와 대화를 시작해요. ('끝' 입력 시 종료)\n")
    while (q := input(f"{USER}: ").strip()) not in ("끝", "quit", "exit"):
        print(f"{NAME}: {bot.reply(q)}\n")
