# 글자 단위 토크나이저: 글자 하나 = 숫자 하나.
# 모델은 글자를 못 읽으니까, 글자 <-> 숫자 사전을 직접 만든다.
import json

UNK = "□"  # 배운 적 없는 글자는 □ 로 처리


class CharTokenizer:
    def __init__(self, chars):
        self.chars = [UNK] + sorted(set(chars) - {UNK})
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            chars = json.load(f)
        tok = cls.__new__(cls)
        tok.chars = chars
        tok.stoi = {c: i for i, c in enumerate(chars)}
        return tok
