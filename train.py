# 학습: "다음 글자 맞히기"를 수천 번 반복하면서 모델 숫자들을 조금씩 고친다.
# 실행: python train.py
import time
import torch
from model import Config, TinyGPT
from tokenizer import CharTokenizer

# ---- 설정 (컴퓨터가 좋으면 키워도 됨) ----
MAX_ITERS = 2000
BATCH = 32
LR = 1e-3
EVAL_EVERY = 250
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

text = open("data.txt", encoding="utf-8").read()
tok = CharTokenizer(text)
tok.save("vocab.json")
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.95 * len(data))
train_data, val_data = data[:n], data[n:]

cfg = Config(vocab_size=tok.vocab_size)
model = TinyGPT(cfg).to(device)
print(f"장치: {device} | 글자 {tok.vocab_size}개 | 파라미터 {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (BATCH,))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])  # 한 칸 밀린 게 정답
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split))[1].item() for _ in range(20)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_ITERS, eta_min=LR / 10)
t0 = time.time()
for it in range(MAX_ITERS + 1):
    if it % EVAL_EVERY == 0:
        l = estimate()
        print(f"{it:5d}회 | 학습 오차 {l['train']:.3f} | 검증 오차 {l['val']:.3f} | {time.time()-t0:.0f}초", flush=True)
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()                                   # 각 숫자를 어느 쪽으로 고칠지 계산
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()                                        # 실제로 고치기
    sched.step()

torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, "model.pt")
print("저장 완료: model.pt")
