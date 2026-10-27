# 학습: "다음 글자 맞히기"를 수천 번 반복하면서 모델 숫자들을 조금씩 고친다.
#
#  성격만 배우기 (빠름):      python train.py --data data.txt --iters 2000
#  한국어부터 배우기 (2단계):
#    1단계 한국어 기초:  python train.py --data pretrain.txt --iters 3000 --out base.pt --vocab-from pretrain.txt dialog.txt
#    2단계 대화+성격:    python train.py --data dialog.txt --init base.pt --iters 1200 --lr 5e-4
import argparse
import time
import torch
from model import Config, TinyGPT
from tokenizer import CharTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data.txt")
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--init", help="이어서 학습할 모델 파일")
ap.add_argument("--out", default="model.pt")
ap.add_argument("--vocab-from", nargs="+", help="글자 사전을 만들 파일들 (기본: --data)")
args = ap.parse_args()

EVAL_EVERY = 250
torch.manual_seed(1337)
torch.set_num_threads(max(1, torch.get_num_threads()))
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

text = open(args.data, encoding="utf-8").read()
if args.init:                                   # 이어서 학습하면 사전도 그대로 써야 함
    tok = CharTokenizer.load("vocab.json")
    ckpt = torch.load(args.init, map_location="cpu")
    cfg = Config(**ckpt["config"])
else:
    src = "".join(open(f, encoding="utf-8").read() for f in (args.vocab_from or [args.data]))
    tok = CharTokenizer(src, min_count=3)
    tok.save("vocab.json")
    cfg = Config(vocab_size=tok.vocab_size)

model = TinyGPT(cfg).to(device)
if args.init:
    model.load_state_dict(ckpt["model"])
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.97 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"장치: {device} | 글자 {tok.vocab_size}개 | 파라미터 {sum(p.numel() for p in model.parameters())/1e6:.2f}M | 데이터 {len(data):,}글자", flush=True)


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (args.batch,))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])  # 한 칸 밀린 게 정답
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split))[1].item() for _ in range(10)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def save():
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, args.out)


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
warmup = min(100, args.iters // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / warmup) * max(0.1, 1 - i / args.iters))
t0 = time.time()
for it in range(args.iters + 1):
    if it % EVAL_EVERY == 0:
        l = estimate()
        print(f"{it:5d}회 | 학습 오차 {l['train']:.3f} | 검증 오차 {l['val']:.3f} | {time.time()-t0:.0f}초", flush=True)
        if it:
            save()                                    # 중간 저장 (끊겨도 이어갈 수 있게)
    if it == args.iters:
        break
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()                                   # 각 숫자를 어느 쪽으로 고칠지 계산
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()                                        # 실제로 고치기
    sched.step()

save()
print(f"저장 완료: {args.out}", flush=True)
