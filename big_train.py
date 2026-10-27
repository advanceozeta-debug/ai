# ============================================================
#  big_train.py — 큰 오미크론 학습 (GPU용, CPU에서도 돌아가지만 아주 느림)
# ------------------------------------------------------------
#  1단계 한국어:   python big_train.py --data data/pre --iters 5000 --out base_big.pt
#  2단계 대화성격: python big_train.py --data data/dlg --mix data/pre --mix-ratio 0.3 \
#                    --init base_big.pt --iters 800 --lr 3e-4 --dropout 0.1 --out omicron_big.pt
# ============================================================
import argparse
import math
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from model import Config, TinyGPT

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="data/pre 처럼 _train.bin 앞부분")
ap.add_argument("--mix", help="함께 섞어 읽을 데이터 (대화 학습 중 한국어를 잊지 않게)")
ap.add_argument("--mix-ratio", type=float, default=0.0)
ap.add_argument("--tokenizer", default="data/tokenizer.json")
ap.add_argument("--init", help="이어서 학습할 모델")
ap.add_argument("--out", default="omicron_big.pt")
ap.add_argument("--iters", type=int, default=5000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dropout", type=float, default=0.0)
# 두뇌 크기 (처음 만들 때만 쓰임)
ap.add_argument("--n-embd", type=int, default=384)
ap.add_argument("--n-layer", type=int, default=8)
ap.add_argument("--n-head", type=int, default=6)
ap.add_argument("--block", type=int, default=256, help="한 번에 볼 수 있는 조각 수")
ap.add_argument("--eval-every", type=int, default=500)
args = ap.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
tok = Tokenizer.from_file(args.tokenizer)
NL = tok.token_to_id("\n")


def load(prefix):
    return {s: np.memmap(f"{prefix}_{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}


main = load(args.data)
mix = load(args.mix) if args.mix else None

if args.init:
    ckpt = torch.load(args.init, map_location="cpu")
    cfg = Config(**{**ckpt["config"], "dropout": args.dropout})
else:
    cfg = Config(vocab_size=tok.get_vocab_size(), block_size=args.block, n_layer=args.n_layer,
                 n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
model = TinyGPT(cfg).to(device)
if args.init:
    model.load_state_dict(ckpt["model"])
n_params = sum(p.numel() for p in model.parameters())
print(f"장치 {device} | 두뇌 숫자 {n_params/1e6:.1f}M | 조각 사전 {cfg.vocab_size:,} | "
      f"데이터 {len(main['train']):,}조각" + (f" + 섞기 {len(mix['train']):,}" if mix else ""), flush=True)


def batch(split):
    T = cfg.block_size
    n_mix = int(args.batch * args.mix_ratio) if mix else 0
    xs, ys = [], []
    for src, n in ((main, args.batch - n_mix), (mix, n_mix)):
        if not n:
            continue
        d = src[split]
        ix = np.random.randint(0, len(d) - T - 1, n)
        for i in ix:
            chunk = torch.from_numpy(d[i:i + T + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])          # 한 칸 밀린 게 정답
    x, y = torch.stack(xs), torch.stack(ys)
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for s in ("train", "val"):
        ls = []
        for _ in range(20):
            x, y = batch(s)
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                ls.append(model(x, y)[1].item())
        out[s] = sum(ls) / len(ls)
    model.train()
    return out


@torch.no_grad()
def sample(q):
    model.eval()
    ids = torch.tensor([tok.encode(f"나: {q}\n오미크론: ").ids], device=device)
    out = model.generate(ids, 60, temperature=0.8, top_k=40, stop_id=NL)
    model.train()
    return "".join(tok.id_to_token(i) for i in out[0, ids.shape[1]:].tolist()).replace("▁", " ").strip()


def save():
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, args.out)


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler(enabled=use_amp)
warmup = min(200, args.iters // 10)


def lr_at(i):  # 처음엔 살살, 중간엔 세게, 끝으로 갈수록 다시 살살
    if i < warmup:
        return args.lr * (i + 1) / warmup
    p = (i - warmup) / max(1, args.iters - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


t0 = time.time()
for it in range(args.iters + 1):
    if it % args.eval_every == 0 or it == args.iters:
        l = evaluate()
        el = time.time() - t0
        left = el / max(it, 1) * (args.iters - it)
        print(f"{it:6d}회 | 학습 오차 {l['train']:.3f} | 검증 오차 {l['val']:.3f} | "
              f"{el/60:.1f}분 지남, 약 {left/60:.0f}분 남음", flush=True)
        for q in ("안녕", "오늘 너무 피곤해", "주말에 뭐 할까"):
            print(f"    나: {q} → 오미크론: {sample(q)}", flush=True)
        if it:
            save()
    if it == args.iters:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    x, y = batch("train")
    with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

save()
print(f"저장 완료: {args.out}", flush=True)
