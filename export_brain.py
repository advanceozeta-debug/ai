# ============================================================
#  export_brain.py — 학습된 두뇌를 웹페이지용 brain/ 폴더로 내보내기
# ------------------------------------------------------------
#  brain/model.bin      : 두뇌 숫자들 (큰 행렬은 8비트로 줄여서 용량 1/4)
#  brain/tokenizer.json : 글자/단어 조각 사전
#
#  작은 오미크론(글자 단위): python export_brain.py --model model.pt --tokenizer vocab.json
#  큰 오미크론(단어 조각):   python export_brain.py --model omicron_big.pt --tokenizer data/tokenizer.json
# ============================================================
import argparse
import json
import os
import struct

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model.pt")
ap.add_argument("--tokenizer", default="vocab.json")
ap.add_argument("--out", default="brain")
ap.add_argument("--name", default="오미크론")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---------- 사전 ----------
raw = json.load(open(args.tokenizer, encoding="utf-8"))
if isinstance(raw, list):                       # 글자 단위 사전 (vocab.json)
    tok_out = {"type": "char", "vocab": raw}
else:                                           # 단어 조각 사전 (tokenizers 라이브러리 형식)
    m = raw["model"]
    vocab = [None] * len(m["vocab"])
    for t, i in m["vocab"].items():
        vocab[i] = t
    merges = [x.split(" ") if isinstance(x, str) else x for x in m["merges"]]
    tok_out = {"type": "bpe", "vocab": vocab, "merges": merges, "unk": m["vocab"].get("<unk>", 0)}
json.dump(tok_out, open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)

# ---------- 두뇌 숫자 ----------
ckpt = torch.load(args.model, map_location="cpu")
header = {"name": args.name, "config": ckpt["config"], "tensors": {}}
chunks, offset = [], 0
for name, t in ckpt["model"].items():
    if name.endswith("attn.mask") or name == "head.weight":
        continue                                # 마스크는 계산으로, head 는 tok_emb 와 같은 숫자
    a = t.float().numpy()
    if a.ndim == 2 and a.size >= 4096:
        # 8비트 압축: 행마다 가장 큰 값을 127로 맞추고 정수로 반올림
        scale = np.abs(a).max(axis=1) / 127.0
        scale[scale == 0] = 1e-8
        q = np.clip(np.round(a / scale[:, None]), -127, 127).astype(np.int8)
        header["tensors"][name] = {"dtype": "q8", "shape": list(a.shape), "offset": offset}
        b = scale.astype("<f4").tobytes() + q.tobytes()
    else:
        header["tensors"][name] = {"dtype": "f32", "shape": list(a.shape), "offset": offset}
        b = a.astype("<f4").tobytes()
    b += b"\0" * (-len(b) % 4)                  # 4바이트 단위로 맞춤
    chunks.append(b)
    offset += len(b)

hj = json.dumps(header, ensure_ascii=False).encode("utf-8")
hj += b" " * (-(len(hj) + 8) % 4)
with open(os.path.join(args.out, "model.bin"), "wb") as f:
    f.write(b"OMI1" + struct.pack("<I", len(hj)) + hj)
    for b in chunks:
        f.write(b)

n = sum(t.numel() for k, t in ckpt["model"].items() if not k.endswith("attn.mask") and k != "head.weight")
size = os.path.getsize(os.path.join(args.out, "model.bin")) / 1e6
print(f"{args.out}/model.bin {size:.1f}MB (두뇌 숫자 {n:,}개) | {args.out}/tokenizer.json ({tok_out['type']}, {len(tok_out['vocab']):,}개)")
