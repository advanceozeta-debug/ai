# 학습된 모델을 웹페이지에서 쓸 수 있게 숫자 묶음(weights.json)으로 내보낸다.
# 실행: python export_web.py
import base64
import json
import numpy as np
import torch

ckpt = torch.load("model.pt", map_location="cpu")
vocab = json.load(open("vocab.json", encoding="utf-8"))

parts, manifest, offset = [], {}, 0
for name, t in ckpt["model"].items():
    if name.endswith("attn.mask") or name == "head.weight":
        continue  # 마스크는 계산으로 대신하고, head는 tok_emb와 같은 숫자라 생략
    a = t.float().numpy().ravel().astype("<f4")
    manifest[name] = [offset, list(t.shape)]
    parts.append(a)
    offset += a.size

blob = np.concatenate(parts).tobytes()
out = {"config": ckpt["config"], "vocab": vocab, "manifest": manifest,
       "weights": base64.b64encode(blob).decode()}
json.dump(out, open("weights.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"weights.json: 숫자 {offset:,}개")
