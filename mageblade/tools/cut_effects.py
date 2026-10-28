# ============================================================
#  cut_effects.py — 컨셉 시트 '스킬 이펙트' 상자 3개에서 이펙트 그림만 분리
# ------------------------------------------------------------
#  상자 테두리와 아래 글자(투사체·장판·폭발)는 빼고, 배경을 지운 뒤
#  캐릭터와 같은 배율(0.49)로 줄여 픽셀아트로 정리한다.
#  시트에는 이펙트마다 그림이 1장뿐이라, 게임용 짧은 애니메이션 프레임은
#  그 1장을 픽셀 단위로 키우기·흩어지기·뒤집기 해서 만든다 (색을 섞지 않음).
#  출력: assets/mageblade/effects/*.png + effects.json
# ============================================================
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from cut_frames import remove_bg, A, BG  # noqa: E402  (같은 배경 제거 규칙 사용)
from build_sheet import downscale, to_palette, SCALE  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "assets", "mageblade", "effects")
os.makedirs(OUT, exist_ok=True)
diff = np.abs(A - BG).max(2)

# (상자 안쪽 x0, y0, x1, y1) — 테두리 4px 안쪽, 아래 글자 줄 위까지
BOXES = {"projectile": (1019, 460, 1192, 558), "field": (1019, 621, 1192, 748), "explosion": (1019, 803, 1192, 930)}
rng = np.random.default_rng(7)
meta = {"scale_from_concept": SCALE, "effects": {}}


def crop_effect(box):
    x0, y0, x1, y1 = box
    fg = remove_bg(diff[y0:y1, x0:x1])
    fg &= ~((np.abs(A[y0:y1, x0:x1] - BG).max(2) < 14))          # 상자 안 배경 얼룩
    lab, n = ndimage.label(ndimage.binary_dilation(fg, iterations=3))
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 12]) & fg
    ys, xs = np.where(keep)
    rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    rgba[..., :3] = A[y0:y1, x0:x1]
    rgba[..., 3] = np.where(keep, 255, 0)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def quantize(sm):
    op = sm[..., 3] > 0
    sm = sm.copy(); sm[op, :3] = to_palette(sm[op, :3])
    return sm


def scale_nn(im, f):
    h, w = im.shape[:2]
    nw, nh = max(1, round(w * f)), max(1, round(h * f))
    return np.array(Image.fromarray(im).resize((nw, nh), Image.NEAREST))


def on_canvas(im, W, H, bottom=False):
    c = np.zeros((H, W, 4), np.uint8)
    h, w = im.shape[:2]
    y, x = (H - h if bottom else (H - h) // 2), (W - w) // 2
    c[y:y + h, x:x + w] = im
    return c


def dissolve(im, keep_frac):
    out = im.copy()
    op = out[..., 3] > 0
    out[op & (rng.random(op.shape) > keep_frac)] = 0
    return out


for name, box in BOXES.items():
    base = quantize(downscale(crop_effect(box)))
    Image.fromarray(base).save(os.path.join(OUT, f"{name}.png"))
    h, w = base.shape[:2]
    W, H = w + (w % 2), h + (h % 2)
    if name == "projectile":                      # 날아가는 동안 반짝임: 원본 / 위아래 뒤집기
        seq, ms, loop = [base, base[::-1]], 70, True
    elif name == "field":                         # 바닥에 펼쳐짐 → 유지
        seq, ms, loop = [scale_nn(base, s) for s in (0.4, 0.7, 1.0)] + [base, dissolve(base, 0.85)], 90, False
    else:                                         # 폭발: 커지고 → 흩어짐
        seq = [scale_nn(base, 0.35), scale_nn(base, 0.65), base, dissolve(base, 0.6), dissolve(base, 0.3)]
        ms, loop = 70, False
    bottom = name == "explosion"                  # 폭발은 바닥에서 솟아오르게 아래 맞춤
    frames = [on_canvas(f, W, H, bottom) for f in seq]
    strip = np.concatenate(frames, axis=1)
    Image.fromarray(strip).save(os.path.join(OUT, f"{name}_strip.png"))
    meta["effects"][name] = {"image": f"effects/{name}_strip.png", "frameW": W, "frameH": H, "frames": len(frames),
                             "frameMs": ms, "loop": loop, "anchor": [W // 2, H - 1 if bottom else H // 2],
                             "source_box": list(box), "note": "시트 이펙트 그림 1장을 바탕으로 픽셀 단위 변형한 프레임"}
    print(name, "원본", base.shape[1], "x", base.shape[0], "| 프레임", len(frames), f"({W}x{H})")
json.dump(meta, open(os.path.join(ROOT, "assets", "mageblade", "effects.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
