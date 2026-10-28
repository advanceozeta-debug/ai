# ============================================================
#  fix_frames.py — 원본 시트의 빈 곳을 보정 (원본 해상도에서 작업)
# ------------------------------------------------------------
#  1) 검 보정: 시트의 달리기 1~7번, 공격 4번에는 검이 그려져 있지 않음
#     (달리기 8번에만 있음). 반복 재생 중 검이 깜빡이지 않도록,
#     달리기 8번의 검(칼날+손잡이 아래)을 떼어 각 프레임의 앞손 위치에 붙인다.
#  2) 쓰러짐 2·3번: 시트의 누운 자세는 가로 166·239픽셀이라 64칸에 들어가지 않음.
#     찌그러뜨리지 않고, 쓰러짐 1번(무릎 꿇고 손 짚음)의 상체를 허리 기준으로
#     앞으로 숙여 "무너지는" 컴팩트 자세 2장을 새로 만든다.
#  출력: work/fixed/*.png + work/fixed/fix_report.json
# ============================================================
import json
import os
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.join(os.path.dirname(__file__), "..", "work")
RAW, OUT = os.path.join(ROOT, "raw"), os.path.join(ROOT, "fixed")
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(RAW):
    if f.endswith(".png"):
        shutil.copy(os.path.join(RAW, f), os.path.join(OUT, f))
report = {}


def load(name):
    return np.array(Image.open(os.path.join(RAW, name + ".png"))).copy()


def front_fist(im):
    """몸 가운데 높이(45~70%)에서 가장 오른쪽으로 나온 점 = 앞손"""
    a = im[..., 3] > 0
    h = a.shape[0]
    band = a[int(h * 0.45):int(h * 0.70)]
    xs = np.where(band.any(0))[0]
    xr = xs.max()
    ys = np.where(band[:, xr - 3:xr + 1].any(1))[0]
    return xr, int(h * 0.45) + int(np.median(ys))


# ---------- 1. 검 떼어내기 (달리기 8번) ----------
src = load("run_7")
a = src[..., 3] > 0
r, g, b = [src[..., i].astype(int) for i in range(3)]
purple = a & (b > g + 35) & (r > g + 15)
yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
sword = a & (xx >= 58) & (yy >= 76)                     # 몸 오른쪽 아래 = 칼날만 있는 곳
sword |= purple & (xx >= 55) & (yy >= 72)
sword = ndimage.binary_dilation(sword, iterations=1) & a & (xx >= 55) & (yy >= 72)
grip = (60, 74)                                          # 칼자루를 쥔 손 위치
nosword = src.copy(); nosword[sword] = 0
fx7, fy7 = front_fist(nosword)                             # 검을 뺀 상태에서 앞손 찾기
sy, sx = np.where(sword)
layer = src[sy, sx]
report["sword_source"] = {"frame": "run_7", "pixels": int(sword.sum()), "grip": grip, "front_fist": [int(fx7), int(fy7)]}


def paste_sword(name):
    im = load(name)
    fx, fy = front_fist(im)
    dx, dy = fx - fx7, fy - fy7                          # 앞손이 움직인 만큼 검도 옮긴다
    tx, ty = sx + dx, sy + dy
    # 캔버스가 모자라면 오른쪽/아래로 늘림 (검이 잘리지 않게)
    pad_r = max(0, tx.max() + 1 - im.shape[1]); pad_b = max(0, ty.max() + 1 - im.shape[0])
    if pad_r or pad_b:
        im = np.pad(im, ((0, pad_b), (0, pad_r), (0, 0)))
    ok = (tx >= 0) & (ty >= 0)
    tgt = im[ty[ok], tx[ok]]
    # 손(장갑) 위에는 덮지 않고, 몸 뒤로 지나가는 부분만 칼날을 그림: 비어 있는 곳 + 칼날 색
    put = (tgt[:, 3] == 0) | purple[sy[ok], sx[ok]]
    im[ty[ok][put], tx[ok][put]] = layer[ok][put]
    Image.fromarray(im).save(os.path.join(OUT, name + ".png"))
    report.setdefault("sword_pasted", {})[name] = {"front_fist": [int(fx), int(fy)], "shift": [int(dx), int(dy)]}


for k in range(7):
    paste_sword(f"run_{k}")
paste_sword("attack_3")


# ---------- 2. 쓰러짐 컴팩트 자세 ----------
def rotate_part(im, mask, pivot, deg, drop):
    """mask 부분만 pivot 기준으로 deg도 회전(가장 가까운 픽셀, 색 섞지 않음) 후 drop 만큼 내림"""
    h, w = mask.shape
    P = 60
    big = np.zeros((h + 2 * P, w + 2 * P, 4), np.uint8)
    part = im.copy(); part[~mask] = 0
    rest = im.copy(); rest[mask] = 0
    big_part = np.pad(part, ((P, P), (P, P), (0, 0)))
    big_rest = np.pad(rest, ((P, P), (P, P), (0, 0)))
    px, py = pivot[0] + P, pivot[1] + P
    t = np.deg2rad(deg)
    Y, X = np.mgrid[0:big.shape[0], 0:big.shape[1]]
    # 목표 좌표 → 원래 좌표 (역회전)
    ox = np.cos(t) * (X - px) + np.sin(t) * (Y - drop - py) + px
    oy = -np.sin(t) * (X - px) + np.cos(t) * (Y - drop - py) + py
    oxi, oyi = np.round(ox).astype(int), np.round(oy).astype(int)
    valid = (oxi >= 0) & (oyi >= 0) & (oxi < big.shape[1]) & (oyi < big.shape[0])
    rot = np.zeros_like(big)
    rot[valid] = big_part[oyi[valid], oxi[valid]]
    out = big_rest.copy()
    on = rot[..., 3] > 0
    out[on] = rot[on]                                   # 숙인 상체가 다리 위로
    a2 = out[..., 3] > 0
    ys, xs = np.where(a2)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


base = load("down_0")
a0 = base[..., 3] > 0
h0, w0 = a0.shape
# 상체 = 허리선 위쪽이면서 바닥을 짚은 팔(오른쪽 아래)은 제외
yy, xx = np.mgrid[0:h0, 0:w0]
waist_y = int(h0 * 0.62)
upper = a0 & (yy < waist_y) & ~((xx > w0 * 0.70) & (yy > h0 * 0.50))
pivot = (int(w0 * 0.42), waist_y)
for k, (deg, drop) in ((1, (22, 6)), (2, (42, 13))):
    im = rotate_part(base, upper, pivot, deg, drop)
    Image.fromarray(im).save(os.path.join(OUT, f"down_{k}.png"))
    report.setdefault("down_compact", {})[f"down_{k}"] = {"from": "down_0", "upper_body_rotate_deg": deg, "drop_px": drop,
                                                          "size": [int(im.shape[1]), int(im.shape[0])],
                                                          "replaces_sheet_size": {1: [166, 63], 2: [239, 60]}[k]}
json.dump(report, open(os.path.join(OUT, "fix_report.json"), "w"), indent=1)
print("보정 완료:", ", ".join(report["sword_pasted"]), "| 쓰러짐", ", ".join(report["down_compact"]))
