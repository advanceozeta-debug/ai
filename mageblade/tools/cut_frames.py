# ============================================================
#  cut_frames.py — 컨셉 시트에서 동작 그림만 하나씩 분리하고 배경 제거
# ------------------------------------------------------------
#  - 시트 전체를 등분하지 않는다. 동작 영역(제목·초상·장비·이펙트·큰 그림 제외)
#    안에서 캐릭터 덩어리를 찾아 하나씩 잘라낸다.
#  - 배경 제거는 "바깥에서부터 번지는 채우기"로 한다. 남색 배경과 거의 같은 색만
#    지우고, 검은 외곽선(배경보다 어둡고 덜 파람)에서 멈추므로 외곽선·어두운 옷 안쪽이
#    뚫리지 않는다. 다리 사이처럼 갇힌 배경 구멍은 더 엄격한 기준으로만 지운다.
#  출력: work/raw/<동작>_<번호>.png  (원본 해상도, 투명 배경)
# ============================================================
import json
import os
import sys
from collections import deque

import numpy as np
from PIL import Image
from scipy import ndimage

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/attachments/9d22d578-c081-568f-8498-4fb3e04e6318/00270500-image.png"
OUT = os.path.join(os.path.dirname(__file__), "..", "work", "raw")
os.makedirs(OUT, exist_ok=True)

A = np.array(Image.open(SRC).convert("RGB")).astype(int)
BG = np.median(A[1000:1020, 300:900].reshape(-1, 3), axis=0)   # 아래쪽 빈 배경에서 배경색 측정
diff = np.abs(A - BG).max(2)

# 동작 영역 (x0, y0, x1, y1): 라벨 글자 아래, 이펙트 상자 왼쪽
REGIONS = {
    "idle":   ((15, 450, 420, 592), 5),
    "run":    ((420, 455, 1005, 592), 8),
    "attack": ((15, 638, 510, 798), 4),
    "skill":  ((510, 642, 915, 798), 3),
    "hit":    ((15, 845, 280, 988), 2),
    "down":   ((305, 860, 955, 988), 3),
}
FILL_TOL = 9      # 바깥과 이어진 배경: 배경색과 채널 차이 9 이하
HOLE_TOL = 5      # 갇힌 구멍: 차이 5 이하이고 80픽셀 이상(다리 사이처럼 확실한 빈틈)일 때만


def remove_bg(crop_diff):
    h, w = crop_diff.shape
    bgm = np.zeros((h, w), bool)
    cand = crop_diff <= FILL_TOL
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if cand[y, x] and not bgm[y, x]:
                bgm[y, x] = True; q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if cand[y, x] and not bgm[y, x]:
                bgm[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and cand[yy, xx] and not bgm[yy, xx]:
                bgm[yy, xx] = True; q.append((yy, xx))
    holes = (crop_diff <= HOLE_TOL) & ~bgm
    lab, n = ndimage.label(holes)
    sizes = ndimage.sum(holes, lab, range(1, n + 1))
    for i, s in enumerate(sizes):
        if s >= 80:
            bgm[lab == i + 1] = True
    return ~bgm


report = {}
for name, ((x0, y0, x1, y1), count) in REGIONS.items():
    region_fg = remove_bg(diff[y0:y1, x0:x1])
    # 작은 점(노이즈) 제거 후 캐릭터 덩어리 찾기. 머리카락 끝이 떨어져 있어도 한 캐릭터로 묶기 위해 살짝 붙여서 찾음
    joined = ndimage.binary_dilation(region_fg, iterations=2)
    lab, n = ndimage.label(joined)
    sizes = ndimage.sum(region_fg, lab, range(1, n + 1))
    masks = [region_fg & (lab == i + 1) for i, s in enumerate(sizes) if s > 800]
    # 붙어 있는 두 캐릭터는 가장 얇은 세로줄에서 나눈다 (가로 폭이 가장 넓은 덩어리부터)
    while len(masks) < count:
        widths = [np.ptp(np.where(m.any(0))[0]) for m in masks]
        m = masks.pop(int(np.argmax(widths)))
        xs = np.where(m.any(0))[0]
        lo, hi = xs[0] + int(0.3 * np.ptp(xs)), xs[0] + int(0.7 * np.ptp(xs))
        cut = lo + int(m.sum(0)[lo:hi].argmin())
        left = m.copy(); left[:, cut:] = False
        masks += [left, m & ~left]
        print(f"  {name}: 붙은 그림을 x={x0 + cut} 에서 나눔 (겹친 픽셀 {int(m[:, cut].sum())})")
    masks.sort(key=lambda m: np.where(m.any(0))[0][0])
    if len(masks) != count:
        raise SystemExit(f"{name}: 그림 {len(masks)}개 발견 (예상 {count})")
    report[name] = []
    for k, m in enumerate(masks):
        ys, xs = np.where(m)
        by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgba = np.zeros((by1 - by0, bx1 - bx0, 4), np.uint8)
        rgba[..., :3] = A[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1]
        rgba[..., 3] = np.where(m[by0:by1, bx0:bx1], 255, 0)
        Image.fromarray(rgba).save(os.path.join(OUT, f"{name}_{k}.png"))
        report[name].append({"src_box": [int(x0 + bx0), int(y0 + by0), int(x0 + bx1), int(y0 + by1)],
                             "size": [int(bx1 - bx0), int(by1 - by0)], "pixels": int(m.sum())})
json.dump({"background": BG.tolist(), "frames": report}, open(os.path.join(OUT, "cut_report.json"), "w"), indent=1)
for k, v in report.items():
    print(k, [f["size"] for f in v])
