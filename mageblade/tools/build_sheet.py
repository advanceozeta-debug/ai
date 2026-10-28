# ============================================================
#  build_sheet.py — 보정된 원본 프레임 → 64×96 픽셀 프레임 25장 + 8×6 시트 + JSON
# ------------------------------------------------------------
#  - 모든 동작을 같은 배율(SCALE)로 줄인다 → 머리 크기·신체 비율 일정
#  - 줄인 뒤 반투명 픽셀은 0 또는 255로 (픽셀아트처럼 선명하게), 색은 공용 팔레트로 정리
#  - 기준점 (32, 92): 몸통 중심 x 를 32에, 발바닥(가장 아래 픽셀)을 92 바로 위(91)에 맞춤
#  - 칸 밖으로 나가는 픽셀이 있으면 멈추고 알려 줌 (잘린 채로 저장하지 않음)
# ============================================================
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIXED = os.path.join(ROOT, "work", "fixed")
OUT = os.path.join(ROOT, "assets", "mageblade")
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)

SCALE = 0.49
CELL_W, CELL_H, AX, AY = 64, 96, 32, 92
COLS = 8
ANIMS = [  # 이름, 프레임 수, 프레임당 ms, 반복, 마지막 프레임 유지
    ("idle", 5, 150, True, False),
    ("run", 8, 80, True, False),
    ("attack", 4, 80, False, False),
    ("skill", 3, 110, False, False),
    ("hit", 2, 100, False, False),
    ("down", 3, 170, False, True),
]
SOURCE_NOTE = {
    "run": "시트 달리기 1~8번. 1~7번은 시트에 검이 없어 8번의 검을 앞손 위치에 합성",
    "attack": "시트 공격 1~4번. 4번은 시트에 검이 없어 달리기 8번의 검을 합성",
    "down": "1번은 시트 쓰러짐 1번. 2·3번은 시트의 누운 자세(가로 166·239px)가 칸에 안 들어가 1번의 상체를 앞으로 숙여 새로 만든 컴팩트 자세",
}


def clean_islands(a):
    """본체에서 4픽셀 넘게 떨어진 조각(옆 그림 부스러기)은 버림"""
    lab, n = ndimage.label(a, structure=np.ones((3, 3)))
    if n <= 1:
        return a
    sizes = ndimage.sum(a, lab, range(1, n + 1))
    main = lab == (int(np.argmax(sizes)) + 1)
    near = ndimage.binary_dilation(main, iterations=4)
    keep = main.copy()
    for i in range(1, n + 1):
        comp = lab == i
        if (comp & near).any():
            keep |= comp
    return keep


def downscale(im):
    """색이 배경과 섞이지 않게 알파를 곱한 상태로 면적 평균 축소 → 알파 0/255"""
    h, w = im.shape[:2]
    nw, nh = max(1, round(w * SCALE)), max(1, round(h * SCALE))
    f = im.astype(np.float32) / 255
    pre = f.copy(); pre[..., :3] *= f[..., 3:4]
    small = np.array(Image.fromarray((pre * 255).astype(np.uint8)).resize((nw, nh), Image.BOX)).astype(np.float32) / 255
    al = small[..., 3]
    rgb = np.where(al[..., None] > 0, small[..., :3] / np.maximum(al[..., None], 1e-6), 0)
    out = np.zeros((nh, nw, 4), np.uint8)
    solid = al >= 0.5
    out[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(solid, 255, 0)
    return out


def anchor_of(im):
    """몸통 중심 x (가는 머리카락·검·옷자락을 깎아 낸 덩어리의 가운데), 발바닥 y"""
    a = im[..., 3] > 0
    r, g, b = [im[..., i].astype(int) for i in range(3)]
    purple = (b > g + 35) & (r > g + 15)
    core = ndimage.binary_erosion(a & ~purple, iterations=7)
    xs = np.where(core)[1]
    ax = xs.mean() if len(xs) else a.shape[1] / 2
    foot = np.where(a.any(1))[0].max()
    return ax, foot


# ---------- 1. 원본 프레임 읽고 줄이기 ----------
frames = {}
for name, n, *_ in ANIMS:
    frames[name] = []
    for k in range(n):
        im = np.array(Image.open(os.path.join(FIXED, f"{name}_{k}.png")).convert("RGBA"))
        im[~clean_islands(im[..., 3] > 0)] = 0
        ax, foot = anchor_of(im)
        frames[name].append({"small": downscale(im), "ax": ax * SCALE, "foot": foot})

# ---------- 2. 공용 팔레트 (모든 프레임에서 48색) ----------
# 컨셉 시트 '색상 팔레트' 10색 (남색·파랑·흰색·연보라·회색·검정·갈색·금색·보라 2색)
SHEET_PALETTE = np.array([(43, 56, 133), (58, 83, 188), (240, 233, 247), (168, 161, 210), (74, 68, 90),
                          (37, 35, 49), (145, 103, 76), (228, 179, 133), (185, 140, 250), (118, 95, 219)])
pix = np.concatenate([f["small"][f["small"][..., 3] > 0][:, :3] for v in frames.values() for f in v])
pr, pg, pb = [pix[:, i].astype(int) for i in range(3)]
is_purple = (pb > pg + 30) & (pr > pg + 10)
pix = np.concatenate([pix] + [pix[is_purple]] * 8)       # 검(보라)이 적어서 묻히지 않게 비중을 높임
pal_img = Image.fromarray(pix.reshape(-1, 1, 3)).quantize(54, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
palette = np.concatenate([np.array(pal_img.getpalette()[:54 * 3]).reshape(-1, 3), SHEET_PALETTE])


def to_palette(rgb):
    d = ((rgb[:, None, :].astype(int) - palette[None]) ** 2).sum(2)
    return palette[d.argmin(1)].astype(np.uint8)


# ---------- 3. 64×96 칸에 기준점 맞춰 배치 ----------
sheet = Image.new("RGBA", (CELL_W * COLS, CELL_H * len(ANIMS)), (0, 0, 0, 0))
meta = {"image": "mageblade_sheet.png", "size": [CELL_W * COLS, CELL_H * len(ANIMS)],
        "cell": [CELL_W, CELL_H], "anchor": [AX, AY], "scale_from_concept": SCALE, "animations": {}}
problems, checks = [], []
for row, (name, n, ms, loop, hold) in enumerate(ANIMS):
    anim = {"row": row, "frames": [], "frameMs": [ms] * n, "loop": loop, "holdLast": hold}
    if name in SOURCE_NOTE:
        anim["note"] = SOURCE_NOTE[name]
    for k, f in enumerate(frames[name]):
        sm = f["small"].copy()
        op = sm[..., 3] > 0
        sm[op, :3] = to_palette(sm[op, :3])
        h, w = sm.shape[:2]
        foot_small = np.where(op.any(1))[0].max()
        ox = int(round(AX - f["ax"]))            # 몸통 중심을 x=32에
        oy = (AY - 1) - foot_small               # 발바닥 픽셀을 y=91에
        x0, y0, x1, y1 = ox, oy, ox + w, oy + h
        ys, xs = np.where(op)
        L, T, R, B = ox + xs.min(), oy + ys.min(), ox + xs.max(), oy + ys.max()
        shift = 0
        if L < 0 or R > CELL_W - 1:              # 1~3픽셀 넘치면 가로로만 살짝 옮김, 더 넘치면 오류
            need = -L if L < 0 else (CELL_W - 1) - R
            if abs(need) <= 3 and (L + need) >= 0 and (R + need) <= CELL_W - 1:
                shift = need; ox += need; L += need; R += need
            else:
                problems.append(f"{name}_{k}: 칸 밖으로 나감 L={L} R={R}")
        if T < 0:
            problems.append(f"{name}_{k}: 위로 잘림 T={T}")
        # 줄인 뒤 몸 안쪽에 남은 작은 투명 점(12픽셀 미만)은 주변의 가장 어두운 색으로 메움 → 의상 내부 뚫림 방지
        lab_h, nh = ndimage.label(~op)
        border = set(np.unique(np.concatenate([lab_h[0], lab_h[-1], lab_h[:, 0], lab_h[:, -1]])))
        for hi in range(1, nh + 1):
            if hi in border:
                continue
            hole = lab_h == hi
            if hole.sum() < 12:
                ring = ndimage.binary_dilation(hole) & op
                cols = sm[ring][:, :3]
                sm[hole, :3] = cols[cols.astype(int).sum(1).argmin()]
                sm[hole, 3] = 255
                filled = anim.setdefault("filledHolesPx", 0)
                anim["filledHolesPx"] = filled + int(hole.sum())
        op = sm[..., 3] > 0
        cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        cy0, cx0 = max(0, oy), max(0, ox)
        cell[cy0:oy + h, cx0:ox + w] = sm[cy0 - oy:, cx0 - ox:][:CELL_H - cy0, :CELL_W - cx0]
        clipped = int(op.sum()) - int((cell[..., 3] > 0).sum())   # 칸에 옮기며 잘려 나간 픽셀 수
        if clipped:
            problems.append(f"{name}_{k}: {clipped}픽셀 잘림")
        Image.fromarray(cell).save(os.path.join(OUT, "frames", f"{name}_{k + 1}.png"))
        sheet.paste(Image.fromarray(cell), (k * CELL_W, row * CELL_H))
        anim["frames"].append({"x": k * CELL_W, "y": row * CELL_H, "w": CELL_W, "h": CELL_H,
                               "bbox": [int(L), int(T), int(R), int(B)], "anchorShiftX": int(shift)})
        checks.append({"frame": f"{name}_{k + 1}", "bbox": [int(L), int(T), int(R), int(B)], "clippedPx": clipped,
                       "touchesEdge": bool(L == 0 or R == CELL_W - 1 or T == 0), "shiftX": int(shift),
                       "opaque": int(op.sum())})
    meta["animations"][name] = anim
if problems:
    raise SystemExit("규격 오류:\n" + "\n".join(problems))
sheet.save(os.path.join(OUT, "mageblade_sheet.png"))
json.dump(meta, open(os.path.join(OUT, "mageblade_sheet.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(checks, open(os.path.join(ROOT, "work", "sheet_checks.json"), "w"), indent=1)
print("시트", sheet.size, "| 프레임", sum(len(a["frames"]) for a in meta["animations"].values()))
for c in checks:
    if c["touchesEdge"] or c["shiftX"]:
        print("  주의:", c)
