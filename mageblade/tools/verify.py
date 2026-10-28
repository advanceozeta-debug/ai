# ============================================================
#  verify.py — 결과물 검수: 크기, 프레임 수, 빈 칸 투명, 잘림, 바닥선, 배경 제거, 중복 프레임
# ============================================================
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AS = os.path.join(ROOT, "assets", "mageblade")
meta = json.load(open(os.path.join(AS, "mageblade_sheet.json"), encoding="utf-8"))
im = Image.open(os.path.join(AS, "mageblade_sheet.png"))
S = np.array(im)
BG = np.array([13, 26, 41])
rep = {"sheet": {"file": "mageblade_sheet.png", "size": list(im.size), "mode": im.mode}, "rows": {}, "frames": [], "issues": []}
assert im.size == (512, 576) and im.mode == "RGBA", "시트 크기/형식 오류"
expected = {"idle": 5, "run": 8, "attack": 4, "skill": 3, "hit": 2, "down": 3}
cells = {}
for name, a in meta["animations"].items():
    n = len(a["frames"])
    row = a["row"]
    unused_ok = all(S[row * 96:(row + 1) * 96, c * 64:(c + 1) * 64, 3].max() == 0 for c in range(n, 8))
    rep["rows"][name] = {"row": row + 1, "frames": n, "expected": expected[name], "unused_cells_transparent": bool(unused_ok)}
    if n != expected[name]:
        rep["issues"].append(f"{name}: 프레임 {n}개 (요청 {expected[name]})")
    if not unused_ok:
        rep["issues"].append(f"{name}: 안 쓰는 칸에 픽셀 있음")
    for k in range(n):
        cell = S[row * 96:(row + 1) * 96, k * 64:(k + 1) * 64]
        cells[f"{name}_{k + 1}"] = cell
        al = cell[..., 3]
        semi = int(((al > 0) & (al < 255)).sum())
        op = al == 255
        ys, xs = np.where(op)
        L, T, R, B = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        # 배경색이 남은 가장자리 픽셀 (배경과 채널 차이 12 이하)
        rgb = cell[..., :3].astype(int)
        near_bg = int((op & (np.abs(rgb - BG).max(2) <= 12)).sum())
        # 안쪽 구멍: 바깥과 이어지지 않은 투명 영역
        lab, nh = ndimage.label(~op)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        holes = [int((lab == i).sum()) for i in range(1, nh + 1) if i not in border]
        f = {"frame": f"{name}_{k + 1}", "bbox": [L, T, R, B], "feet_row": B, "opaque_px": int(op.sum()),
             "semi_transparent_px": semi, "bg_colored_px": near_bg, "enclosed_holes": holes}
        rep["frames"].append(f)
        if B != 91:
            rep["issues"].append(f"{f['frame']}: 발바닥 줄 {B} (기준 91)")
        if semi:
            rep["issues"].append(f"{f['frame']}: 반투명 픽셀 {semi}개")
        if holes and max(holes) >= 4:
            rep["issues"].append(f"{f['frame']}: 안쪽 투명 구멍 {holes}")
# 중복 프레임 검사
names = list(cells)
dups, sims = [], []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = cells[names[i]], cells[names[j]]
        if np.array_equal(a, b):
            dups.append((names[i], names[j]))
        same_row = names[i].rsplit("_", 1)[0] == names[j].rsplit("_", 1)[0]
        if same_row:
            diff = int(((a[..., 3] > 0) != (b[..., 3] > 0)).sum() + (np.abs(a[..., :3].astype(int) - b[..., :3]).max(2) > 24).sum())
            sims.append((names[i], names[j], diff))
rep["identical_frames"] = dups
rep["least_different_pairs"] = sorted(sims, key=lambda t: t[2])[:5]
if dups:
    rep["issues"].append(f"완전히 같은 프레임: {dups}")
# 개별 프레임 파일 = 시트 칸
for name in names:
    fp = os.path.join(AS, "frames", f"{name}.png")
    fr = np.array(Image.open(fp))
    if fr.shape != (96, 64, 4) or not np.array_equal(fr, cells[name]):
        rep["issues"].append(f"{name}.png 가 시트 칸과 다름")
# 이펙트
fx = json.load(open(os.path.join(AS, "effects.json"), encoding="utf-8"))
rep["effects"] = {}
for k, e in fx["effects"].items():
    st = Image.open(os.path.join(AS, e["image"]))
    a = np.array(st)
    rep["effects"][k] = {"size": list(st.size), "mode": st.mode, "frames": e["frames"],
                         "frame": [e["frameW"], e["frameH"]],
                         "edge_touch": bool(a[0, :, 3].any() or a[-1, :, 3].any()),
                         "semi_transparent_px": int(((a[..., 3] > 0) & (a[..., 3] < 255)).sum())}
json.dump(rep, open(os.path.join(ROOT, "verify_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("시트", rep["sheet"])
for k, v in rep["rows"].items():
    print(f"  {v['row']}행 {k}: {v['frames']}/{v['expected']} 프레임, 빈 칸 투명={v['unused_cells_transparent']}")
print("발바닥 줄:", sorted(set(f["feet_row"] for f in rep["frames"])))
print("반투명 픽셀 합:", sum(f["semi_transparent_px"] for f in rep["frames"]))
print("배경색 남은 픽셀(프레임별 최대):", max(f["bg_colored_px"] for f in rep["frames"]), "| 합", sum(f["bg_colored_px"] for f in rep["frames"]))
print("안쪽 구멍:", {f["frame"]: f["enclosed_holes"] for f in rep["frames"] if f["enclosed_holes"]})
print("가장자리 닿는 프레임:", [f["frame"] for f in rep["frames"] if f["bbox"][0] == 0 or f["bbox"][2] == 63 or f["bbox"][1] == 0])
print("같은 줄에서 가장 비슷한 쌍(차이 픽셀):", rep["least_different_pairs"])
print("이펙트:", rep["effects"])
print("문제:", rep["issues"] or "없음")
