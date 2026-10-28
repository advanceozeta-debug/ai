# 동작 움짤: config.js 의 재생 속도와 몸 움직임(dx·dy·step)을 그대로 적용해 6개 동작을 나란히 재생
import json, os, re
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AS = os.path.join(ROOT, "assets", "mageblade")
m = json.load(open(os.path.join(AS, "mageblade_sheet.json"), encoding="utf-8"))
S = Image.open(os.path.join(AS, "mageblade_sheet.png"))
cfg_txt = open(os.path.join(ROOT, "config.js"), encoding="utf-8").read()


def js_obj(key):  # config.js 에서 animations / motion 블록만 읽기 (간단한 파서)
    i = cfg_txt.index(key + ": {")
    depth, j = 0, cfg_txt.index("{", i)
    for k in range(j, len(cfg_txt)):
        depth += cfg_txt[k] == "{"; depth -= cfg_txt[k] == "}"
        if depth == 0:
            body = cfg_txt[j:k + 1]; break
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"(\w+)\s*:", r'"\1":', body)
    body = re.sub(r",(\s*[}\]])", r"\1", body)
    return json.loads(body)


anims, motion = js_obj("animations"), js_obj("motion")
names = ["idle", "run", "attack", "skill", "hit", "down"]
Z, DT, T, CW = 3, 20, 3000, 96
frames = []
for t in range(0, T, DT):
    can = Image.new("RGBA", (CW * Z * 6, 96 * Z + 20), (13, 26, 41, 255)); d = ImageDraw.Draw(can)
    for j, n in enumerate(names):
        a = m["animations"][n]; ms = anims[n]["frameMs"]
        ms = ms if isinstance(ms, list) else [ms] * len(a["frames"])
        total = sum(ms)
        tt = t % total if a["loop"] else t % (total + 900)
        i, acc = 0, 0
        while i < len(ms) - 1 and tt >= acc + ms[i]:
            acc += ms[i]; i += 1
        mo = motion.get(n, {})
        dx = (mo.get("dx") or [0] * 9)[i] + sum((mo.get("step") or [0] * 9)[1:i + 1])
        dy = (mo.get("dy") or [0] * 9)[i]
        f = a["frames"][i]
        cell = S.crop((f["x"], f["y"], f["x"] + 64, f["y"] + 96)).resize((64 * Z, 96 * Z), Image.NEAREST)
        can.alpha_composite(cell, (j * CW * Z + (16 + dx) * Z, dy * Z))
        d.rectangle([j * CW * Z, 92 * Z, (j + 1) * CW * Z - 1, 92 * Z], fill=(60, 70, 100, 255))
        d.text((j * CW * Z + 8, 96 * Z + 4), f"{n} {i + 1}/{len(ms)}", fill=(168, 161, 210, 255))
    frames.append(can.convert("RGB").quantize(160))
frames[0].save(os.path.join(AS, "preview.gif"), save_all=True, append_images=frames[1:], duration=DT, loop=0)
# 확인용: 공격 4프레임 + 달리기 4프레임 (움직임 적용)
print("preview.gif", len(frames), "frames", frames[0].size)
