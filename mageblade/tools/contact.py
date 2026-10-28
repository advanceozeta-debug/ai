# 확인용 모음 이미지: 체크무늬(투명 확인) 위에 프레임을 나란히, 배율 zoom
import sys, glob, os
from PIL import Image, ImageDraw

def checker(w, h, s=4):
    im = Image.new("RGBA", (w, h), (200, 60, 200, 255))      # 진분홍 = 투명 영역이 잘 보이게
    d = ImageDraw.Draw(im)
    for y in range(0, h, s):
        for x in range(0, w, s):
            if (x // s + y // s) % 2: d.rectangle([x, y, x + s - 1, y + s - 1], fill=(120, 230, 120, 255))
    return im

def contact(files, out, zoom=2, pad=6):
    ims = [Image.open(f).convert("RGBA") for f in files]
    W = sum(i.width for i in ims) * zoom + pad * (len(ims) + 1)
    H = max(i.height for i in ims) * zoom + pad * 2
    sheet = Image.new("RGBA", (W, H), (30, 30, 30, 255))
    x = pad
    for im in ims:
        big = im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)
        bg = checker(big.width, big.height, 4 * zoom)
        bg.alpha_composite(big)
        sheet.paste(bg, (x, pad)); x += big.width + pad
    sheet.save(out)

if __name__ == "__main__":
    contact(sorted(glob.glob(sys.argv[1]), key=lambda f: (os.path.basename(f).rsplit("_", 1)[0], int(f.rsplit("_", 1)[1].split(".")[0]))), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
