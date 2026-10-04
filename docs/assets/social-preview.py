#!/usr/bin/env python3
"""Generates docs/assets/social-preview.png (1280x640, GitHub social preview).

    python3 docs/assets/social-preview.py
"""
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 640
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "social-preview.png")

BG_TOP, BG_BOTTOM = (8, 12, 20), (17, 24, 39)
GREEN = (118, 185, 0)          # NVIDIA green
GREEN_DIM = (60, 95, 10)
TEXT = (236, 240, 245)
MUTED = (148, 163, 184)
PANEL = (22, 31, 48)
PANEL_EDGE = (51, 65, 85)
CHIP = (30, 41, 59)

FONTS = "/usr/share/fonts/truetype"
bold = lambda s: ImageFont.truetype(f"{FONTS}/noto/NotoSans-Bold.ttf", s)
regular = lambda s: ImageFont.truetype(f"{FONTS}/noto/NotoSans-Regular.ttf", s)
mono = lambda s: ImageFont.truetype(f"{FONTS}/dejavu/DejaVuSansMono.ttf", s)


def background():
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        d.line([(0, y), (W, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)))
    # faint grid
    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    for x in range(0, W, 40):
        g.line([(x, 0), (x, H)], fill=(255, 255, 255, 8))
    for y in range(0, H, 40):
        g.line([(0, y), (W, y)], fill=(255, 255, 255, 8))
    img.paste(grid, (0, 0), grid)
    # green glow behind the diagram
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([760, 120, 1240, 560], fill=(118, 185, 0, 38))
    glow = glow.filter(ImageFilter.GaussianBlur(90))
    img.paste(glow, (0, 0), glow)
    return img


def chip(d, x, y, label, font):
    w = d.textlength(label, font=font) + 28
    d.rounded_rectangle([x, y, x + w, y + 38], radius=19, fill=CHIP, outline=PANEL_EDGE, width=1)
    d.text((x + 14, y + 19), label, font=font, fill=TEXT, anchor="lm")
    return x + w + 10


def diagram(img):
    trays = [(790, 330, 1000, 520), (1010, 330, 1220, 520)]
    sx0, sy0, sx1, sy1 = 790, 92, 1220, 196
    switch_boxes = [[sx0 + 20, sy0 + 46, sx0 + 208, sy0 + 88], [sx0 + 222, sy0 + 46, sx0 + 410, sy0 + 88]]

    # NVLinks first, so boxes sit on top: every GPU (one anchor per GPU on
    # its tray's top edge) to both switches.
    links = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(links)
    for tx0, ty0, tx1, _ in trays:
        for g in range(4):
            ax = tx0 + 30 + g * (tx1 - tx0 - 60) / 3
            for bx0, _, bx1, by1 in switch_boxes:
                ld.line([(ax, ty0), ((bx0 + bx1) / 2, by1)], fill=(118, 185, 0, 80), width=2)
    img.paste(links, (0, 0), links)

    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([sx0, sy0, sx1, sy1], radius=14, fill=PANEL, outline=PANEL_EDGE, width=2)
    d.text((sx0 + 18, sy0 + 14), "NVLink switch tray", font=bold(18), fill=TEXT)
    for i, box in enumerate(switch_boxes):
        d.rounded_rectangle(box, radius=8, fill=(14, 22, 12), outline=GREEN, width=2)
        d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), f"NVSwitch {i} · 72 ports", font=mono(14), fill=GREEN, anchor="mm")

    caption = "8 GPUs × 18 NVLinks → 144 switch ports"
    cw = d.textlength(caption, font=mono(14))
    d.rounded_rectangle([1005 - cw / 2 - 10, 250, 1005 + cw / 2 + 10, 274], radius=6, fill=BG_BOTTOM)
    d.text((1005, 262), caption, font=mono(14), fill=MUTED, anchor="mm")

    for t, (tx0, ty0, tx1, ty1) in enumerate(trays):
        d.rounded_rectangle([tx0, ty0, tx1, ty1], radius=14, fill=PANEL, outline=PANEL_EDGE, width=2)
        d.text((tx0 + 16, ty0 + 12), f"slurm-worker{t + 1}", font=bold(17), fill=TEXT)
        for g in range(4):
            gx = tx0 + 16 + (g % 2) * 92
            gy = ty0 + 50 + (g // 2) * 64
            d.rounded_rectangle([gx, gy, gx + 84, gy + 52], radius=7, fill=(14, 22, 12), outline=GREEN, width=2)
            d.text((gx + 42, gy + 26), "GB200", font=bold(15), fill=GREEN, anchor="mm")
        # the tray's BMC, centred below it
        cx = (tx0 + tx1) / 2
        d.line([(cx, ty1), (cx, ty1 + 12)], fill=PANEL_EDGE, width=2)
        d.rounded_rectangle([cx - 62, ty1 + 12, cx + 62, ty1 + 40], radius=8, fill=CHIP, outline=PANEL_EDGE, width=1)
        d.text((cx, ty1 + 26), "BMC · Redfish", font=regular(13), fill=MUTED, anchor="mm")


def main():
    img = background()
    diagram(img)
    d = ImageDraw.Draw(img)

    x = 64
    d.text((x, 92), "slurm-lab", font=bold(88), fill=TEXT)
    d.rectangle([x, 206, x + 120, 212], fill=GREEN)
    d.text((x, 236), "An emulated NVIDIA GB200 NVL8", font=bold(36), fill=TEXT)
    d.text((x, 282), "GPU cluster on one Linux machine", font=bold(36), fill=TEXT)
    d.text((x, 340), "Real Slurm, storage and monitoring on fake GPUs:", font=regular(22), fill=MUTED)
    d.text((x, 368), "every call succeeds, takes realistic time, nothing computes.", font=regular(22), fill=MUTED)

    f = regular(18)
    rows = [["Slurm", "CUDA · NVML · NCCL", "NVLink partitions", "Redfish BMCs"],
            ["PyTorch · Ray", "JuiceFS · S3", "Prometheus · Grafana", "Incus · Ansible"]]
    y = 430
    for row in rows:
        cx = x
        for label in row:
            cx = chip(d, cx, y, label, f)
        y += 50

    d.text((x, 574), "$ make up && make test", font=mono(20), fill=GREEN)
    d.text((W - 60, 606), "github.com/arkady-emelyanov/slurm-lab", font=regular(17), fill=MUTED, anchor="rs")
    img.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
