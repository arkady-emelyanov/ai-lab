#!/usr/bin/env python3
"""Generates docs/assets/social-preview.png (1280x640, GitHub social preview)
from the lab overview picture (docs/assets/overview.png, the README's, made
for the blog's part 1): the picture fitted to the height on the same dark
background.

    python3 docs/assets/social-preview.py
"""
import os

from PIL import Image, ImageDraw

W, H = 1280, 640
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "social-preview.png")

# The overview's background: a vertical gradient with a faint 40 px grid.
BG_TOP, BG_BOTTOM = (8, 12, 20), (16, 23, 38)


def background():
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        d.line([(0, y), (W, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)))
    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    for x in range(0, W, 40):
        g.line([(x, 0), (x, H)], fill=(255, 255, 255, 8))
    for y in range(0, H, 40):
        g.line([(0, y), (W, y)], fill=(255, 255, 255, 8))
    img.paste(grid, (0, 0), grid)
    return img


def main():
    img = background()
    overview = Image.open(os.path.join(HERE, "overview.png")).convert("RGB")
    scale = min(W / overview.width, H / overview.height)
    size = (round(overview.width * scale), round(overview.height * scale))
    img.paste(overview.resize(size, Image.LANCZOS), ((W - size[0]) // 2, (H - size[1]) // 2))
    img.save(OUT, optimize=True)


if __name__ == "__main__":
    main()
