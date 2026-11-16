"""Create a reproducible H&E-like *geometry* example (not histology ground truth)."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

rng = np.random.default_rng(12)
w, h = 512, 384
im = Image.new('RGB', (w, h), (246, 213, 224))
d = ImageDraw.Draw(im)
for j in range(38):
    y0 = 8 + j * 10 + rng.uniform(-4, 4)
    pts = [(x, int(y0 + 7 * np.sin(x / 47 + j * .36) + 2 * np.sin(x / 11 + j))) for x in range(-5, w + 10, 3)]
    col = (int(rng.integers(197, 227)), int(rng.integers(105, 158)), int(rng.integers(157, 194)))
    d.line(pts, fill=col, width=int(rng.integers(3, 8)), joint='curve')
for _ in range(110):
    x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
    d.ellipse((x-2, y-4, x+2, y+4), fill=(109, 69, 155))
im = im.filter(ImageFilter.GaussianBlur(.55))
out = Path(__file__).parent / 'synthetic_bundles.png'
im.save(out)
print(out)
