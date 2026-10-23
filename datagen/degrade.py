"""Image degradations for robustness testing. Simulated, not real phone photos."""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def scan(img: Image.Image, rng: random.Random) -> Image.Image:
    """Slight skew, softening, downscale and JPEG, like a cheap flatbed or a screenshot of a PDF."""
    out = img.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True, fillcolor="white")
    out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.1)))
    out = out.resize((int(out.width * 0.72), int(out.height * 0.72)), Image.BILINEAR)
    return _jpeg(out, rng.randint(40, 60))


def phone(img: Image.Image, rng: random.Random) -> Image.Image:
    """Perspective tilt, uneven light, a shadow band, sensor noise and heavy JPEG."""
    w, h = img.size
    pad = int(w * 0.06)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (92, 84, 74))  # table top
    canvas.paste(img, (pad, pad))
    W, H = canvas.size
    j = lambda: rng.uniform(-0.035, 0.035) * W  # noqa: E731
    src = [(j(), j()), (W + j(), j()), (W + j(), H + j()), (j(), H + j())]
    dst = [(0, 0), (W, 0), (W, H), (0, H)]
    coeffs = _perspective_coeffs(dst, src)
    out = canvas.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    a = np.asarray(out).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H
    light = 1.05 - 0.35 * (((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2)
    band = rng.uniform(0.25, 0.75) * W
    light *= np.where(np.abs(xx + 0.4 * yy - band) < W * 0.12, rng.uniform(0.7, 0.85), 1.0)
    a = a * light[..., None] * np.array([1.0, 0.97, 0.9])  # warm cast
    a += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 6, a.shape)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    out = out.resize((int(W * 0.6), int(H * 0.6)), Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.7))
    return _jpeg(out, rng.randint(35, 55))


def _perspective_coeffs(src, dst):
    m = []
    for (x, y), (u, v) in zip(src, dst):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(m, dtype=float)
    b = np.array(dst, dtype=float).reshape(8)
    return np.linalg.solve(a, b).tolist()


VARIANTS = {"clean": None, "scan": scan, "phone": phone}
