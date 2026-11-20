"""Synthetic, deterministic test images (no downloads needed)."""
import numpy as np


def shapes(N=8, q=8):
    top = 2 ** q - 1
    img = np.full((N, N), top // 6, dtype=int)
    a, b = N // 4, N - N // 4
    img[a:b, a:b] = top * 4 // 5
    c, h = N // 2, max(N // 8, 1)
    img[c - h:c + h, c - h:c + h] = top // 2
    return img


def gradient_circle(N=8, q=8):
    top = 2 ** q - 1
    yy, xx = np.mgrid[:N, :N]
    img = (xx / (N - 1) * top * 0.4).astype(int)
    r = np.hypot(yy - (N - 1) / 2, xx - (N - 1) / 2)
    img[r < N / 3] = top * 3 // 4
    return img


def checker(N=8, q=8, block=2):
    yy, xx = np.mgrid[:N, :N]
    return (((yy // block + xx // block) % 2) * (2 ** q - 1)).astype(int)


def color_shapes(N=4, q=8):
    top = 2 ** q - 1
    rgb = np.zeros((N, N, 3), dtype=int)
    h = N // 2
    rgb[:h, :h] = [top, 0, 0]
    rgb[:h, h:] = [0, top, 0]
    rgb[h:, :h] = [0, 0, top]
    rgb[h:, h:] = [top, top, top // 2]
    return rgb


def requantize(img, q_from=8, q_to=4):
    return (np.asarray(img) >> (q_from - q_to)).astype(int)
