"""Оценка на парах с известной гомографией: синтез пар, повторяемость,
точность/полнота сопоставления, доля inlier'ов."""
import numpy as np
import cv2

from .geometry import project


def synthetic_homography(w, h, angle=0.0, scale=1.0, persp=(0.0, 0.0), shift=(0.0, 0.0)):
    """Гомография: поворот на angle° и масштаб scale вокруг центра + перспективный
    «наклон» persp=(p1, p2) (в долях 1/размер) + сдвиг. Отображает img1 → img2."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    A = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    Hs = np.vstack([A, [0, 0, 1]])
    P = np.array([[1, 0, 0], [0, 1, 0], [persp[0] / w, persp[1] / h, 1.0]])
    C = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    Hp = np.linalg.inv(C) @ P @ C                  # перспектива относительно центра
    Hp /= Hp[2, 2]
    T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    H = T @ Hp @ Hs
    return H / H[2, 2]


def make_pair(img, H, noise=0.0, seed=0):
    """Второе изображение пары: warpPerspective + (опц.) гауссов шум σ=noise (в шкале 0..255)."""
    h, w = img.shape[:2]
    img2 = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    if noise > 0:
        rng = np.random.default_rng(seed)
        img2 = np.clip(img2 + rng.normal(0, noise, img2.shape), 0, 255).astype(np.uint8)
    return img2


def inside(p, w, h, margin=0):
    """Маска точек, лежащих внутри кадра w×h (с отступом margin)."""
    return (p[:, 0] >= margin) & (p[:, 0] <= w - 1 - margin) \
        & (p[:, 1] >= margin) & (p[:, 1] <= h - 1 - margin)


def repeatability(p1, p2, H, shape1, shape2, eps=3.0):
    """Повторяемость (упрощённо — только по положениям, без перекрытия областей).

    Берём точки общей видимой области. Точка «повторилась», если после переноса
    гомографией рядом (< eps px) есть точка другого изображения. Считаем в обе
    стороны (1→2 в пикселях img2, 2→1 в пикселях img1) и берём меньшее число,
    чтобы при сильном уменьшении несколько точек не засчитывались за одну.
    Делим на min(числа видимых точек) — как у Mikolajczyk et al."""
    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2]
    q1 = project(H, p1)
    v1 = inside(q1, w2, h2)
    q2 = project(np.linalg.inv(H), p2)
    v2 = inside(q2, w1, h1)
    if v1.sum() == 0 or v2.sum() == 0:
        return 0.0

    def hits(a, b):
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
        return int((np.sqrt(np.maximum(d2.min(1), 0)) < eps).sum())

    n_rep = min(hits(q1[v1], p2[v2]), hits(q2[v2], p1[v1]))
    return float(n_rep / min(v1.sum(), v2.sum()))


def match_correctness(p1, p2, matches, H, thr=3.0):
    """Булев массив: совпадение (i, j) верно, если ‖H·p1[i] − p2[j]‖ < thr."""
    if len(matches) == 0:
        return np.zeros(0, bool)
    q = project(H, p1[matches[:, 0]])
    return np.linalg.norm(q - p2[matches[:, 1]], axis=1) < thr


def n_possible(p1, p2, H, thr=3.0):
    """Число точек img1, для которых в img2 есть истинный партнёр (ближе thr) — знаменатель recall."""
    q = project(H, p1)
    d2 = (q * q).sum(1)[:, None] + (p2 * p2).sum(1)[None] - 2 * q @ p2.T
    return int((np.sqrt(np.maximum(d2.min(1), 0)) < thr).sum())
