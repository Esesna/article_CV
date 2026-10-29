"""Геометрическая проверка: гомография нормализованным DLT и RANSAC
(см. также ../../rus/RANSAC_конспект.md; в статье SIFT — Хаф + аффинная МНК, разд. 7.3–7.4).
"""
import numpy as np


def project(H, p):
    """Применить гомографию H (3×3) к точкам p (n, 2)."""
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def normalize_points(p):
    """Нормировка Хартли: центр масс в 0, среднее расстояние до центра √2. Возвращает (p', T)."""
    m = p.mean(0)
    sc = np.sqrt(2) / max(np.mean(np.linalg.norm(p - m, axis=1)), 1e-12)
    T = np.array([[sc, 0, -sc * m[0]], [0, sc, -sc * m[1]], [0, 0, 1]])
    return (p - m) * sc, T


def dlt_homography(p1, p2):
    """Нормализованный DLT: H такая, что p2 ~ H·p1. Нужно ≥ 4 пар; при n > 4 — МНК (SVD)."""
    a, T1 = normalize_points(p1)
    b, T2 = normalize_points(p2)
    n = len(a)
    A = np.zeros((2 * n, 9))
    x, y, u, v = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    A[0::2, 0:3] = np.c_[-x, -y, -np.ones(n)]
    A[0::2, 6:9] = np.c_[u * x, u * y, u]
    A[1::2, 3:6] = np.c_[-x, -y, -np.ones(n)]
    A[1::2, 6:9] = np.c_[v * x, v * y, v]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def ransac_homography(p1, p2, thr=3.0, conf=0.99, max_iter=2000, seed=0):
    """RANSAC для гомографии с адаптивным числом итераций.

    N = log(1 − p) / log(1 − w⁴), где w — доля inlier'ов лучшей гипотезы.
    После поиска — переоценка H по всем inlier'ам (DLT-МНК) и пересчёт inlier'ов.
    Возвращает H, маску inlier'ов, число выполненных итераций.
    """
    rng = np.random.default_rng(seed)
    n = len(p1)
    if n < 4:
        return None, np.zeros(n, bool), 0
    best_mask, best_cnt = np.zeros(n, bool), 0
    N, it = max_iter, 0
    while it < N:
        it += 1
        s = rng.choice(n, 4, replace=False)
        H = dlt_homography(p1[s], p2[s])
        if not np.all(np.isfinite(H)):
            continue
        with np.errstate(all="ignore"):
            err = np.linalg.norm(project(H, p1) - p2, axis=1)
        mask = err < thr
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_mask = cnt, mask
            w = cnt / n
            if w >= 1.0:
                N = it
            else:
                N = min(max_iter, int(np.ceil(np.log(1 - conf) / np.log(1 - w ** 4))))
    if best_cnt < 4:
        return None, best_mask, it
    H = dlt_homography(p1[best_mask], p2[best_mask])
    mask = np.linalg.norm(project(H, p1) - p2, axis=1) < thr
    if mask.sum() >= 4:
        H = dlt_homography(p1[mask], p2[mask])
    return H, mask, it


def corner_error(H_est, H_gt, w, h):
    """Средняя ошибка (px) положения 4 углов изображения w×h под H_est относительно H_gt."""
    if H_est is None:
        return np.inf
    cr = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    return float(np.mean(np.linalg.norm(project(H_est, cr) - project(H_gt, cr), axis=1)))
