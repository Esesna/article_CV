"""Назначение ориентации (разд. 5 конспекта).

Гистограмма из 36 бинов (по 10°) направлений градиента в окрестности точки,
взвешенных модулем градиента и гауссианом с σ = 1.5·σ_точки. Пики ≥ 80% от
максимума порождают отдельные ключевые точки; положение пика уточняется параболой.

Соглашение об угле — как в OpenCV: KeyPoint.angle ∈ [0, 360) — направление
градиента в координатах изображения (ось y вниз), т.е. по часовой стрелке на экране.
"""
import numpy as np


def gradient_polar(img):
    """Модуль и направление (градусы, ось y ВВЕРХ — как в коде OpenCV) градиента.

    dx = I[y, x+1] − I[y, x−1],  dy = I[y−1, x] − I[y+1, x].
    На крайних пикселях модуль = 0 (OpenCV их пропускает).
    """
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[1:-1, 1:-1] = img[1:-1, 2:] - img[1:-1, :-2]
    dy[1:-1, 1:-1] = img[:-2, 1:-1] - img[2:, 1:-1]
    mag = np.sqrt(dx * dx + dy * dy)
    ori = np.degrees(np.arctan2(dy, dx)) % 360.0
    return mag, ori


def gradient_pyramid(gpyr, s=3):
    """Предвычислить (mag, ori) для слоёв 1..s каждой октавы гауссовой пирамиды."""
    return [[gradient_polar(g[i]) if 1 <= i <= s else None for i in range(len(g))]
            for g in gpyr]


def window(mag, ori, r, c, radius):
    """Квадратное окно (2·radius+1)² вокруг (r, c) с обрезкой по краям изображения.

    Возвращает mag, ori окна и смещения i (строки), j (столбцы) каждого пикселя.
    """
    H, W = mag.shape
    r0, r1 = max(r - radius, 0), min(r + radius + 1, H)
    c0, c1 = max(c - radius, 0), min(c + radius + 1, W)
    i, j = np.mgrid[r0 - r:r1 - r, c0 - c:c1 - c]
    return mag[r0:r1, c0:c1], ori[r0:r1, c0:c1], i, j


def orientation_histogram(mag, ori, r, c, sigma_oct, nbins=36):
    """Сглаженная 36-бинная гистограмма ориентаций для точки (r, c) октавы."""
    radius = int(np.rint(4.5 * sigma_oct))      # OpenCV: SIFT_ORI_RADIUS = 3·1.5
    sig = 1.5 * sigma_oct
    m, a, i, j = window(mag, ori, r, c, radius)
    w = np.exp(-(i * i + j * j) / (2 * sig * sig)) * m
    b = np.rint(a * nbins / 360.0).astype(int) % nbins
    hist = np.bincount(b.ravel(), weights=w.ravel(), minlength=nbins)
    # сглаживание круговым ядром [1 4 6 4 1]/16
    hist = (np.roll(hist, 2) + np.roll(hist, -2)) / 16 \
        + (np.roll(hist, 1) + np.roll(hist, -1)) * 4 / 16 + hist * 6 / 16
    return hist


def histogram_peaks(hist, peak_ratio=0.8):
    """Пики ≥ peak_ratio·max, уточнённые параболой. Возвращает углы OpenCV (градусы)."""
    n = len(hist)
    left, right = np.roll(hist, 1), np.roll(hist, -1)
    pk = np.nonzero((hist > left) & (hist > right) & (hist >= peak_ratio * hist.max()))[0]
    den = left[pk] - 2 * hist[pk] + right[pk]
    b = (pk + 0.5 * (left[pk] - right[pk]) / den) % n
    ang = 360.0 - 360.0 / n * b               # перевод «ось y вверх» → «ось y вниз»
    ang[np.abs(ang - 360.0) < 1e-6] = 0.0
    return ang


def assign_orientations(kp, grads, nbins=36, peak_ratio=0.8):
    """Для каждой точки — гистограмма, пики; точка дублируется на каждый пик."""
    rows, angles = [], []
    for k in range(len(kp["x"])):
        mag, ori = grads[kp["octave"][k]][kp["layer"][k]]
        h = orientation_histogram(mag, ori, kp["r"][k], kp["c"][k], kp["sigma_oct"][k], nbins)
        for a in histogram_peaks(h, peak_ratio):
            rows.append(k)
            angles.append(a)
    rows = np.array(rows, int)
    out = {key: v[rows] for key, v in kp.items()}
    out["angle"] = np.array(angles)
    return out
