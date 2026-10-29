"""Дескриптор SIFT (разд. 6.1 конспекта): 4×4 ячейки × 8 направлений = 128 чисел.

Окно поворачивается на ориентацию точки; каждая ячейка имеет ширину
3·σ_точки пикселей октавы (константа OpenCV SIFT_DESCR_SCL_FCTR = 3).
Вклад пикселя = модуль градиента × гауссиан (σ = половина ширины окна
дескриптора, т.е. d/2 ячеек) и распределяется трилинейно по 2×2×2 соседним
бинам (строка, столбец, ориентация). Затем: нормировка → обрезка 0.2 → нормировка.
"""
import numpy as np


def sift_descriptor(mag, ori, r, c, angle, sigma_oct, d=4, n=8, clip=0.2):
    """Дескриптор одной точки по предвычисленным mag/ori (ось y вверх) слоя октавы.

    r, c — целочисленная позиция в октаве; angle — угол OpenCV (ось y вниз);
    sigma_oct — масштаб точки в пикселях октавы. Возвращает вектор d·d·n единичной нормы.
    """
    H, W = mag.shape
    ori_kp = (360.0 - angle) % 360.0          # обратно в систему «ось y вверх»
    cos_t, sin_t = np.cos(np.radians(ori_kp)), np.sin(np.radians(ori_kp))
    hist_width = 3.0 * sigma_oct              # ширина ячейки в пикселях октавы
    radius = int(np.rint(hist_width * np.sqrt(2) * (d + 1) * 0.5))
    radius = min(radius, int(np.sqrt(H * H + W * W)))
    i, j = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    # поворот смещений в систему координат точки, единица — ширина ячейки
    c_rot = (j * cos_t - i * sin_t) / hist_width
    r_rot = (j * sin_t + i * cos_t) / hist_width
    rbin = r_rot + d / 2 - 0.5
    cbin = c_rot + d / 2 - 0.5
    rr, cc = r + i, c + j
    ok = (rbin > -1) & (rbin < d) & (cbin > -1) & (cbin < d) \
        & (rr > 0) & (rr < H - 1) & (cc > 0) & (cc < W - 1)
    rbin, cbin, rr, cc = rbin[ok], cbin[ok], rr[ok], cc[ok]
    w = np.exp(-(c_rot[ok] ** 2 + r_rot[ok] ** 2) / (d * d * 0.5))  # σ = d/2 ячеек
    m = mag[rr, cc] * w
    obin = (ori[rr, cc] - ori_kp) * (n / 360.0)
    r0, c0, o0 = np.floor(rbin), np.floor(cbin), np.floor(obin)
    fr, fc, fo = rbin - r0, cbin - c0, obin - o0
    r0, c0, o0 = r0.astype(int), c0.astype(int), o0.astype(int) % n
    hist = np.zeros((d + 2) * (d + 2) * (n + 2))
    for dr, wr in ((0, 1 - fr), (1, fr)):
        for dc, wc in ((0, 1 - fc), (1, fc)):
            for do, wo in ((0, 1 - fo), (1, fo)):
                idx = ((r0 + 1 + dr) * (d + 2) + (c0 + 1 + dc)) * (n + 2) + (o0 + do)
                hist += np.bincount(idx, m * wr * wc * wo, minlength=hist.size)
    hist = hist.reshape(d + 2, d + 2, n + 2)
    hist[:, :, 0] += hist[:, :, n]            # круговой перенос по ориентации
    desc = hist[1:d + 1, 1:d + 1, :n].ravel()
    return normalize_clip(desc, clip)


def normalize_clip(v, clip=0.2):
    """Нормировка → обрезка компонент > clip·‖v‖ → повторная нормировка (разд. 6.1)."""
    nrm = np.linalg.norm(v)
    v = np.minimum(v, clip * nrm)
    return (v / max(np.linalg.norm(v), 1e-12)).astype(np.float32)


def compute_descriptors(kp, grads, d=4, n=8, clip=0.2):
    """Дескрипторы для всех точек kp (нужны поля octave, layer, r, c, angle, sigma_oct)."""
    out = np.zeros((len(kp["x"]), d * d * n), np.float32)
    for k in range(len(kp["x"])):
        mag, ori = grads[kp["octave"][k]][kp["layer"][k]]
        out[k] = sift_descriptor(mag, ori, kp["r"][k], kp["c"][k], kp["angle"][k],
                                 kp["sigma_oct"][k], d, n, clip)
    return out


def to_opencv_scale(desc):
    """Перевод в шкалу OpenCV: ×512, округление и насыщение до 255 (как uchar)."""
    return np.clip(np.rint(desc * 512.0), 0, 255).astype(np.float32)


def root_sift(desc, eps=1e-12):
    """RootSIFT (Arandjelović & Zisserman, 2012): L1-нормировка и поэлементный корень.
    Евклидово расстояние между RootSIFT ≈ ядро Хеллингера между исходными SIFT."""
    desc = desc / (np.abs(desc).sum(1, keepdims=True) + eps)
    return np.sqrt(desc).astype(np.float32)
