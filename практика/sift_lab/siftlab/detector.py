"""Детектор SIFT: экстремумы DoG по 26 соседям, уточнение по Тейлору,
отбраковка по контрасту и по краям (разд. 3.1, 4, 4.1 конспекта).

Ключевые точки храним как dict одномерных массивов одинаковой длины:
  x, y        — координаты во входном изображении (как KeyPoint.pt в OpenCV);
  octave      — индекс октавы в пирамиде (0 — удвоенное изображение);
  layer       — номер слоя DoG (1..s); r, c — целочисленная позиция в октаве;
  xi          — дробное смещение по масштабу; sigma_oct — σ точки в пикселях октавы;
  size        — диаметр как в OpenCV (2·σ во входном изображении); response — |D(x̂)|;
  angle       — ориентация в градусах (заполняет orientation.py).
"""
import numpy as np


def find_extrema(dog_oct, threshold, border=5):
    """Векторизованный поиск экстремумов по 26 соседям (3×3×3 − 1) в одной октаве.

    dog_oct: (L, H, W). Проверяются слои 1..L−2 (нужны соседи сверху и снизу).
    Условие как в OpenCV: |v| > threshold и v ≥ всех 26 соседей (для максимума)
    или v ≤ всех (для минимума). Возвращает массивы (layer, r, c).
    """
    D = dog_oct
    L, H, W = D.shape
    C = D[1:-1, 1:-1, 1:-1]
    mx = np.full(C.shape, -np.inf, np.float32)
    mn = np.full(C.shape, np.inf, np.float32)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                nb = D[1 + dz:L - 1 + dz, 1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
                np.maximum(mx, nb, out=mx)
                np.minimum(mn, nb, out=mn)
    ext = ((C > threshold) & (C >= mx)) | ((C < -threshold) & (C <= mn))
    # отступ от края (координаты C сдвинуты на 1)
    b = border - 1
    ext[:, :b, :] = False
    ext[:, H - 2 - b:, :] = False
    ext[:, :, :b] = False
    ext[:, :, W - 2 - b:] = False
    l, r, c = np.nonzero(ext)
    return l + 1, r + 1, c + 1


def derivatives(D, l, r, c):
    """Градиент (3,) и гессиан (3×3) DoG конечными разностями в точках (l, r, c).

    Порядок координат (x, y, σ) = (c, r, l). Возвращает g: (n, 3), Hs: (n, 3, 3), v: (n,).
    """
    v = D[l, r, c]
    dx = (D[l, r, c + 1] - D[l, r, c - 1]) * 0.5
    dy = (D[l, r + 1, c] - D[l, r - 1, c]) * 0.5
    ds = (D[l + 1, r, c] - D[l - 1, r, c]) * 0.5
    dxx = D[l, r, c + 1] + D[l, r, c - 1] - 2 * v
    dyy = D[l, r + 1, c] + D[l, r - 1, c] - 2 * v
    dss = D[l + 1, r, c] + D[l - 1, r, c] - 2 * v
    dxy = (D[l, r + 1, c + 1] - D[l, r + 1, c - 1] - D[l, r - 1, c + 1] + D[l, r - 1, c - 1]) * 0.25
    dxs = (D[l + 1, r, c + 1] - D[l + 1, r, c - 1] - D[l - 1, r, c + 1] + D[l - 1, r, c - 1]) * 0.25
    dys = (D[l + 1, r + 1, c] - D[l + 1, r - 1, c] - D[l - 1, r + 1, c] + D[l - 1, r - 1, c]) * 0.25
    g = np.stack([dx, dy, ds], 1).astype(np.float64)
    Hs = np.stack([np.stack([dxx, dxy, dxs], 1),
                   np.stack([dxy, dyy, dys], 1),
                   np.stack([dxs, dys, dss], 1)], 1).astype(np.float64)
    return g, Hs, v


def refine_extrema(D, l, r, c, s=3, contrast_thr=0.04 / 3, edge_r=10.0,
                   border=5, max_iter=5, return_stats=False):
    """Субпиксельное/субмасштабное уточнение (формулы (2)–(3)) + два фильтра.

    x̂ = −H⁻¹·∇D. Если |x̂| > 0.5 по любой координате — переходим в соседний
    отсчёт и повторяем (до max_iter раз). Затем:
      контраст: |D(x̂)| = |D + ½∇Dᵀx̂| < contrast_thr → отброс;
      края:     Tr(H)²/Det(H) ≥ (r+1)²/r или Det ≤ 0 → отброс (гессиан 2×2 по x,y).
    contrast_thr задаётся в шкале яркости [0, 1]. OpenCV использует
    contrastThreshold / nOctaveLayers = 0.04/3 ≈ 0.0133; в статье — 0.03.
    Возвращает dict с массивами l, r, c, xc, xr, xi, contr (и статистику этапов).
    """
    L, H, W = D.shape
    l, r, c = l.copy(), r.copy(), c.copy()
    n = len(l)
    off = np.zeros((n, 3))
    alive = np.ones(n, bool)
    conv = np.zeros(n, bool)
    for it in range(max_iter):
        idx = np.nonzero(alive & ~conv)[0]
        if len(idx) == 0:
            break
        g, Hs, _ = derivatives(D, l[idx], r[idx], c[idx])
        det = np.linalg.det(Hs)
        ok = np.abs(det) > 1e-12
        X = np.zeros_like(g)
        X[ok] = np.linalg.solve(Hs[ok], g[ok][..., None])[..., 0]
        xh = -X                                   # (xc, xr, xi)
        small = np.all(np.abs(xh) < 0.5, 1)
        conv[idx[small]] = True
        off[idx[small]] = xh[small]
        mv = idx[~small]
        step = np.rint(xh[~small])
        huge = np.any(np.abs(xh[~small]) > 1e6, 1)
        alive[mv[huge]] = False
        c[mv] += step[:, 0].astype(int)
        r[mv] += step[:, 1].astype(int)
        l[mv] += step[:, 2].astype(int)
        out = (l[mv] < 1) | (l[mv] > s) | (c[mv] < border) | (c[mv] >= W - border) \
              | (r[mv] < border) | (r[mv] >= H - border)
        alive[mv[out]] = False
    alive &= conv
    n_conv = int(alive.sum())
    idx = np.nonzero(alive)[0]
    g, Hs, v = derivatives(D, l[idx], r[idx], c[idx])
    contr = v + 0.5 * np.sum(g * off[idx], 1)
    keep_c = np.abs(contr) >= contrast_thr
    dxx, dyy, dxy = Hs[:, 0, 0], Hs[:, 1, 1], Hs[:, 0, 1]
    tr = dxx + dyy
    det2 = dxx * dyy - dxy ** 2
    keep_e = (det2 > 0) & (tr ** 2 * edge_r < (edge_r + 1) ** 2 * det2)
    keep = keep_c & keep_e
    idx2 = idx[keep]
    res = dict(l=l[idx2], r=r[idx2], c=c[idx2], xc=off[idx2, 0], xr=off[idx2, 1],
               xi=off[idx2, 2], contr=contr[keep])
    if return_stats:
        res["stats"] = dict(candidates=n, converged=n_conv,
                            after_contrast=int(keep_c.sum()),
                            after_edge=int(keep.sum()))
        # позиции (x, y в пикселях октавы) отброшенных точек — для визуализации
        xy = np.c_[c[idx] + off[idx, 0], r[idx] + off[idx, 1]]
        res["rejected"] = dict(contrast=xy[~keep_c], edge=xy[keep_c & ~keep_e])
    return res


def detect_keypoints(ss, contrast_thr=0.04 / 3, edge_r=10.0, border=5,
                     return_stats=False):
    """Детектор по всем октавам масштабного пространства ss (см. build_scale_space).

    Предварительный порог как в OpenCV: floor(0.5·thr·255)/255 (в шкале uint8).
    Возвращает dict массивов ключевых точек (angle пока не заполнен); при
    return_stats=True — ещё и счётчики этапов и координаты отброшенных точек.
    """
    s, sigma, fo = ss["s"], ss["sigma"], ss["first_octave"]
    pre_thr = np.floor(0.5 * contrast_thr * 255) / 255
    keys = ["x", "y", "octave", "layer", "r", "c", "xi", "sigma_oct", "size", "response"]
    out = {k: [] for k in keys}
    stats = dict(candidates=0, converged=0, after_contrast=0, after_edge=0)
    rejected = dict(contrast=[], edge=[])
    for o, D in enumerate(ss["dog"]):
        if min(D.shape[1:]) <= 2 * border + 2:
            continue
        l, r, c = find_extrema(D, pre_thr, border)
        e = refine_extrema(D, l, r, c, s, contrast_thr, edge_r, border,
                           return_stats=True)
        for k in stats:
            stats[k] += e["stats"][k]
        scale = 2.0 ** (o + fo)                   # пиксель октавы → пиксель входа
        for k in rejected:
            rejected[k].append(e["rejected"][k] * scale)
        sig_oct = sigma * 2.0 ** ((e["l"] + e["xi"]) / s)
        out["x"].append((e["c"] + e["xc"]) * scale)
        out["y"].append((e["r"] + e["xr"]) * scale)
        out["octave"].append(np.full(len(e["l"]), o))
        out["layer"].append(e["l"])
        out["r"].append(e["r"])
        out["c"].append(e["c"])
        out["xi"].append(e["xi"])
        out["sigma_oct"].append(sig_oct)
        out["size"].append(2 * sig_oct * scale)
        out["response"].append(np.abs(e["contr"]))
    kp = {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
    for k in ("octave", "layer", "r", "c"):
        kp[k] = kp[k].astype(int)
    if return_stats:
        stats["rejected"] = {k: np.concatenate(v) if v else np.zeros((0, 2))
                             for k, v in rejected.items()}
        return kp, stats
    return kp
