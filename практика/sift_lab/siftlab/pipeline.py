"""Полный конвейер «своего» SIFT и конвертация в/из cv2.KeyPoint."""
import numpy as np
import cv2

from .pyramid import build_scale_space
from .detector import detect_keypoints
from .orientation import gradient_pyramid, assign_orientations
from .descriptor import compute_descriptors


def remove_duplicates(kp):
    """Удалить точные дубликаты (x, y, size, angle) — аналог KeyPointsFilter::removeDuplicated."""
    key = np.round(np.c_[kp["x"], kp["y"], kp["size"], kp["angle"]], 4)
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    return {k: v[first] for k, v in kp.items()}


def top_n(kp, n):
    """Оставить n точек с наибольшим откликом |D(x̂)| (аналог nfeatures в OpenCV)."""
    if n is None or n <= 0 or len(kp["x"]) <= n:
        return kp
    order = np.argsort(-kp["response"], kind="stable")[:n]
    return {k: v[order] for k, v in kp.items()}


def sift_detect_and_compute(img, n_features=0, s=3, sigma=1.6,
                            contrast_thr=0.04 / 3, edge_r=10.0):
    """Свой SIFT целиком: пирамида → детектор → ориентации → дескрипторы.
    Возвращает (kp: dict массивов, desc: (n, 128) float32 единичной нормы)."""
    ss = build_scale_space(img, s, sigma)
    kp = detect_keypoints(ss, contrast_thr, edge_r)
    grads = gradient_pyramid(ss["gauss"], s)
    kp = assign_orientations(kp, grads)
    kp = top_n(remove_duplicates(kp), n_features)
    desc = compute_descriptors(kp, grads)
    return kp, desc


def kp_to_cv(kp, first_octave=-1):
    """dict массивов → список cv2.KeyPoint (octave закодирован как в OpenCV)."""
    res = []
    for k in range(len(kp["x"])):
        xi = kp["xi"][k]
        o = int(kp["octave"][k]) + first_octave
        code = (o & 255) | (int(kp["layer"][k]) << 8) | (int(np.rint((xi + 0.5) * 255)) << 16)
        res.append(cv2.KeyPoint(float(kp["x"][k]), float(kp["y"][k]), float(kp["size"][k]),
                                float(kp["angle"][k]), float(kp["response"][k]), code))
    return res


def unpack_octave(code):
    """Расшифровка KeyPoint.octave OpenCV: (октава со знаком, слой, xi).
    Младший байт — октава (−1 = удвоенное изображение), второй — слой, третий — (xi+0.5)·255."""
    o = code & 255
    o = o - 256 if o >= 128 else o
    return o, (code >> 8) & 255, ((code >> 16) & 255) / 255.0 - 0.5


def kp_from_cv(kps, s=3, sigma=1.6, first_octave=-1):
    """Список cv2.KeyPoint → dict массивов в нашем формате (для расчёта своих дескрипторов
    на точках OpenCV). Позиция в октаве = pt / 2^o, масштаб σ_oct = size/2 / 2^o."""
    cols = {k: [] for k in ["x", "y", "octave", "layer", "r", "c", "xi",
                            "sigma_oct", "size", "response", "angle"]}
    for p in kps:
        o, layer, xi = unpack_octave(p.octave)
        sc = 2.0 ** (-o)
        cols["x"].append(p.pt[0]); cols["y"].append(p.pt[1])
        cols["octave"].append(o - first_octave); cols["layer"].append(layer)
        cols["r"].append(int(np.rint(p.pt[1] * sc))); cols["c"].append(int(np.rint(p.pt[0] * sc)))
        cols["xi"].append(xi); cols["sigma_oct"].append(p.size * sc * 0.5)
        cols["size"].append(p.size); cols["response"].append(p.response)
        cols["angle"].append(p.angle)
    kp = {k: np.array(v) for k, v in cols.items()}
    for k in ("octave", "layer", "r", "c"):
        kp[k] = kp[k].astype(int)
    return kp


def kp_xy(kps):
    """Координаты (n, 2) из списка cv2.KeyPoint или из нашего dict."""
    if isinstance(kps, dict):
        return np.c_[kps["x"], kps["y"]]
    return np.array([p.pt for p in kps], float).reshape(-1, 2)
