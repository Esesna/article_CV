"""Масштабное пространство SIFT: базовое изображение, гауссова пирамида, DoG.

Соответствует разд. 3 и 3.2–3.3 конспекта (../../rus/SIFT_конспект.md).
Схема повторяет OpenCV (modules/features2d/src/sift.dispatch.cpp), чтобы
результаты можно было сравнивать «точка в точку».
"""
import numpy as np
import cv2


def to_float01(img):
    """Перевести изображение (uint8, серое или BGR) в float32 серое в диапазоне [0, 1]."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1.5:          # uint8-шкала 0..255
        img = img / 255.0
    return img


def base_image(img, sigma=1.6, upsample=True, assumed_blur=0.5):
    """Базовое изображение октавы 0 (разд. 3.3).

    Считается, что вход уже размыт с σ=assumed_blur (0.5). При upsample=True
    изображение удваивается билинейной интерполяцией (как в статье и OpenCV),
    после чего его размытие становится 2·0.5 = 1.0. Досглаживаем до σ:
    σ_diff = sqrt(σ² − σ_имеющееся²).
    """
    img = to_float01(img)
    if upsample:
        h, w = img.shape
        img = cv2.resize(img, (2 * w, 2 * h), interpolation=cv2.INTER_LINEAR)
        sig_diff = np.sqrt(max(sigma ** 2 - (2 * assumed_blur) ** 2, 0.01))
    else:
        sig_diff = np.sqrt(max(sigma ** 2 - assumed_blur ** 2, 0.01))
    return cv2.GaussianBlur(img, (0, 0), sig_diff)


def num_octaves(base_shape, upsample=True):
    """Число октав, как в OpenCV: round(log2(min(H, W)) − 2) + (1, если было удвоение)."""
    return int(round(np.log2(min(base_shape)) - 2)) + (1 if upsample else 0)


def octave_sigmas(s=3, sigma=1.6):
    """Инкрементальные σ для s+3 изображений октавы.

    Полное размытие уровня i: σ_i = σ·k^i, k = 2^(1/s). Каждый следующий уровень
    получается из предыдущего свёрткой с σ_inc = sqrt(σ_i² − σ_{i−1}²)
    (свёртка гауссианов складывает дисперсии). Элемент 0 — σ самого базового уровня.
    """
    k = 2.0 ** (1.0 / s)
    sig = [sigma]
    for i in range(1, s + 3):
        sig_prev = sigma * k ** (i - 1)
        sig_total = sig_prev * k
        sig.append(np.sqrt(sig_total ** 2 - sig_prev ** 2))
    return np.array(sig)


def gaussian_pyramid(base, n_oct, s=3, sigma=1.6):
    """Гауссова пирамида: список октав, каждая — массив (s+3, H_o, W_o) float32.

    Первое изображение октавы o>0 — изображение с индексом s предыдущей октавы
    (его σ = 2σ), прореженное в 2 раза (берём каждый второй пиксель).
    """
    sig = octave_sigmas(s, sigma)
    pyr = []
    for o in range(n_oct):
        if o == 0:
            first = base
        else:
            first = pyr[o - 1][s][::2, ::2]
            # OpenCV: resize(..., Size(cols/2, rows/2), INTER_NEAREST) == срез [::2, ::2]
            h2, w2 = pyr[o - 1][s].shape[0] // 2, pyr[o - 1][s].shape[1] // 2
            first = np.ascontiguousarray(first[:h2, :w2])
        layers = [first]
        for i in range(1, s + 3):
            layers.append(cv2.GaussianBlur(layers[-1], (0, 0), sig[i]))
        pyr.append(np.stack(layers).astype(np.float32))
    return pyr


def dog_pyramid(gpyr):
    """DoG: D_i = L_{i+1} − L_i для каждой октавы → массивы (s+2, H_o, W_o)."""
    return [g[1:] - g[:-1] for g in gpyr]


def build_scale_space(img, s=3, sigma=1.6, upsample=True):
    """Собрать всё масштабное пространство. Возвращает dict с ключами
    gauss, dog, s, sigma, first_octave (−1 при удвоении, иначе 0)."""
    base = base_image(img, sigma, upsample)
    n_oct = num_octaves(base.shape, upsample)
    gpyr = gaussian_pyramid(base, n_oct, s, sigma)
    return dict(gauss=gpyr, dog=dog_pyramid(gpyr), s=s, sigma=sigma,
                first_octave=-1 if upsample else 0)
