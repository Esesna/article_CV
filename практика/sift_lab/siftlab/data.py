"""Тестовые данные: загрузка примеров OpenCV с кэшированием в data/ и
синтетическая текстура на случай работы без сети."""
from pathlib import Path
import urllib.request

import numpy as np
import cv2

OPENCV_SAMPLES = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"


def synthetic_texture(w=480, h=360, seed=0):
    """Синтетическое «текстурное» изображение: пятна разных масштабов, прямоугольники, текст."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 128, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(120):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        s = rng.uniform(2, 25)
        img += rng.choice([-1, 1]) * rng.uniform(30, 90) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(25):
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        cv2.rectangle(img, (x, y), (x + int(rng.uniform(8, 60)), y + int(rng.uniform(8, 60))),
                      int(rng.uniform(0, 255)), -1)
    for k in range(6):
        cv2.putText(img, "SIFT%d" % k, (int(rng.uniform(0, w - 120)), int(rng.uniform(30, h))),
                    cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.6, 1.5), int(rng.uniform(0, 255)), 2)
    return img


def load_sample(name, data_dir="data", timeout=10):
    """Загрузить серое изображение name из примеров OpenCV (кэш в data_dir).
    Без сети — синтетическая текстура (возвращается флаг synthetic=True)."""
    path = Path(data_dir) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(OPENCV_SAMPLES + name, timeout=timeout) as r:
                path.write_bytes(r.read())
        except Exception as e:                                    # нет сети и т.п.
            print(f"[load_sample] не удалось скачать {name}: {e}; беру синтетику")
            return synthetic_texture(seed=sum(name.encode()) % 1000), True
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return synthetic_texture(seed=sum(name.encode()) % 1000), True
    return img, False
