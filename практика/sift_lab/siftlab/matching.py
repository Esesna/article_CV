"""Сопоставление дескрипторов (разд. 7.1 конспекта): полный перебор L2,
тест отношения расстояний (ratio test) и взаимная проверка (cross-check).
"""
import numpy as np


def l2_distances(A, B):
    """Матрица евклидовых расстояний через ‖a‖² + ‖b‖² − 2·a·b (одно матричное умножение)."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(d2, 0.0))


def knn2(A, B, block=2048):
    """Два ближайших соседа из B для каждой строки A. Возвращает idx (n,2), dist (n,2).
    Считаем блоками, чтобы матрица расстояний не занимала много памяти."""
    idx = np.zeros((len(A), 2), int)
    dist = np.zeros((len(A), 2), np.float32)
    for s in range(0, len(A), block):
        Dm = l2_distances(A[s:s + block], B)
        top = np.argpartition(Dm, 1, axis=1)[:, :2]
        dt = np.take_along_axis(Dm, top, 1)
        order = np.argsort(dt, 1)
        idx[s:s + block] = np.take_along_axis(top, order, 1)
        dist[s:s + block] = np.take_along_axis(dt, order, 1)
    return idx, dist


def ratio_match(A, B, ratio=0.8):
    """Тест отношения Лоу: принять NN, если d1/d2 < ratio.
    Возвращает matches (m, 2) пар индексов (i в A, j в B), отношения всех точек и knn."""
    idx, dist = knn2(A, B)
    r = dist[:, 0] / np.maximum(dist[:, 1], 1e-12)
    keep = r < ratio
    matches = np.stack([np.nonzero(keep)[0], idx[keep, 0]], 1)
    return matches, r, idx, dist


def mutual_match(A, B):
    """Взаимно ближайшие соседи (аналог BFMatcher(crossCheck=True))."""
    Dm = l2_distances(A, B)
    ab = Dm.argmin(1)
    ba = Dm.argmin(0)
    i = np.nonzero(ba[ab] == np.arange(len(A)))[0]
    return np.stack([i, ab[i]], 1)

