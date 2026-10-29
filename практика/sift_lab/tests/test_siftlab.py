"""Быстрые проверки пакета siftlab.

Запуск:  .venv/bin/python -m pytest -q tests      (pytest установлен в .venv)
   или:  .venv/bin/python -m tests.test_siftlab   (без pytest)
Сеть не нужна: используется синтетическая текстура.
"""
import numpy as np
import cv2

import siftlab as sl

IMG = sl.synthetic_texture(320, 240, seed=3)


def test_matcher_equals_brute_force():
    rng = np.random.default_rng(0)
    A = rng.random((60, 128)).astype(np.float32)
    B = rng.random((80, 128)).astype(np.float32)
    idx, dist = sl.knn2(A, B, block=16)
    D = np.linalg.norm(A[:, None] - B[None], axis=2)
    ref = np.argsort(D, 1)[:, :2]
    assert np.array_equal(idx, ref)
    assert np.allclose(dist, np.take_along_axis(D, ref, 1), atol=1e-4)
    M = sl.mutual_match(A, B)
    assert all(D[:, j].argmin() == i and D[i].argmin() == j for i, j in M)


def test_ransac_recovers_homography():
    H = sl.synthetic_homography(400, 300, angle=20, scale=0.9, persp=(0.2, -0.1))
    rng = np.random.default_rng(1)
    p1 = rng.uniform(0, 400, (300, 2))
    p2 = sl.project(H, p1) + rng.normal(0, 0.3, (300, 2))
    out = rng.random(300) < 0.5
    p2[out] = rng.uniform(0, 400, (out.sum(), 2))
    H_est, mask, _ = sl.ransac_homography(p1, p2, thr=2.0)
    assert sl.corner_error(H_est, H, 400, 300) < 0.5
    assert (mask == ~out).mean() > 0.97


def test_descriptor_norm_and_clip():
    kp, d = sl.sift_detect_and_compute(IMG, n_features=200)
    assert len(kp["x"]) > 50
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
    # после обрезки 0.2 и перенормировки компонента может немного превысить 0.2, но не сильно
    assert d.max() < 0.35 and d.min() >= 0


def test_detector_matches_opencv():
    kp, _ = sl.sift_detect_and_compute(IMG)
    kps = cv2.SIFT_create().detect(IMG, None)
    P, Q = sl.kp_xy(kp), sl.kp_xy(kps)
    d = np.sqrt(((P[:, None] - Q[None]) ** 2).sum(-1)).min(1)
    assert (d < 2).mean() > 0.95
    assert abs(len(P) - len(Q)) <= 0.05 * len(Q)


def test_descriptor_matches_opencv_on_same_keypoints():
    # измерено: cos ≥ 0.99997 для всех точек (graf1, 400×320); порог с запасом
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(IMG, None)
    ss = sl.build_scale_space(IMG)
    grads = sl.gradient_pyramid(ss["gauss"], ss["s"])
    mine = sl.compute_descriptors(sl.kp_from_cv(kps), grads)
    cos = (mine * des / np.linalg.norm(des, axis=1, keepdims=True)).sum(1)
    assert np.median(cos) > 0.9999
    assert cos.min() > 0.99


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK ", t.__name__)
    print(f"все {len(tests)} тестов пройдены")
