"""siftlab — учебная реализация SIFT «своими руками» (numpy + базовые примитивы OpenCV).

Модули повторяют разделы методички sift_методичка.ipynb и конспекта
../../rus/SIFT_конспект.md: pyramid (разд. 3), detector (разд. 3.1, 4, 4.1),
orientation (разд. 5), descriptor (разд. 6.1), matching (разд. 7.1),
geometry (RANSAC/гомография), evaluation (метрики на парах с известной H).
"""
from .pyramid import (to_float01, base_image, num_octaves, octave_sigmas,
                      gaussian_pyramid, dog_pyramid, build_scale_space)
from .detector import find_extrema, derivatives, refine_extrema, detect_keypoints
from .orientation import (gradient_polar, gradient_pyramid, window,
                          orientation_histogram, histogram_peaks, assign_orientations)
from .descriptor import (sift_descriptor, normalize_clip, compute_descriptors,
                         to_opencv_scale, root_sift)
from .matching import l2_distances, knn2, ratio_match, mutual_match
from .geometry import (project, normalize_points, dlt_homography,
                       ransac_homography, corner_error)
from .evaluation import (synthetic_homography, make_pair, inside, repeatability,
                         match_correctness, n_possible)
from .pipeline import (remove_duplicates, top_n, sift_detect_and_compute, kp_to_cv,
                       unpack_octave, kp_from_cv, kp_xy)
from .data import synthetic_texture, load_sample
