#!/usr/bin/env python3
"""Сборка sift_методичка.ipynb из текста ячеек ниже и исходников модуля siftlab/.

Функции в ячейках ноутбука вставляются ДОСЛОВНО из siftlab/*.py (через ast),
поэтому код методички и модуля не расходится. После правки модуля:
    .venv/bin/python tools/build_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace sift_методичка.ipynb
"""
import ast
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = {}
for f in sorted((ROOT / "siftlab").glob("*.py")):
    text = f.read_text()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            name = node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id
            SRC[name] = (f.name, ast.get_source_segment(text, node))


def defs(*names):
    """Код функций из модуля + комментарий-источник."""
    files = sorted({SRC[n][0] for n in names})
    head = f"# Код ниже совпадает с siftlab/{', siftlab/'.join(files)} (вставлен сборщиком tools/build_notebook.py)\n"
    return head + "\n\n\n".join(SRC[n][1] for n in names)


K = "../../rus/SIFT_конспект.md"
cells = []


def md(t):
    cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))


def code(t):
    cells.append(nbf.v4.new_code_cell(t.strip("\n")))


# ---------------------------------------------------------------- титул
md(f"""
# Практикум: SIFT своими руками

Методичка к конспекту [SIFT (Lowe, 2004)]({K}). Мы шаг за шагом реализуем весь конвейер
SIFT на `numpy` с базовыми примитивами OpenCV (`cv2.GaussianBlur`, `cv2.resize`), и на каждом шаге
сверяемся с эталоном — `cv2.SIFT_create`, `cv2.BFMatcher`, `cv2.FlannBasedMatcher`, `cv2.findHomography`.

| № | Раздел | Раздел конспекта |
|---|---|---|
| 0 | Подготовка, данные, синтетическая пара с известной гомографией | — |
| 1 | Эталон: SIFT из OpenCV, параметры, поля `KeyPoint` | Сводка гиперпараметров |
| 2 | Масштабное пространство: гауссова пирамида и DoG | 3, 3.0, 3.2, 3.3 |
| 3 | Детектор: 26 соседей, уточнение по Тейлору, контраст, края | 3.1, 4, 4.1 |
| 4 | Ориентация: 36-бинная гистограмма | 5 |
| 5 | Дескриптор 4×4×8 и проверка против OpenCV | 6.1 |
| 6 | Сопоставление: перебор L2, ratio test, cross-check, FLANN | 7.1, 7.2 |
| 7 | Геометрия: RANSAC + нормализованный DLT | 7.3–7.4, [RANSAC](../../rus/RANSAC_конспект.md) |
| 8 | Оценка и эксперименты на парах с известной H | Ключевые численные результаты |
| 9 | Упражнения | — |

**Как работать.** Выполняйте ячейки сверху вниз (`Run → Run All Cells` выполняется целиком примерно за полминуты на CPU).
Все функции определены прямо в ячейках, чтобы их можно было читать и менять; тот же код лежит в пакете
`siftlab/` (его использует `tests/test_siftlab.py`). В конце ноутбука есть ячейка, проверяющая, что код ноутбука
и пакета даёт одинаковый результат.

Соглашения: изображение — `float32` в $[0,1]$; координаты $(x, y)$ — столбец и строка, ось $y$ направлена **вниз**,
центр пикселя — целые координаты (как в OpenCV).
""")

# ---------------------------------------------------------------- 0
md("""
## 0. Подготовка

Импорты и вспомогательные функции для рисования.
""")
code("""
import time
import urllib.request
from pathlib import Path

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

%matplotlib inline
plt.rcParams.update({"figure.dpi": 80, "font.size": 9, "axes.titlesize": 10})
np.set_printoptions(precision=4, suppress=True)
T_START = time.time()
print("numpy", np.__version__, "| OpenCV", cv2.__version__, "| matplotlib", matplotlib.__version__)
""")
code('''
def show(*imgs, titles=None, size=4.0, cmap="gray", **kw):
    """Показать несколько изображений в ряд."""
    fig, axs = plt.subplots(1, len(imgs), figsize=(size * len(imgs), size * 0.8), squeeze=False)
    for k, (ax, im) in enumerate(zip(axs[0], imgs)):
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im, cmap=cmap, **kw)
        ax.set_title(titles[k] if titles else "")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_kps(ax, img, xy, size=None, angle=None, color="lime", lw=0.8):
    """Точки поверх изображения: окружность радиуса size/2 и отрезок ориентации.
    angle — в градусах по соглашению OpenCV (ось y вниз, по часовой стрелке на экране)."""
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    if size is None:
        ax.plot(xy[:, 0], xy[:, 1], ".", ms=2, color=color)
        return
    for k in range(len(xy)):
        x, y, rad = xy[k, 0], xy[k, 1], size[k] / 2
        ax.add_patch(Circle((x, y), rad, fill=False, color=color, lw=lw))
        if angle is not None:
            a = np.radians(angle[k])
            ax.plot([x, x + rad * np.cos(a)], [y, y + rad * np.sin(a)], color=color, lw=lw)
''')
md("""
### 0.1. Тестовые данные

Берём классические примеры OpenCV (`samples/data`): `box.png` / `box_in_scene.png` (объект и сцена) и
`graf1.png` / `graf3.png` (граффити, сильная смена ракурса). Файлы кэшируются в `data/`.
Если сети нет — подставляется синтетическая текстура, и ноутбук всё равно выполнится.
""")
code(defs("OPENCV_SAMPLES", "synthetic_texture", "load_sample"))
code("""
box, fb1 = load_sample("box.png")
scene, fb2 = load_sample("box_in_scene.png")
graf1, fb3 = load_sample("graf1.png")
graf3, fb4 = load_sample("graf3.png")
OFFLINE = fb1 or fb2 or fb3 or fb4
print("box", box.shape, "| scene", scene.shape, "| graf1", graf1.shape, "| graf3", graf3.shape,
      "| офлайн-режим:", OFFLINE)
show(box, scene, graf1, graf3, titles=["box.png", "box_in_scene.png", "graf1.png", "graf3.png"], size=3.2)
""")
md("""
### 0.2. Синтетическая пара с известной гомографией

Для честной количественной оценки нужна «истина»: где на втором изображении *должна* оказаться каждая точка
первого. Делаем второе изображение сами: $I_2 = \\mathrm{warp}(I_1, H)$, где
$H = T\\,P\\,S_{\\theta,s}$ — поворот на $\\theta$ и масштаб $s$ вокруг центра, затем перспективный «наклон» $P$.
Тогда точка $\\mathbf{p}_1$ переходит в $\\mathbf{p}_2 \\sim H\\mathbf{p}_1$ (в однородных координатах),
и совпадение $(\\mathbf{p}_1, \\mathbf{p}_2)$ считаем **верным**, если $\\lVert H\\mathbf{p}_1 - \\mathbf{p}_2\\rVert < 3$ px.

Базовое изображение — `graf1`, уменьшенное до 400×320 (чтобы «свой» SIFT на чистом Python работал быстро).
Ко второму изображению добавлен гауссов шум σ = 3 (из 255).
""")
code(defs("synthetic_homography", "make_pair"))
code("""
img_a = cv2.resize(graf1, (400, 320), interpolation=cv2.INTER_AREA)
H_gt = synthetic_homography(400, 320, angle=25, scale=0.85, persp=(0.25, -0.15))
img_b = make_pair(img_a, H_gt, noise=3)
print("H_gt =\\n", H_gt)

# визуальная проверка: сетка точек img_a, перенесённая в img_b
gx, gy = np.meshgrid(np.linspace(40, 360, 9), np.linspace(40, 280, 7))
grid = np.c_[gx.ravel(), gy.ravel()]
q = np.c_[grid, np.ones(len(grid))] @ H_gt.T
q = q[:, :2] / q[:, 2:]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
axs[0].imshow(img_a, cmap="gray"); axs[0].plot(grid[:, 0], grid[:, 1], "r+"); axs[0].set_title("img_a")
axs[1].imshow(img_b, cmap="gray"); axs[1].plot(q[:, 0], q[:, 1], "r+"); axs[1].set_title("img_b = warp(img_a, H_gt) + шум")
for ax in axs: ax.axis("off")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 1
md(f"""
## 1. Эталон: SIFT из OpenCV

Начнём с готовой реализации — она будет «линейкой» для всех следующих шагов
(см. «Сводка гиперпараметров» в [конспекте]({K})).

`cv2.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)`:

| Параметр | По умолчанию | Смысл | В статье |
|---|---|---|---|
| `nfeatures` | 0 (все) | оставить N лучших по `response` | — |
| `nOctaveLayers` | 3 | $s$ — интервалов на октаву, $k = 2^{{1/s}}$ | $s=3$ (разд. 3.2) |
| `contrastThreshold` | 0.04 | порог $\\lvert D(\\hat{{\\mathbf x}})\\rvert$, **делится на `nOctaveLayers`**: фактически $0.04/3 \\approx 0.0133$ | 0.03 (разд. 4) |
| `edgeThreshold` | 10 | $r$ в тесте краёв $\\mathrm{{Tr}}^2/\\mathrm{{Det}} < (r+1)^2/r$ | $r=10$ (разд. 4.1) |
| `sigma` | 1.6 | σ базового уровня октавы | 1.6 (разд. 3.3) |

**Про порог контраста.** OpenCV сравнивает $\\lvert D\\rvert\\cdot s$ с `contrastThreshold`, т.е. эффективный порог
$0.04/s$. Смысл деления: амплитуда DoG пропорциональна $k-1 \\approx \\ln 2 / s$, поэтому при изменении $s$ порог
масштабируется вместе с откликом. Документация OpenCV советует для «порога из статьи 0.03» ставить
`contrastThreshold=0.09` (тогда $0.09/3 = 0.03$). Порог по умолчанию OpenCV ($\\approx 0.0133$) **мягче** статьи
примерно в 2.25 раза → точек больше.
""")
code("""
sift_cv = cv2.SIFT_create()
print("nfeatures =", sift_cv.getNFeatures(), "| nOctaveLayers =", sift_cv.getNOctaveLayers(),
      "| contrastThreshold =", sift_cv.getContrastThreshold(), "| edgeThreshold =", sift_cv.getEdgeThreshold(),
      "| sigma =", sift_cv.getSigma())
t = time.perf_counter()
kps_box, des_box = sift_cv.detectAndCompute(box, None)
print(f"box.png: {len(kps_box)} точек, дескрипторы {des_box.shape} {des_box.dtype}, "
      f"{(time.perf_counter() - t) * 1e3:.1f} мс")
vis = cv2.drawKeypoints(box, kps_box, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show(vis, titles=["cv2.drawKeypoints(..., DRAW_RICH_KEYPOINTS): окружность = size, радиус-отрезок = angle"], size=6)
""")
md("""
### 1.1. Поля `cv2.KeyPoint`

* `pt` — $(x, y)$ во входном изображении (субпиксельно);
* `size` — **диаметр** области: $2\\sigma$, где $\\sigma$ — масштаб точки в пикселях входа;
* `angle` — ориентация в градусах $[0, 360)$, ось $y$ вниз (на экране — по часовой стрелке);
* `response` — $\\lvert D(\\hat{\\mathbf x})\\rvert$ (контраст), по нему работает `nfeatures`;
* `octave` — упакованное целое: младший байт — октава со знаком ($-1$ = удвоенное изображение),
  второй байт — слой DoG, третий — $(\\hat\\xi + 0.5)\\cdot 255$, где $\\hat\\xi$ — дробное смещение по масштабу.
""")
code(defs("unpack_octave"))
code("""
print(f"{'pt':>18} {'size':>6} {'σ=size/2':>8} {'angle':>7} {'response':>9} {'octave(hex)':>12}  (окт, слой, ξ)")
for p in sorted(kps_box, key=lambda p: -p.response)[:8]:
    o, layer, xi = unpack_octave(p.octave)
    print(f"({p.pt[0]:7.2f},{p.pt[1]:7.2f}) {p.size:6.2f} {p.size / 2:8.2f} {p.angle:7.2f} {p.response:9.5f} "
          f"{p.octave:#12x}  ({o:+d}, {layer}, {xi:+.3f})")

oct_ = np.array([unpack_octave(p.octave)[0] for p in kps_box])
fig, axs = plt.subplots(1, 3, figsize=(11, 2.8))
axs[0].hist(oct_, bins=np.arange(oct_.min(), oct_.max() + 2) - 0.5, rwidth=0.8); axs[0].set_title("октава (−1 = удвоенное)")
axs[1].hist([p.size for p in kps_box], bins=40); axs[1].set_title("size (px)"); axs[1].set_yscale("log")
axs[2].hist([p.response for p in kps_box], bins=40); axs[2].set_title("response = |D(x̂)|")
plt.tight_layout(); plt.show()
""")
code("""
# Влияние порогов: число точек на box.png
for ct in (0.02, 0.04, 0.09):
    n = len(cv2.SIFT_create(contrastThreshold=ct).detect(box, None))
    print(f"contrastThreshold={ct:<5} (эффективно {ct / 3:.4f}) → {n} точек")
for et in (3, 5, 10, 20):
    n = len(cv2.SIFT_create(edgeThreshold=et).detect(box, None))
    print(f"edgeThreshold={et:<3} → {n} точек")
""")

# ---------------------------------------------------------------- 2
md(f"""
## 2. Масштабное пространство (разд. 3, 3.2, 3.3 [конспекта]({K}))

$$L(x,y,\\sigma) = G(x,y,\\sigma) * I(x,y),\\qquad
D(x,y,\\sigma) = L(x,y,k\\sigma) - L(x,y,\\sigma) \\approx (k-1)\\,\\sigma^2\\nabla^2 G * I.$$

Устройство пирамиды (как в статье и OpenCV):

1. **Удвоение.** Вход считается уже размытым с $\\sigma_{{in}}=0.5$; после удвоения билинейной интерполяцией
   размытие становится $1.0$, досглаживаем до $\\sigma_0 = 1.6$: $\\sigma_{{diff}} = \\sqrt{{1.6^2 - 1^2}}$.
2. **Октава** — $s+3 = 6$ гауссовых изображений с $\\sigma_i = \\sigma_0 k^i$, $k = 2^{{1/s}}$.
   Каждое получается из предыдущего *инкрементальной* свёрткой
   $\\sigma_{{inc}} = \\sqrt{{\\sigma_i^2 - \\sigma_{{i-1}}^2}}$ (дисперсии гауссианов складываются).
3. **Следующая октава** начинается с изображения $i=s$ (у него $\\sigma = 2\\sigma_0$), прореженного в 2 раза.
4. **DoG** — разности соседних уровней: $s+2 = 5$ изображений на октаву; экстремумы ищутся в средних $s=3$.

Число октав, как в OpenCV: $\\mathrm{{round}}(\\log_2 \\min(H,W) - 2)$ для удвоенного изображения.
""")
code(defs("to_float01", "base_image", "num_octaves", "octave_sigmas",
          "gaussian_pyramid", "dog_pyramid", "build_scale_space"))
code("""
ss_a = build_scale_space(img_a)
s, sigma0 = ss_a["s"], ss_a["sigma"]
print("инкрементальные σ:", octave_sigmas(s, sigma0).round(4))
print("октав:", len(ss_a["gauss"]))
for o, g in enumerate(ss_a["gauss"]):
    # полное σ уровня i в пикселях ВХОДНОГО изображения: σ0·k^i·2^(o−1)
    sig_abs = sigma0 * 2 ** (np.arange(s + 3) / s) * 2.0 ** (o - 1)
    print(f"  октава {o}: {g.shape[1:]}  σ (px входа) = {sig_abs.round(2)}")

# проверка: инкрементальное размытие == прямое размытие базового уровня
base = ss_a["gauss"][0][0]
direct = cv2.GaussianBlur(base, (0, 0), np.sqrt((sigma0 * 2 ** (5 / 3)) ** 2 - sigma0 ** 2))
print("max |L_5(инкр.) − L_5(прямое)| =", np.abs(ss_a["gauss"][0][5] - direct).max())
""")
code("""
n_show = 4
fig, axs = plt.subplots(n_show, s + 3, figsize=(13, 7.5))
for o in range(n_show):
    for i in range(s + 3):
        axs[o, i].imshow(ss_a["gauss"][o][i], cmap="gray", vmin=0, vmax=1)
        axs[o, i].set_title(f"окт {o}, L{i}  σ={sigma0 * 2 ** (i / s) * 2.0 ** (o - 1):.2f}", fontsize=8)
        axs[o, i].axis("off")
fig.suptitle("Гауссова пирамида (масштаб изображений в каждой октаве свой)")
plt.tight_layout(); plt.show()

fig, axs = plt.subplots(n_show, s + 2, figsize=(12, 7.5))
for o in range(n_show):
    D = ss_a["dog"][o]
    v = np.percentile(np.abs(D), 99.5)
    for i in range(s + 2):
        axs[o, i].imshow(D[i], cmap="RdBu_r", vmin=-v, vmax=v)
        axs[o, i].set_title(f"окт {o}, D{i}" + (" (поиск)" if 1 <= i <= s else ""), fontsize=8)
        axs[o, i].axis("off")
fig.suptitle("DoG: красное — D>0 (тёмные пятна), синее — D<0 (светлые пятна)")
plt.tight_layout(); plt.show()
""")
md(f"""
Разобранный «на числах» пример DoG (ядра, отклик пятна, выбор масштаба) — в [конспекте, разд. 3.0а]({K})
и скрипте `../../rus/figures/sift/dog_example/dog_example.py`.

OpenCV не отдаёт свою пирамиду наружу, поэтому сверять её будем косвенно — по совпадению найденных точек
(раздел 3) и дескрипторов (раздел 5): если пирамида хоть немного отличается, точного совпадения не будет.
""")

# ---------------------------------------------------------------- 3
md(f"""
## 3. Детектор (разд. 3.1, 4, 4.1 [конспекта]({K}))

**Шаг 1 — экстремумы.** Отсчёт DoG — кандидат, если он не меньше (не больше) всех 26 соседей: 8 в своём слое и
по 9 в соседних по масштабу. Векторизуем: 26 сдвинутых срезов массива и поэлементные `max`/`min`.
Предварительный порог как в OpenCV: $\\lvert D\\rvert > 0.5\\cdot\\text{{порог}}$ (с округлением в шкале 0..255).

**Шаг 2 — уточнение (формулы (2)–(3)).** Тейлор до 2-го порядка по $\\mathbf x = (x, y, \\sigma)$:
$$D(\\mathbf x) \\approx D + \\nabla D^\\top \\mathbf x + \\tfrac12 \\mathbf x^\\top \\mathbf H \\mathbf x,\\qquad
\\hat{{\\mathbf x}} = -\\mathbf H^{{-1}}\\nabla D .$$
Производные — центральные разности. Если $\\lvert\\hat x_j\\rvert > 0.5$ по любой координате — переходим в соседний
отсчёт и повторяем (не более 5 итераций, иначе отброс).

**Шаг 3 — контраст.** $D(\\hat{{\\mathbf x}}) = D + \\tfrac12\\nabla D^\\top\\hat{{\\mathbf x}}$; отброс при
$\\lvert D(\\hat{{\\mathbf x}})\\rvert < $ порог (статья: 0.03; OpenCV: $0.04/s \\approx 0.0133$).

**Шаг 4 — края.** Гессиан 2×2 по $(x,y)$; при отношении главных кривизн $r$:
$$\\frac{{\\mathrm{{Tr}}(\\mathbf H)^2}}{{\\mathrm{{Det}}(\\mathbf H)}} < \\frac{{(r+1)^2}}{{r}},\\quad r = 10,$$
а при $\\mathrm{{Det}} \\le 0$ (седло) — отброс.

Уточнение тоже векторизовано: все кандидаты октавы решаются одним вызовом `np.linalg.solve` над стопкой 3×3.
""")
code(defs("find_extrema", "derivatives", "refine_extrema", "detect_keypoints"))
code("""
rows = []
for name, thr in [("OpenCV: 0.04/3", 0.04 / 3), ("статья: 0.03", 0.03)]:
    t = time.perf_counter()
    kp_det, st = detect_keypoints(ss_a, contrast_thr=thr, return_stats=True)
    dt = time.perf_counter() - t
    rows.append((name, st["candidates"], st["converged"], st["after_contrast"], st["after_edge"], dt))
    if thr < 0.02:
        kp_det_cv, st_cv = kp_det, st
print(f"{'порог контраста':<16} {'кандидаты':>10} {'сошлись':>8} {'контраст':>9} {'края':>6} {'время, с':>9}")
for r in rows:
    print(f"{r[0]:<16} {r[1]:>10} {r[2]:>8} {r[3]:>9} {r[4]:>6} {r[5]:>9.2f}")
print("по октавам (порог OpenCV):", np.bincount(kp_det_cv["octave"]))
""")
code("""
rej = st_cv["rejected"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
axs[0].imshow(img_a, cmap="gray")
axs[0].plot(rej["contrast"][:, 0], rej["contrast"][:, 1], ".", ms=3, color="tab:red", label=f"низкий контраст ({len(rej['contrast'])})")
axs[0].plot(rej["edge"][:, 0], rej["edge"][:, 1], ".", ms=3, color="tab:blue", label=f"край ({len(rej['edge'])})")
axs[0].plot(kp_det_cv["x"], kp_det_cv["y"], ".", ms=3, color="lime", label=f"оставлены ({len(kp_det_cv['x'])})")
axs[0].legend(loc="lower left", fontsize=7, markerscale=3); axs[0].axis("off")
axs[0].set_title("Кто и почему отброшен")
draw_kps(axs[1], img_a, np.c_[kp_det_cv["x"], kp_det_cv["y"]], kp_det_cv["size"], color="lime", lw=0.5)
axs[1].set_title("Итоговые точки: окружность радиуса σ (= size/2)")
plt.tight_layout(); plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 2.6))
rcl = detect_keypoints(ss_a, contrast_thr=0.04 / 3)
axs[0].hist(np.c_[rcl["x"] * 2.0 ** -(rcl["octave"] - 1) - rcl["c"], rcl["y"] * 2.0 ** -(rcl["octave"] - 1) - rcl["r"]],
            bins=30, label=["x̂ (столбец)", "ŷ (строка)"]); axs[0].legend(); axs[0].set_title("субпиксельное смещение")
axs[1].hist(rcl["xi"], bins=30); axs[1].set_title("субмасштабное смещение ξ̂ (в слоях)")
plt.tight_layout(); plt.show()
""")
md("""
### 3.1. Сравнение с OpenCV

OpenCV возвращает точки уже *после* назначения ориентации (одна точка с двумя ориентациями — две записи),
поэтому сравниваем уникальные позиции. Точка «совпала», если у OpenCV есть точка ближе 2 px с отношением
`size` в $[0.8, 1.25]$.
""")
code("""
kps_cv_a, des_cv_a = sift_cv.detectAndCompute(img_a, None)
P_own = np.unique(np.c_[kp_det_cv["x"], kp_det_cv["y"], kp_det_cv["size"]].round(3), axis=0)
P_cv = np.unique(np.array([(*p.pt, p.size) for p in kps_cv_a]).round(3), axis=0)


def nearest(P, Q):
    d2 = (P[:, :1] - Q[:, 0]) ** 2 + (P[:, 1:2] - Q[:, 1]) ** 2
    j = d2.argmin(1)
    return np.sqrt(d2[np.arange(len(P)), j]), j


d_own, j_own = nearest(P_own, P_cv)
ok_own = (d_own < 2) & (np.abs(np.log(P_own[:, 2] / P_cv[j_own, 2])) < np.log(1.25))
d_cv, j_cv = nearest(P_cv, P_own)
ok_cv = (d_cv < 2) & (np.abs(np.log(P_cv[:, 2] / P_own[j_cv, 2])) < np.log(1.25))
print(f"уникальных позиций: свои {len(P_own)}, OpenCV {len(P_cv)}")
print(f"доля своих точек, найденных и OpenCV: {ok_own.mean():.3f};  доля точек OpenCV, найденных нами: {ok_cv.mean():.3f}")
print(f"расхождение координат у совпавших: медиана {np.median(d_own[ok_own]):.2e} px, "
      f"95-й перцентиль {np.percentile(d_own[ok_own], 95):.2e} px, max {d_own[ok_own].max():.2e} px")
""")
md("""
Почти полное совпадение не случайно: мы воспроизвели все «невидимые» детали OpenCV — удвоение `INTER_LINEAR`,
досглаживание до σ=1.6 с учётом $\\sigma_{in}=0.5$, прореживание `[::2, ::2]`, отступ 5 px от края,
нестрогие неравенства при сравнении с соседями, порядок проверок при итерациях уточнения.
Остаточные расхождения (медиана 0, максимум ~$10^{-3}$ px) — следствие арифметики: OpenCV хранит изображение в шкале 0..255 и решает систему
3×3 в `float32`, мы — в $[0,1]$ и `float64`.

**Замечание о смещении.** `cv2.resize(INTER_LINEAR)` при удвоении сопоставляет пиксель $X$ удвоенного изображения
точке $X/2 - 0.25$ исходного, а координата точки потом просто делится на 2. Поэтому у OpenCV (и у нас) все координаты
сдвинуты примерно на +0.25 px. В OpenCV ≥ 4.8 это лечится флагом `enable_precise_upscale=True`.
""")

# ---------------------------------------------------------------- 4
md(f"""
## 4. Ориентация (разд. 5 [конспекта]({K}))

В слое $L$ с ближайшим масштабом считаем модуль и направление градиента:
$$m = \\sqrt{{(L_{{x+1,y}}-L_{{x-1,y}})^2 + (L_{{x,y+1}}-L_{{x,y-1}})^2}},\\qquad
\\theta = \\operatorname{{atan2}}(L_{{x,y+1}}-L_{{x,y-1}},\\; L_{{x+1,y}}-L_{{x-1,y}}).$$

* Гистограмма из **36 бинов** по 10°, вклад пикселя $= m\\cdot \\exp\\!\\big(-\\frac{{i^2+j^2}}{{2(1.5\\sigma)^2}}\\big)$, окно радиуса $3\\cdot1.5\\sigma$.
* Сглаживание круговым ядром $[1,4,6,4,1]/16$ (деталь OpenCV).
* Каждый локальный пик $\\ge 80\\%$ максимума даёт **отдельную** ключевую точку (в статье — ~15% точек).
* Положение пика уточняется параболой по трём бинам:
  $b^* = b + \\tfrac12\\,\\frac{{h_{{b-1}} - h_{{b+1}}}}{{h_{{b-1}} - 2h_b + h_{{b+1}}}}$.

**Ловушка с осями.** Внутри OpenCV считает $dy = L_{{y-1}} - L_{{y+1}}$ (ось $y$ *вверх*), а наружу отдаёт
`angle = 360 − θ` — это то же направление, но в координатах изображения (ось $y$ вниз). Мы делаем ровно так же.
""")
code(defs("gradient_polar", "gradient_pyramid", "window", "orientation_histogram",
          "histogram_peaks", "assign_orientations"))
code("""
grads_a = gradient_pyramid(ss_a["gauss"], s)
t = time.perf_counter()
kp_ori = assign_orientations(kp_det_cv, grads_a)
print(f"ориентации: {len(kp_det_cv['x'])} → {len(kp_ori['x'])} точек за {time.perf_counter() - t:.2f} с; "
      f"доля точек с >1 ориентацией: {(np.bincount(np.unique(kp_ori['x'], return_inverse=True)[1]) > 1).mean():.3f}")

# одна точка с двумя пиками
cnt = {}
for x in kp_ori["x"]:
    cnt[x] = cnt.get(x, 0) + 1
multi = [k for k in range(len(kp_det_cv["x"])) if cnt.get(kp_det_cv["x"][k], 0) >= 2 and kp_det_cv["sigma_oct"][k] > 2.5]
k = multi[0] if multi else int(np.argmax(kp_det_cv["response"]))
o, l, r, c, so = (kp_det_cv[f][k] for f in ("octave", "layer", "r", "c", "sigma_oct"))
mag, ori = grads_a[o][l]
hist = orientation_histogram(mag, ori, r, c, so)
peaks = histogram_peaks(hist)
rad = int(np.rint(4.5 * so))
m_w, a_w, ii, jj = window(mag, ori, r, c, rad)
G = ss_a["gauss"][o][l]

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].imshow(G[r - rad:r + rad + 1, c - rad:c + rad + 1], cmap="gray",
              extent=(-rad - 0.5, rad + 0.5, rad + 0.5, -rad - 0.5))
wgt = np.exp(-(ii ** 2 + jj ** 2) / (2 * (1.5 * so) ** 2)) * m_w
th = np.radians(a_w)
axs[0].quiver(jj, ii, wgt * np.cos(th), -wgt * np.sin(th), color="tab:orange", angles="xy", scale_units="xy",
              scale=wgt.max() / 2.5)   # −sin: на экране ось y вниз
axs[0].add_patch(Circle((0, 0), 1.5 * so, fill=False, color="cyan", ls="--"))
for a in peaks:
    axs[0].arrow(0, 0, rad * 0.8 * np.cos(np.radians(a)), rad * 0.8 * np.sin(np.radians(a)), color="red", width=0.15)
axs[0].set_title(f"окно точки (окт {o}, слой {l}, σ_oct={so:.2f}); пунктир — 1.5σ")
bins_deg = np.arange(36) * 10
axs[1].bar(bins_deg, hist, width=9, color="tab:gray")
axs[1].axhline(0.8 * hist.max(), color="red", ls="--", label="80% от максимума")
axs[1].set_xlabel("θ (ось y вверх), градусы"); axs[1].legend()
axs[1].set_title("Сглаженная гистограмма; пики → angle OpenCV = " + ", ".join(f"{a:.1f}°" for a in peaks))
plt.tight_layout(); plt.show()
""")
code("""
# сравнение углов с OpenCV на совпавших точках
xy_own = np.c_[kp_ori["x"], kp_ori["y"]]
xy_cv = np.array([p.pt for p in kps_cv_a])
ang_cv = np.array([p.angle for p in kps_cv_a])
d2 = ((xy_own[:, None, :] - xy_cv[None]) ** 2).sum(-1)
d2[d2 > 0.01 ** 2] = np.inf                                         # только «те же» точки
dang = np.abs((kp_ori["angle"][:, None] - ang_cv[None] + 180) % 360 - 180) + np.where(np.isfinite(d2), 0, np.inf)
best = dang.min(1)
best = best[np.isfinite(best)]
print(f"точек с парой в OpenCV: {len(best)} из {len(kp_ori['x'])}; |Δangle|: медиана {np.median(best):.2e}°, "
      f"доля < 0.5°: {(best < 0.5).mean():.3f}")
print(f"OpenCV: {len(kps_cv_a)} записей; мы: {len(kp_ori['x'])}")

top = np.argsort(-kp_ori["response"])[:150]
fig, ax = plt.subplots(figsize=(7, 5.5))
draw_kps(ax, img_a, xy_own[top], kp_ori["size"][top] * 2, kp_ori["angle"][top], color="yellow", lw=0.8)
ax.set_title("150 самых контрастных точек: окружность радиуса 2σ, отрезок — ориентация")
plt.tight_layout(); plt.show()
""")
md("""
**Почему записей у нас больше, чем у OpenCV?** Иногда два соседних кандидата после уточнения сходятся в одну и ту же
точку (сравните число точек детектора и число уникальных позиций в разделе 3.1). OpenCV удаляет такие дубликаты
(`KeyPointsFilter::removeDuplicatedSorted`), мы сделаем то же в `remove_duplicates` (раздел 5.2) — после этого число
точек совпадёт. Редкие расхождения угла (доли процента) — из-за того, что OpenCV считает угол через `cv::fastAtan2`
(точность ~0.3°) и экспоненту через `cv::exp`: пиксель на границе бина может попасть в соседний бин.
""")

# ---------------------------------------------------------------- 5
md(f"""
## 5. Дескриптор (разд. 6.1 [конспекта]({K}))

* Окно поворачивается на ориентацию точки: координаты пикселя относительно точки поворачиваются на $-\\theta$,
  направления градиентов — тоже ($\\theta_{{pix}} - \\theta$).
* Сетка $d\\times d = 4\\times4$ ячеек, ширина ячейки $3\\sigma$ пикселей октавы (константа OpenCV
  `SIFT_DESCR_SCL_FCTR = 3`; в статье — «4×4 отсчёта на масштабе точки»), т.е. окно ≈ $12\\sigma$.
* Вклад пикселя $m\\cdot w$, $w = \\exp\\!\\big(-\\frac{{u^2+v^2}}{{2(d/2)^2}}\\big)$ ($u,v$ — в ячейках):
  σ гауссиана = **половина ширины окна**.
* **Трилинейная интерполяция**: вклад делится между 2×2 соседними ячейками и 2 соседними бинами ориентации
  с весами $(1-d_r)$/$d_r$ и т.д. — дескриптор не «прыгает» при сдвиге точки на долю ячейки.
* $4\\cdot4\\cdot8 = 128$ чисел → нормировка к 1 → обрезка компонент на **0.2** (подавление больших градиентов
  от нелинейной освещённости) → повторная нормировка.
* OpenCV в конце умножает на 512 и округляет до `uchar` — функция `to_opencv_scale`.
""")
code(defs("sift_descriptor", "normalize_clip", "compute_descriptors", "to_opencv_scale", "root_sift"))
code("""
# дескриптор одной точки: окно на изображении октавы и 4×4 «розы» направлений
k = int(np.argmax(kp_ori["sigma_oct"] * (kp_ori["response"] > np.median(kp_ori["response"]))))
o, l, r, c, so, ang = (kp_ori[f][k] for f in ("octave", "layer", "r", "c", "sigma_oct", "angle"))
mag, ori = grads_a[o][l]
d_raw = sift_descriptor(mag, ori, r, c, ang, so, clip=1.0)   # clip=1 → без обрезки
d_fin = sift_descriptor(mag, ori, r, c, ang, so)

G = ss_a["gauss"][o][l]
hw = 3 * so
a = np.radians(ang)
ex, ey = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])   # оси окна (ось y вниз)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
R = int(hw * 3.2)
axs[0].imshow(G[r - R:r + R + 1, c - R:c + R + 1], cmap="gray", extent=(c - R - .5, c + R + .5, r + R + .5, r - R - .5))
for t_ in np.linspace(-2, 2, 5) * hw:
    for u0, u1 in ((ex * t_ - ey * 2 * hw, ex * t_ + ey * 2 * hw), (ey * t_ - ex * 2 * hw, ey * t_ + ex * 2 * hw)):
        axs[0].plot([c + u0[0], c + u1[0]], [r + u0[1], r + u1[1]], color="yellow", lw=0.8)
axs[0].arrow(c, r, *(ex * 2 * hw), color="red", width=0.3)
axs[0].set_title(f"окно 4×4 ячейки по {hw:.1f} px, повёрнуто на {ang:.1f}°")
axs[0].axis("off")
# розы: бин b — направление ang + 45°·b относительно оси окна (в системе «ось y вверх» углы идут против часовой)
H8 = d_fin.reshape(4, 4, 8)
axs[1].set_xlim(-0.5, 3.5); axs[1].set_ylim(3.5, -0.5); axs[1].set_aspect("equal")
for i in range(4):
    for j in range(4):
        axs[1].add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, color="gray", lw=0.5))
        for b in range(8):
            phi = np.radians(45 * b)
            L_ = H8[i, j, b] / H8.max() * 0.48
            axs[1].plot([j, j + L_ * np.cos(phi)], [i, i - L_ * np.sin(phi)], color="tab:blue", lw=1.5)
axs[1].set_title("8 направлений в каждой из 16 ячеек\\n(в системе окна)"); axs[1].axis("off")
axs[2].bar(np.arange(128), d_raw, color="tab:gray", label="до обрезки (норм.)")
axs[2].bar(np.arange(128), d_fin, color="tab:blue", alpha=0.7, label="после обрезки 0.2 и перенормировки")
axs[2].axhline(0.2, color="red", ls="--", lw=0.8)
axs[2].set_xlabel("индекс = (строка·4 + столбец)·8 + бин"); axs[2].legend(fontsize=8)
axs[2].set_title(f"128-мерный вектор: ‖d‖ = {np.linalg.norm(d_fin):.4f}, max = {d_fin.max():.3f}")
plt.tight_layout(); plt.show()
""")
md("""
### 5.1. Ключевой тест: свой дескриптор на точках OpenCV

Чтобы проверить *только* дескриптор (без влияния детектора), берём ключевые точки OpenCV, восстанавливаем по полю
`octave` номер октавы/слоя, позицию в октаве $\\mathbf{p}/2^{o}$ и масштаб $\\sigma_{oct} = \\tfrac12\\,\\mathrm{size}/2^{o}$,
и считаем наш дескриптор на нашей пирамиде. Сравниваем с `des` из `sift_cv.compute` косинусной близостью.
""")
code(defs("kp_from_cv"))
code("""
kp_cvfmt = kp_from_cv(kps_cv_a, s, sigma0, ss_a["first_octave"])
t = time.perf_counter()
d_own_on_cv = compute_descriptors(kp_cvfmt, grads_a)
t_desc = time.perf_counter() - t
d_cv_n = des_cv_a / np.linalg.norm(des_cv_a, axis=1, keepdims=True)
cos_sim = (d_own_on_cv * d_cv_n).sum(1)
diff255 = np.abs(to_opencv_scale(d_own_on_cv) - des_cv_a)
print(f"{len(kps_cv_a)} точек, {t_desc:.2f} с ({t_desc / len(kps_cv_a) * 1e3:.2f} мс/точку)")
print(f"косинусная близость: среднее {cos_sim.mean():.6f}, медиана {np.median(cos_sim):.6f}, "
      f"1-й перцентиль {np.percentile(cos_sim, 1):.6f}, min {cos_sim.min():.6f}")
print(f"в шкале OpenCV (uchar): доля дескрипторов, совпавших с точностью ±1: {np.all(diff255 <= 1, 1).mean():.4f}; "
      f"доля точно равных компонент: {(diff255 == 0).mean():.4f}")

# Типичные ошибки при переносе дескриптора — что будет, если ошибиться в соглашениях
def variant(**ch):
    kp2 = dict(kp_cvfmt)
    kp2.update(ch)
    return (compute_descriptors(kp2, grads_a) * d_cv_n).sum(1)


variants = {
    "правильно": cos_sim,
    "угол без перевода 360−θ (знак поворота)": variant(angle=(360 - kp_cvfmt["angle"]) % 360),
    "σ = size/2^o (забыли /2: size — диаметр)": variant(sigma_oct=kp_cvfmt["sigma_oct"] * 2),
    "без поворота окна (angle = 0)": variant(angle=np.zeros(len(kp_cvfmt["angle"]))),
    "соседний слой (не тот уровень пирамиды)": variant(layer=np.where(kp_cvfmt["layer"] < s, kp_cvfmt["layer"] + 1, kp_cvfmt["layer"] - 1)),
}
fig, ax = plt.subplots(figsize=(9, 3.2))
for name, v in variants.items():
    print(f"  {name:<45} медиана cos = {np.median(v):.4f}")
    ax.hist(v, bins=np.linspace(0, 1, 101), histtype="step", lw=1.5, label=name, density=True)
ax.set_yscale("log"); ax.set_xlabel("cos(свой, OpenCV)"); ax.legend(fontsize=7, loc="upper left")
ax.set_title("Распределение косинусной близости дескрипторов на одних и тех же точках")
plt.tight_layout(); plt.show()
""")
md("""
Результат — практически побитовое совпадение (расхождение не больше 1 уровня `uchar` из-за `fastAtan2` и округления).
Ошибки в соглашениях (знак угла, «диаметр vs радиус», уровень пирамиды) сразу видны по падению близости —
это хороший способ отлаживать собственную реализацию.

**Сопоставление на одних и тех же точках.** Проверим, что наш дескриптор не хуже и в деле: на синтетической паре
берём точки OpenCV в обоих изображениях и сопоставляем (одним и тем же `cv2.BFMatcher`, ratio 0.8) дескрипторы OpenCV
и наши. Верность — по истинной гомографии.
""")
code("""
kps_cv_b, des_cv_b = sift_cv.detectAndCompute(img_b, None)
ss_b = build_scale_space(img_b)
grads_b = gradient_pyramid(ss_b["gauss"], s)
d_own_on_cv_b = compute_descriptors(kp_from_cv(kps_cv_b), grads_b)
pa = np.array([p.pt for p in kps_cv_a]); pb = np.array([p.pt for p in kps_cv_b])
bf = cv2.BFMatcher(cv2.NORM_L2)
for name, DA, DB in [("дескрипторы OpenCV", des_cv_a, des_cv_b), ("наши дескрипторы", d_own_on_cv, d_own_on_cv_b)]:
    mm = [m for m, n in bf.knnMatch(DA.astype(np.float32), DB.astype(np.float32), k=2) if m.distance < 0.8 * n.distance]
    q = np.c_[pa[[m.queryIdx for m in mm]], np.ones(len(mm))] @ H_gt.T
    err = np.linalg.norm(q[:, :2] / q[:, 2:] - pb[[m.trainIdx for m in mm]], axis=1)
    print(f"{name:<20}: {len(mm)} совпадений после ratio 0.8, верных {np.sum(err < 3)} (precision {np.mean(err < 3):.3f})")
""")
md("""
### 5.2. Сборка конвейера

Склеиваем шаги 2–5 в одну функцию `sift_detect_and_compute` — аналог `sift.detectAndCompute`. Как и OpenCV,
удаляем точные дубликаты и (опционально) оставляем `n_features` точек с наибольшим откликом.
""")
code(defs("remove_duplicates", "top_n", "sift_detect_and_compute", "kp_to_cv", "kp_xy"))
code("""
t = time.perf_counter()
kp_a, d_a = sift_detect_and_compute(img_a)
t_own = time.perf_counter() - t
t = time.perf_counter()
kps_tmp, _ = sift_cv.detectAndCompute(img_a, None)
t_cv = time.perf_counter() - t
print(f"свой SIFT: {len(kp_a['x'])} точек за {t_own:.2f} с;  OpenCV: {len(kps_tmp)} точек за {t_cv * 1e3:.0f} мс "
      f"(в {t_own / t_cv:.0f} раз быстрее — C++ и SIMD)")
""")

# ---------------------------------------------------------------- 6
md(f"""
## 6. Сопоставление (разд. 7.1–7.2 [конспекта]({K}))

**Полный перебор.** Квадраты расстояний между всеми парами — одним матричным умножением:
$$\\lVert \\mathbf a - \\mathbf b\\rVert^2 = \\lVert\\mathbf a\\rVert^2 + \\lVert\\mathbf b\\rVert^2 - 2\\,\\mathbf a^\\top\\mathbf b .$$

**Тест отношения (ratio test).** Принимаем ближайшего соседа, только если он заметно ближе второго:
$d_1/d_2 < 0.8$. Второй сосед оценивает «плотность ложных совпадений» в этой области пространства дескрипторов
(рис. 11 статьи: отсекает ~90% ложных и <5% верных).

**Взаимная проверка (cross-check).** $i\\to j$ и $j \\to i$ должны быть ближайшими друг для друга.
""")
code(defs("l2_distances", "knn2", "ratio_match", "mutual_match"))
code("""
kp_b, d_b = sift_detect_and_compute(img_b)
p_a, p_b = kp_xy(kp_a), kp_xy(kp_b)
print(f"img_a: {len(p_a)} точек, img_b: {len(p_b)} точек")

# 1) корректность: сравнение с наивным перебором и с cv2.BFMatcher
idx, dist = knn2(d_a, d_b)
naive = np.array([np.argsort([np.linalg.norm(d_a[i] - d_b[j]) for j in range(len(d_b))])[:2] for i in range(50)])
print("совпадение с наивным перебором (50 строк):", np.array_equal(naive, idx[:50]))
knn_cv = bf.knnMatch(d_a, d_b, k=2)
same = np.mean([m.trainIdx == idx[i, 0] and n.trainIdx == idx[i, 1] for i, (m, n) in enumerate(knn_cv)])
print(f"совпадение (1-й и 2-й сосед) с cv2.BFMatcher.knnMatch: {same:.4f};  "
      f"max |Δd1| = {np.max(np.abs(dist[:, 0] - [m.distance for m, n in knn_cv])):.2e}")
""")
code("""
# 2) аналог рис. 11 статьи: распределение d1/d2 для верных и неверных ближайших соседей
from_gt = np.c_[p_a, np.ones(len(p_a))] @ H_gt.T
from_gt = from_gt[:, :2] / from_gt[:, 2:]
nn_correct = np.linalg.norm(from_gt - p_b[idx[:, 0]], axis=1) < 3
ratio = dist[:, 0] / dist[:, 1]
bins = np.linspace(0, 1, 41)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(ratio[nn_correct], bins, density=True, histtype="step", lw=2, label=f"верный NN ({nn_correct.sum()})")
ax.hist(ratio[~nn_correct], bins, density=True, histtype="step", lw=2, ls="--", label=f"неверный NN ({(~nn_correct).sum()})")
ax.axvline(0.8, color="k", lw=0.8)
ax.set_xlabel("d1 / d2"); ax.set_ylabel("плотность"); ax.legend()
ax.set_title("Аналог рис. 11 Лоу: отношение расстояний до 1-го и 2-го соседа")
plt.tight_layout(); plt.show()
print(f"порог 0.8 отбрасывает {np.mean(ratio[~nn_correct] >= 0.8):.1%} неверных и {np.mean(ratio[nn_correct] >= 0.8):.1%} верных")
""")
code("""
# 3) стратегии отбора: NN / ratio / cross-check
def report(name, M):
    ok = np.linalg.norm(from_gt[M[:, 0]] - p_b[M[:, 1]], axis=1) < 3
    print(f"{name:<24} совпадений {len(M):5d}, верных {ok.sum():5d}, precision {ok.mean():.3f}")


report("ближайший сосед", np.c_[np.arange(len(d_a)), idx[:, 0]])
M_ratio, _, _, _ = ratio_match(d_a, d_b, 0.8)
report("ratio 0.8", M_ratio)
M_mut = mutual_match(d_a, d_b)
report("cross-check", M_mut)
both = np.array(sorted(set(map(tuple, M_ratio)) & set(map(tuple, M_mut)))).reshape(-1, 2)
report("ratio 0.8 + cross-check", both)
""")
code("""
# 4) скорость: свой перебор, cv2.BFMatcher, FLANN (KD-деревья) — на увеличенном наборе
rep = 4                                             # «база» побольше: дескрипторы с шумом
A = d_a
B = np.vstack([d_b] + [d_b + np.random.default_rng(i).normal(0, 0.02, d_b.shape).astype(np.float32) for i in range(rep - 1)])
t = time.perf_counter(); i_own, _ = knn2(A, B); t_own = time.perf_counter() - t
t = time.perf_counter(); k_bf = bf.knnMatch(A, B, k=2); t_bf = time.perf_counter() - t
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))   # 1 = FLANN_INDEX_KDTREE
t = time.perf_counter(); k_fl = flann.knnMatch(A, B, k=2); t_fl = time.perf_counter() - t
agree_bf = np.mean([m.trainIdx == i_own[q, 0] for q, (m, n) in enumerate(k_bf)])
agree_fl = np.mean([m.trainIdx == i_own[q, 0] for q, (m, n) in enumerate(k_fl)])
print(f"запросов {len(A)}, база {len(B)}")
print(f"свой numpy-перебор : {t_own * 1e3:7.1f} мс")
print(f"cv2.BFMatcher      : {t_bf * 1e3:7.1f} мс, тот же 1-й сосед: {agree_bf:.4f}")
print(f"FLANN KD-tree      : {t_fl * 1e3:7.1f} мс, тот же 1-й сосед: {agree_fl:.4f} (приближённый поиск, ср. BBF в разд. 7.2)")
""")

# ---------------------------------------------------------------- 7
md("""
## 7. Геометрическая проверка: RANSAC + гомография

В статье (разд. 7.3–7.4) Лоу кластеризует совпадения преобразованием Хафа и решает аффинную задачу МНК.
Для плоских сцен и для визуального позиционирования сегодня стандарт — **RANSAC** с гомографией
(см. [RANSAC_конспект](../../rus/RANSAC_конспект.md)).

**Нормализованный DLT.** Для пары $\\mathbf p \\leftrightarrow \\mathbf p'$ из $\\mathbf p' \\times H\\mathbf p = 0$
получаем две строки матрицы $A$ (2n×9); $\\mathbf h$ — правый сингулярный вектор $A$ с наименьшим
сингулярным числом. Перед этим точки нормируются (Хартли): центр в 0, среднее расстояние $\\sqrt2$ —
без этого задача плохо обусловлена.

**RANSAC.** Повторяем: случайные 4 пары → $H$ → inlier'ы с ошибкой $\\lVert H\\mathbf p - \\mathbf p'\\rVert < 3$ px.
Число итераций адаптивно:
$$N = \\frac{\\log(1-p)}{\\log(1-w^4)},$$
$w$ — доля inlier'ов лучшей гипотезы, $p = 0.99$. В конце $H$ переоценивается по всем inlier'ам.
""")
code(defs("project", "normalize_points", "dlt_homography", "ransac_homography", "corner_error"))
code("""
# тест на синтетических точках: 200 пар, шум 0.5 px, 40% выбросов
rng = np.random.default_rng(1)
P1 = rng.uniform(0, 400, (200, 2))
P2 = project(H_gt, P1) + rng.normal(0, 0.5, (200, 2))
out = rng.random(200) < 0.4
P2[out] = rng.uniform(0, 400, (out.sum(), 2))
H_r, mask_r, it_r = ransac_homography(P1, P2)
print(f"итераций: {it_r} (адаптивно), inlier'ов {mask_r.sum()} (истинных {(~out).sum()}), "
      f"угловая ошибка {corner_error(H_r, H_gt, 400, 320):.3f} px")
""")
code("""
pa_m, pb_m = p_a[M_ratio[:, 0]], p_b[M_ratio[:, 1]]
rows = []
t = time.perf_counter(); H_own, mask_own, it_own = ransac_homography(pa_m, pb_m, 3.0); dt = time.perf_counter() - t
rows.append(("свой RANSAC", H_own, mask_own, dt, it_own))
for name, flag in [("cv2 RANSAC", cv2.RANSAC), ("cv2 USAC_MAGSAC", cv2.USAC_MAGSAC)]:
    t = time.perf_counter(); Hc, mc = cv2.findHomography(pa_m, pb_m, flag, 3.0); dt = time.perf_counter() - t
    rows.append((name, Hc, mc.ravel().astype(bool), dt, None))
gt_ok = np.linalg.norm(from_gt[M_ratio[:, 0]] - pb_m, axis=1) < 3
print(f"совпадений после ratio test: {len(M_ratio)}, из них верных по H_gt: {gt_ok.sum()}")
print(f"{'метод':<16} {'inliers':>8} {'доля':>6} {'верных среди inl.':>18} {'ошибка углов, px':>17} {'время, мс':>10}")
for name, Hm, mk, dt, it in rows:
    print(f"{name:<16} {mk.sum():>8} {mk.mean():>6.3f} {np.mean(gt_ok[mk]):>18.3f} "
          f"{corner_error(Hm, H_gt, 400, 320):>17.3f} {dt * 1e3:>10.1f}" + (f"   (итераций: {it})" if it else ""))
""")
code("""
ka, kb = kp_to_cv(kp_a), kp_to_cv(kp_b)
dm = [cv2.DMatch(int(i), int(j), 0) for i, j in M_ratio]
sel = np.nonzero(mask_own)[0][::4]                       # каждый 4-й inlier, чтобы было видно линии
vis = cv2.drawMatches(img_a, ka, img_b, kb, [dm[i] for i in sel], None, matchColor=(0, 255, 0),
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show(vis, titles=[f"inlier'ы своего RANSAC (показан каждый 4-й из {mask_own.sum()})"], size=9)

warped = cv2.warpPerspective(img_a, H_own, (img_b.shape[1], img_b.shape[0]))
overlay = np.dstack([img_b, warped, img_b])                 # совпадение → серый, расхождение → цветное
diff = np.abs(warped.astype(int) - img_b.astype(int)).astype(np.uint8)
diff[warped == 0] = 0
show(overlay[..., ::-1], diff, titles=["наложение: warp(img_a, H_own) (зелёный) и img_b (пурпурный)",
                                       "|warp(img_a) − img_b| (остаток — шум)"], size=5)
""")
md("""
### 7.1. Реальные пары: объект в сцене и смена ракурса

Здесь истинной гомографии нет — проверяем визуально: контур объекта, перенесённый найденной $H$.
""")
code("""
def locate(img1, img2, use_own=True, n_features=0):
    if use_own:
        k1, d1 = sift_detect_and_compute(img1, n_features)
        k2, d2 = sift_detect_and_compute(img2, n_features)
        p1, p2 = kp_xy(k1), kp_xy(k2)
        M, *_ = ratio_match(d1, d2, 0.8)
        H, mask, _ = ransac_homography(p1[M[:, 0]], p2[M[:, 1]], 3.0)
    else:
        k1, d1 = sift_cv.detectAndCompute(img1, None)
        k2, d2 = sift_cv.detectAndCompute(img2, None)
        p1, p2 = kp_xy(k1), kp_xy(k2)
        mm = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.8 * n.distance]
        M = np.array([(m.queryIdx, m.trainIdx) for m in mm]).reshape(-1, 2)
        H, mask = cv2.findHomography(p1[M[:, 0]], p2[M[:, 1]], cv2.RANSAC, 3.0)
        mask = mask.ravel().astype(bool)
    return H, M, mask


graf1s = cv2.resize(graf1, (400, 320), interpolation=cv2.INTER_AREA)
graf3s = cv2.resize(graf3, (400, 320), interpolation=cv2.INTER_AREA)
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for row, (i1, i2, nm) in enumerate([(box, scene, "box → box_in_scene"), (graf1s, graf3s, "graf1 → graf3")]):
    for col, own in enumerate([True, False]):
        t = time.perf_counter()
        H, M, mask = locate(i1, i2, own)
        dt = time.perf_counter() - t
        h1, w1 = i1.shape
        cr = project(H, np.array([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]], float))
        axs[row, col].imshow(i2, cmap="gray")
        axs[row, col].add_patch(Polygon(cr, fill=False, color="lime", lw=2))
        axs[row, col].axis("off")
        axs[row, col].set_title(f"{nm}, {'свой' if own else 'OpenCV'}: {len(M)} совп., {mask.sum()} inl. ({dt:.1f} с)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 8
md(f"""
## 8. Оценка и эксперименты

На парах с известной $H$ считаем (ср. «Ключевые численные результаты» в [конспекте]({K})):

* **повторяемость** — доля точек общей видимой области, у которых во втором изображении есть точка ближе $\\varepsilon=3$ px
  после переноса $H$ (знаменатель — меньшее из чисел видимых точек);
* **precision** — доля верных среди принятых совпадений; **recall** — верные принятые / число точек, у которых
  вообще есть истинный партнёр;
* **доля inlier'ов** после RANSAC и **ошибка углов** найденной $H$.
""")
code(defs("inside", "repeatability", "match_correctness", "n_possible"))
code("""
orb = cv2.ORB_create(nfeatures=1500)
bf_ham = cv2.BFMatcher(cv2.NORM_HAMMING)


def features(method, img, n=1500):
    if method == "свой SIFT":
        k, d = sift_detect_and_compute(img, n)
        return kp_xy(k), d
    det = cv2.SIFT_create(nfeatures=n) if method == "OpenCV SIFT" else orb
    k, d = det.detectAndCompute(img, None)
    return kp_xy(k), d


def knn_ratios(method, d1, d2):
    m = bf_ham if method == "OpenCV ORB" else bf
    kn = [x for x in m.knnMatch(d1, d2, k=2) if len(x) == 2]
    M = np.array([(a.queryIdx, a.trainIdx) for a, b in kn]).reshape(-1, 2)
    r = np.array([a.distance / max(b.distance, 1e-9) for a, b in kn])
    return M, r


METHODS = ["свой SIFT", "OpenCV SIFT", "OpenCV ORB"]
feats_a = {m: features(m, img_a) for m in METHODS}
fig, ax = plt.subplots(figsize=(6, 4))
for m in METHODS:
    p1, d1 = feats_a[m]
    p2, d2 = features(m, img_b)
    M, r = knn_ratios(m, d1, d2)
    ok = match_correctness(p1, p2, M, H_gt)
    npos = n_possible(p1, p2, H_gt)
    th = np.linspace(0.4, 1.0, 25)
    prec = [ok[r < t].mean() if (r < t).any() else 1 for t in th]
    rec = [ok[r < t].sum() / npos for t in th]
    ax.plot(rec, prec, ".-", label=m)
    j = np.argmin(np.abs(th - 0.8))
    ax.plot(rec[j], prec[j], "ko", ms=5)
    sel = r < 0.8
    Hm, mk = cv2.findHomography(p1[M[sel, 0]], p2[M[sel, 1]], cv2.RANSAC, 3.0)
    print(f"{m:<12} точек {len(p1)}/{len(p2)}, повторяемость {repeatability(p1, p2, H_gt, img_a.shape, img_b.shape):.3f}, "
          f"@0.8: precision {prec[j]:.3f}, recall {rec[j]:.3f}, inlier'ов {mk.mean():.3f}, "
          f"ошибка углов {corner_error(Hm, H_gt, 400, 320):.2f} px")
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Precision–recall при изменении порога ratio (0.4…1.0); ● — 0.8")
plt.tight_layout(); plt.show()
""")
md("""
### 8.1. Устойчивость к повороту и масштабу

Для каждого преобразования синтезируем $I_2 = \\mathrm{warp}(I_1, H)$ (+шум σ=2) и считаем метрики для трёх методов:
свой SIFT, OpenCV SIFT и OpenCV ORB (бинарный дескриптор, расстояние Хэмминга; см. `../../rus/ORB_конспект.md`).
Число точек ограничено 1500 (для своего SIFT — top-N по отклику), ratio = 0.8, RANSAC — `cv2.findHomography`.
Гомография считается найденной, если ошибка углов < 5 px.
""")
code("""
def evaluate(method, H, noise=2.0):
    img2 = make_pair(img_a, H, noise=noise)
    p1, d1 = feats_a[method]
    p2, d2 = features(method, img2)
    res = dict(rep=repeatability(p1, p2, H, img_a.shape, img2.shape), prec=0.0, ncorr=0, cerr=np.inf)
    if len(p2) < 2:
        return res
    M, r = knn_ratios(method, d1, d2)
    M = M[r < 0.8]
    ok = match_correctness(p1, p2, M, H)
    res.update(prec=ok.mean() if len(M) else 0.0, ncorr=int(ok.sum()))
    if len(M) >= 4:
        Hm, _ = cv2.findHomography(p1[M[:, 0]], p2[M[:, 1]], cv2.RANSAC, 3.0)
        res["cerr"] = corner_error(Hm, H, 400, 320) if Hm is not None else np.inf
    return res


angles = [0, 15, 30, 45, 60, 90, 120, 150, 180]
scales = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
t = time.perf_counter()
R_rot = {m: [evaluate(m, synthetic_homography(400, 320, angle=a)) for a in angles] for m in METHODS}
R_sc = {m: [evaluate(m, synthetic_homography(400, 320, scale=sc)) for sc in scales] for m in METHODS}
print(f"эксперименты: {time.perf_counter() - t:.1f} с")

fig, axs = plt.subplots(2, 4, figsize=(15, 6.5))
for row, (R, xs, xl) in enumerate([(R_rot, angles, "поворот, °"), (R_sc, scales, "масштаб")]):
    for col, (key, yl) in enumerate([("rep", "повторяемость"), ("prec", "precision @ ratio 0.8"),
                                     ("ncorr", "верных совпадений"), ("cerr", "ошибка углов H, px")]):
        ax = axs[row, col]
        for m, st in zip(METHODS, ["o-", "s--", "^-"]):
            ax.plot(xs, [min(v[key], 1e3) if v[key] > 0 else np.nan for v in R[m]], st, label=m, ms=4)
        ax.set_xlabel(xl); ax.set_title(yl); ax.grid(alpha=0.3)
        if row == 1:
            ax.set_xscale("log", base=2); ax.set_xticks(scales); ax.set_xticklabels(scales)
        if key == "cerr":
            ax.set_yscale("log"); ax.axhline(5, color="k", lw=0.7, ls=":")
        if key == "ncorr":
            ax.set_yscale("log")
axs[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")
code("""
print(f"{'':<12}" + "".join(f"{a:>7}°" for a in angles))
for m in METHODS:
    print(f"{m:<12}" + "".join(f"{v['ncorr']:>8}" for v in R_rot[m]) + "   ← верных совпадений, поворот")
print(f"{'':<12}" + "".join(f"{sc:>8}" for sc in scales))
for m in METHODS:
    print(f"{m:<12}" + "".join(f"{v['ncorr']:>8}" for v in R_sc[m]) + "   ← верных совпадений, масштаб")
""")
md("""
### 8.2. Откуда систематическая ошибка ~0.5 px при повороте?

Ошибка углов растёт с углом поворота даже без шума. Причина — смещение координат на +0.25 px от удвоения
`INTER_LINEAR` (раздел 3.1): при повороте на 180° смещения в двух изображениях направлены «в разные стороны»
относительно сцены и складываются в ~0.5 px. Проверим на OpenCV: средняя невязка $H\\mathbf p_1 - \\mathbf p_2$ по верным
совпадениям (без шума) с флагом `enable_precise_upscale` и без него.
""")
code("""
for precise in (False, True):
    det = cv2.SIFT_create(enable_precise_upscale=precise)
    k1, d1 = det.detectAndCompute(img_a, None)
    for ang in (0, 90, 180):
        Hr = synthetic_homography(400, 320, angle=ang)
        k2, d2 = det.detectAndCompute(make_pair(img_a, Hr), None)
        mm = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.8 * n.distance]
        p1, p2 = kp_xy(k1)[[m.queryIdx for m in mm]], kp_xy(k2)[[m.trainIdx for m in mm]]
        e = project(Hr, p1) - p2
        ok = np.linalg.norm(e, axis=1) < 3
        Hm, _ = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        print(f"precise_upscale={precise!s:<5} поворот {ang:>3}°: средняя невязка (x, y) = {e[ok].mean(0).round(3)} px, "
              f"ошибка углов H = {corner_error(Hm, Hr, 400, 320):.3f} px")
""")
md("""
**Что видно на графиках** (числа — из выполненного ноутбука; на вашей машине они должны совпасть, т.к. всё детерминировано).

* **Свой SIFT и OpenCV SIFT идут вплотную** во всех экспериментах: это одна и та же реализация. Различия в единицы
  совпадений — из-за `fastAtan2` и порядка отбора top-N.
* **Поворот.** Точность SIFT после ratio test 0.95–0.98 при любом угле. Число верных совпадений падает с ~950 (0°) до
  ~600–700 в основном потому, что при повороте кадра одинакового размера срезаются углы и общая область меньше, плюс
  интерполяция растра; при 90° и 180° поворот растра точный — совпадений больше. Ошибка $H$ растёт до ~0.7 px — это
  систематический сдвиг от удвоения (раздел 8.2), а не «неинвариантность».
* **Масштаб.** SIFT находит гомографию во всём диапазоне 0.25…2 (ошибка углов < 0.6 px), повторяемость держится на
  уровне 0.6–0.75. Число верных совпадений падает при уменьшении — у мелких деталей просто нет аналогов, а при ×2
  общая область — лишь центральная четверть кадра.
* **ORB** (1500 точек FAST против ~1100, которые SIFT находит на кадре 400×320) даёт *больше* верных совпадений при
  повороте и умеренном масштабе, но гомография по ним менее точна (0.5–2 px против 0.03–0.7 px у SIFT): угол — по
  центроиду интенсивности, положение — на уровне пикселя пирамиды без субпиксельного уточнения. На краях диапазона
  масштаба (0.25 и 2) повторяемость ORB падает до 0.2–0.5, а ошибка $H$ доходит до 4–5 px: пирамида ORB
  (8 уровней × 1.2 ≈ ×3.6) покрывает меньший диапазон, чем октавы SIFT.
* Значит, «SIFT лучше ORB» — не абсолютное утверждение: на этом текстурном изображении при небольших изменениях масштаба
  ORB конкурентоспособен и в десятки раз быстрее; преимущество SIFT — точность локализации и широкий диапазон масштабов.
""")

# ---------------------------------------------------------------- 9
md(f"""
## 9. Упражнения

1. **Число слоёв $s$ и σ.** Постройте пирамиду с $s=2, 4, 5$ и $\\sigma_0 = 1.2, 2.0$. Как меняется число точек и
   повторяемость (раздел 8)? Сравните с рис. 3–4 статьи (разд. 3.2–3.3 [конспекта]({K})).
   *Подсказка:* `sift_detect_and_compute(img, s=4)`; не забудьте, что порог контраста OpenCV-стиля — `0.04/s`.
2. **Порог контраста как в статье.** Задайте `contrast_thr=0.03` (и `cv2.SIFT_create(contrastThreshold=0.09)`). Сколько
   точек остаётся, как меняются precision/recall?
3. **Тест краёв.** Постройте зависимость числа точек и повторяемости от $r \\in \\{{2, 5, 10, 20, 50\\}}$. Нарисуйте
   отброшенные «краевые» точки (`stats["rejected"]["edge"]`) — лежат ли они на контурах?
4. **RootSIFT.** Примените `root_sift` к дескрипторам обоих изображений и повторите кривую precision–recall
   из раздела 8. *Подсказка:* после RootSIFT расстояние L2 ≈ расстояние Хеллингера; ratio test работает так же.
5. **Размер сетки дескриптора.** Посчитайте дескрипторы с `d=2` и `d=8` (`compute_descriptors(kp, grads, d=...)`).
   Сравните с рис. 8 статьи (разд. 6.2): 4×4×8 — компромисс различимости и устойчивости.
   *Подсказка:* ширина окна растёт как $d\\cdot3\\sigma$ — при `d=8` точки у края будут обрезаны.
6. **Без ratio test.** Замените ratio test фиксированным порогом на $d_1$ (например, 0.3, 0.4, 0.5 для
   нормированных дескрипторов). Нарисуйте precision–recall и сравните с ratio test — почему глобальный порог хуже?
7. **Без трилинейной интерполяции / без обрезки 0.2.** Уберите распределение по соседним бинам (всё в ближайший)
   или поставьте `clip=1.0`. Как падает precision на паре с перспективой и шумом? На паре с изменением яркости
   (`img_b ** 0.5`)?
8. **Свои снимки.** Сделайте два снимка плоской сцены (страница книги, плакат) под разными углами, загрузите через
   `cv2.imread(..., cv2.IMREAD_GRAYSCALE)` и прогоните раздел 7.1. Для задачи позиционирования БПЛА возьмите пару
   надирных снимков с перекрытием (или снимок и ортофото): насколько хватает SIFT при сезонных изменениях?
   (см. «Значение для задачи визуального позиционирования» в [конспекте]({K})).
9. **ORB vs SIFT на повороте.** Повторите раздел 8.1 для ORB с `WTA_K=3/4` и для `cv2.ORB_create(scaleFactor=1.1, nlevels=16)`.
   Где ORB догоняет SIFT, а где нет? Сравните время работы.
10. **Точный апсемплинг.** Сравните систематическое смещение координат (раздел 3.1) у `cv2.SIFT_create()` и
    `cv2.SIFT_create(enable_precise_upscale=True)` на синтетической паре со сдвигом на целое число пикселей.
""")
md("""
## Проверка: код ноутбука = код пакета `siftlab`

Тот же конвейер из пакета `siftlab/` (его используют тесты) должен дать идентичный результат.
""")
code("""
import importlib, siftlab
importlib.reload(siftlab)
k_pkg, d_pkg = siftlab.sift_detect_and_compute(box)
k_nb, d_nb = sift_detect_and_compute(box)
assert np.array_equal(kp_xy(k_pkg), kp_xy(k_nb)) and np.array_equal(d_pkg, d_nb)
print(f"OK: пакет и ноутбук совпадают ({len(d_nb)} точек на box.png)")
print(f"Время выполнения всего ноутбука: {time.time() - T_START:.0f} с")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (sift_lab .venv)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "sift_методичка.ipynb"
nbf.write(nb, out)
print("записано:", out, "ячеек:", len(cells))
