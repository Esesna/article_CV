#!/usr/bin/env python3
"""Учебный разбор DoG (SIFT, разд. 3) на синтетических данных.

Генерирует рисунки в OUT и печатает markdown-таблицы с числами в OUT/numbers.md.
Только numpy + matplotlib.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})
md = []


def gauss1d(sigma, radius=None):
    r = int(np.ceil(3 * sigma)) if radius is None else radius
    x = np.arange(-r, r + 1)
    g = np.exp(-x**2 / (2 * sigma**2))
    return g / g.sum()


def gauss2d(sigma, radius):
    g = gauss1d(sigma, radius)
    return np.outer(g, g)


def blur(img, sigma):
    """Сепарабельная гауссова свёртка с отражением на границах."""
    g = gauss1d(sigma)
    r = len(g) // 2
    p = np.pad(img, r, mode="reflect")
    tmp = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 1, p)
    return np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 0, tmp)


def conv2d_full_kernel(img, K):
    """Прямая 2D-свёртка (для маленького числового примера)."""
    r = K.shape[0] // 2
    p = np.pad(img, r, mode="reflect")
    out = np.zeros_like(img, dtype=float)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y, x] = (p[y:y + 2*r + 1, x:x + 2*r + 1] * K).sum()
    return out


def mat_md(M, fmt="{:.3f}", hl=None):
    n = M.shape[1]
    c = n // 2
    head = "| |" + "|".join(f"{j - c:+d}" if j != c else "0" for j in range(n)) + "|"
    sep = "|---" * (n + 1) + "|"
    rows = []
    for i in range(M.shape[0]):
        cells = []
        for j in range(n):
            s = fmt.format(M[i, j])
            if hl is not None and (i, j) == hl:
                s = f"**{s}**"
            cells.append(s)
        lab = f"{i - M.shape[0]//2:+d}" if i != M.shape[0]//2 else "0"
        rows.append(f"| **{lab}** |" + "|".join(cells) + "|")
    return "\n".join([head, sep] + rows)


# ---------------------------------------------------------------- 1. Ядра
sigma, k = 1.0, np.sqrt(2)
R = 3
G1 = gauss2d(sigma, R)
G2 = gauss2d(k * sigma, R)
DoGK = G2 - G1
md.append("### ядро G(σ=1.0), 7×7\n\n" + mat_md(G1))
md.append("### ядро G(kσ=1.414), 7×7\n\n" + mat_md(G2))
md.append("### ядро DoG = G(kσ) − G(σ), 7×7 (×1000)\n\n" + mat_md(DoGK * 1000, "{:.1f}"))
md.append(f"сумма G1={G1.sum():.4f}, G2={G2.sum():.4f}, DoG={DoGK.sum():.2e}; "
          f"центр DoG={DoGK[R,R]:.4f}; min={DoGK.min():.4f}; max={DoGK.max():.4f}")
# радиус, где DoG меняет знак (по центральной строке, непрерывно)
xs = np.linspace(0, 4, 4001)
g1 = np.exp(-xs**2/(2*sigma**2))/(2*np.pi*sigma**2)
g2 = np.exp(-xs**2/(2*(k*sigma)**2))/(2*np.pi*(k*sigma)**2)
zc = xs[np.argmax(g2 - g1 > 0)]
md.append(f"смена знака DoG (непрерывно) при r ≈ {zc:.3f}")

fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for a, M, t in [(ax[0], G1, "G(x,y,σ=1.0)"), (ax[1], G2, "G(x,y,kσ=1.41)")]:
    im = a.imshow(M, cmap="viridis")
    for (i, j), v in np.ndenumerate(M):
        a.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5,
               color="white" if v < M.max()*0.6 else "black")
    a.set_title(t); a.set_xticks(range(7), range(-3, 4)); a.set_yticks(range(7), range(-3, 4))
lim = np.abs(DoGK).max()
im = ax[2].imshow(DoGK, cmap="RdBu_r", vmin=-lim, vmax=lim)
for (i, j), v in np.ndenumerate(DoGK):
    ax[2].text(j, i, f"{v*1000:.0f}", ha="center", va="center", fontsize=7)
ax[2].set_title("DoG = G(kσ) − G(σ), ×1000"); ax[2].set_xticks(range(7), range(-3, 4)); ax[2].set_yticks(range(7), range(-3, 4))
# 1D-профиль и приближение лапласианом
x = np.linspace(-5, 5, 1001)
def g2d_row(s): return np.exp(-x**2/(2*s**2))/(2*np.pi*s**2)
dog = g2d_row(k*sigma) - g2d_row(sigma)
s_mid = sigma * (1 + k) / 2  # конечная разность по σ берётся в середине [σ, kσ]
log = (x**2/s_mid**4 - 2/s_mid**2) * g2d_row(s_mid)  # ∇²G в точке (x, 0)
ax[3].plot(x, g2d_row(sigma), label="G(σ)", color="#1f77b4")
ax[3].plot(x, g2d_row(k*sigma), label="G(kσ)", color="#2ca02c")
ax[3].plot(x, dog, label="DoG = G(kσ)−G(σ)", color="#d62728", lw=2)
ax[3].plot(x, (k-1)*sigma*s_mid*log, "--", label=f"(k−1)σ·σₘ∇²G(σₘ), σₘ={s_mid:.2f}", color="black")
ax[3].axhline(0, color="gray", lw=0.6); ax[3].legend(fontsize=8); ax[3].set_title("Срез по y=0: DoG ≈ (k−1)σ²∇²G", fontsize=9)
fig.suptitle("Шаг 1. Матрицы ядер: два гауссиана и их разность («перевёрнутая мексиканская шляпа»)")
fig.savefig(OUT / "dog01_kernels.png"); plt.close(fig)

# ------------------------------------------------- 2. Маленький числовой пример
N = 11
I = np.full((N, N), 20.0)
I[4:7, 4:7] = 200.0            # светлое пятно 3×3 на тёмном фоне
L1 = conv2d_full_kernel(I, G1)
L2 = conv2d_full_kernel(I, G2)
D = L2 - L1
D_direct = conv2d_full_kernel(I, DoGK)
assert np.allclose(D, D_direct)
c = N // 2
crop = slice(c-3, c+4)
md.append("### I (центр 7×7)\n\n" + mat_md(I[crop, crop], "{:.0f}"))
md.append("### L1 = G(σ)*I\n\n" + mat_md(L1[crop, crop], "{:.1f}"))
md.append("### L2 = G(kσ)*I\n\n" + mat_md(L2[crop, crop], "{:.1f}"))
md.append("### D = L2 − L1\n\n" + mat_md(D[crop, crop], "{:.1f}", hl=(3, 3)))
# ручной расчёт для центра
patch = I[c-3:c+4, c-3:c+4]
md.append(f"ручной расчёт D(центр) = Σ DoG·I = {(patch*DoGK).sum():.2f}; "
          f"L1(центр)={L1[c,c]:.2f}, L2(центр)={L2[c,c]:.2f}")
md.append("строка центра D: " + ", ".join(f"{v:.1f}" for v in D[c]))

fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
for a, M, t, cmap in [(ax[0], I, "I — исходное (пятно 3×3)", "gray"),
                      (ax[1], L1, "L₁ = G(σ=1)∗I", "gray"),
                      (ax[2], L2, "L₂ = G(σ=1.41)∗I", "gray")]:
    a.imshow(M, cmap=cmap, vmin=0, vmax=200)
    for (i, j), v in np.ndenumerate(M):
        a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
               color="black" if v > 110 else "white")
    a.set_title(t); a.axis("off")
lim = np.abs(D).max()
ax[3].imshow(D, cmap="RdBu_r", vmin=-lim, vmax=lim)
for (i, j), v in np.ndenumerate(D):
    ax[3].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5)
ax[3].add_patch(plt.Rectangle((c-0.5, c-0.5), 1, 1, fill=False, ec="black", lw=2))
ax[3].set_title("D = L₂ − L₁ (синее < 0, красное > 0)"); ax[3].axis("off")
fig.suptitle("Шаг 2. Свёртка на числах: светлое пятно → сильный отрицательный отклик DoG в центре, положительное кольцо вокруг")
fig.savefig(OUT / "dog02_numeric.png"); plt.close(fig)

# --------------------------------------------- 3. Масштабное пространство
rng = np.random.default_rng(0)
H, W = 200, 360
yy, xx = np.mgrid[0:H, 0:W]
img = np.full((H, W), 0.5)
disks = [(50, 50, 3, 1.0), (50, 130, 6, 1.0), (60, 240, 12, 1.0), (145, 70, 9, 0.0)]
for cy, cx, r, v in disks:
    img[(yy - cy)**2 + (xx - cx)**2 <= r**2] = v
img[120:175, 170:335] = 0.9           # большой прямоугольник: края и углы
img += rng.uniform(-0.01, 0.01, img.shape)  # 1% шума, как в экспериментах статьи
img = blur(img, 0.5)                  # исходное «уже размыто» σ≈0.5

s = 3
k3 = 2 ** (1 / s)
sig0 = 1.6
n_g = 3 * s + 3                       # три октавы без прореживания (для наглядности)
sigmas = sig0 * k3 ** np.arange(n_g)
Ls = np.stack([blur(img, sq) for sq in sigmas])
Ds = Ls[1:] - Ls[:-1]                 # D_i при масштабе σ_i
md.append("сигмы гауссианов: " + ", ".join(f"{v:.2f}" for v in sigmas))

show = range(0, s + 3)                # первая «октава»: s+3 гауссианов, s+2 DoG
fig, ax = plt.subplots(2, s + 3, figsize=(17, 4.6))
for col, i in enumerate(show):
    ax[0, col].imshow(Ls[i], cmap="gray", vmin=0, vmax=1)
    ax[0, col].set_title(f"L, σ={sigmas[i]:.2f}", fontsize=9); ax[0, col].axis("off")
lim = np.abs(Ds[:s + 2]).max() * 0.5
for col in range(s + 3):
    ax[1, col].axis("off")
    if col < s + 2:
        ax[1, col].imshow(Ds[col], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[1, col].set_title(f"D = L({sigmas[col+1]:.2f}) − L({sigmas[col]:.2f})", fontsize=8)
fig.suptitle(f"Шаг 3. Одна октава при s={s}: {s+3} сглаженных изображения L (верх) → {s+2} изображений DoG (низ). k = 2^(1/{s}) ≈ {k3:.3f}")
fig.savefig(OUT / "dog03_stack.png"); plt.close(fig)

# --------------------------------------------- 4. Выбор масштаба
fig, ax = plt.subplots(figsize=(7.5, 3.8))
rows = []
for cy, cx, r, v in disks:
    resp = Ds[:, cy, cx]
    ax.plot(sigmas[:-1], resp, marker="o", ms=3, label=f"{'светлый' if v > 0.5 else 'тёмный'} диск r={r}")
    i = np.argmax(np.abs(resp))
    rows.append((r, v, sigmas[i], r / np.sqrt(2), resp[i]))
ax.axhline(0, color="gray", lw=0.6); ax.set_xscale("log", base=2)
ax.set_xlabel("σ (лог. шкала)"); ax.set_ylabel("D(центр диска, σ)")
ax.set_title("Шаг 4. Отклик DoG в центре диска как функция масштаба: пик при σ ≈ r/√2")
ax.legend(fontsize=8)
fig.savefig(OUT / "dog04_scale_selection.png"); plt.close(fig)
md.append("### выбор масштаба\n\n| радиус r | диск | σ пика | r/√2 | D в пике |\n|---|---|---|---|---|\n" +
          "\n".join(f"| {r} | {'светлый' if v > 0.5 else 'тёмный'} | {sp:.2f} | {th:.2f} | {dv:+.3f} |"
                    for r, v, sp, th, dv in rows))

# --------------------------------------------- 5. Экстремумы по 26 соседям
thr = 0.03 * (k3 - 1) / (2 ** 0.5 - 1) * 0  # см. ниже: порог задаём по |D| напрямую
thr = 0.01
kps = []
for i in range(1, len(Ds) - 1):
    cube = Ds[i-1:i+2]
    c0 = Ds[i]
    for y in range(1, H - 1):
        row = c0[y]
        cand = np.where(np.abs(row[1:-1]) > thr)[0] + 1
        for x in cand:
            v = c0[y, x]
            nb = cube[:, y-1:y+2, x-1:x+2]
            if (v > 0 and v >= nb.max() and (nb == v).sum() == 1) or \
               (v < 0 and v <= nb.min() and (nb == v).sum() == 1):
                kps.append((y, x, i, v))
# тест на края (r=10) по гессиану D
rr = 10
keep, edge = [], []
for y, x, i, v in kps:
    d = Ds[i]
    dxx = d[y, x+1] + d[y, x-1] - 2*v
    dyy = d[y+1, x] + d[y-1, x] - 2*v
    dxy = (d[y+1, x+1] - d[y+1, x-1] - d[y-1, x+1] + d[y-1, x-1]) / 4
    tr, det = dxx + dyy, dxx*dyy - dxy**2
    (keep if det > 0 and tr**2 / det < (rr + 1)**2 / rr else edge).append((y, x, i, v))
md.append(f"порог |D|>{thr}; экстремумов: {len(kps)}, после теста краёв r={rr}: {len(keep)} (отброшено {len(edge)})")

fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
for a, pts, t in [(ax[0], kps, f"Все экстремумы по 26 соседям, |D|>{thr}: {len(kps)}"),
                  (ax[1], keep, f"После теста краёв (r={rr}): {len(keep)}")]:
    a.imshow(img, cmap="gray", vmin=0, vmax=1)
    for y, x, i, v in pts:
        a.add_patch(Circle((x, y), np.sqrt(2) * sigmas[i], fill=False,
                           ec="#d62728" if v < 0 else "#1f77b4", lw=1.2))
        a.plot(x, y, ".", ms=2, color="yellow")
    a.set_title(t); a.axis("off")
fig.suptitle("Шаг 5. Найденные ключевые точки: окружность радиуса √2·σ (красные — минимумы D, светлые пятна; синие — максимумы, тёмные пятна)")
fig.savefig(OUT / "dog05_keypoints.png"); plt.close(fig)

# 3×3×3 окрестность для точки на диске r=6
cy, cx, r, _ = disks[1]
best = min((p for p in keep if abs(p[0]-cy) <= 1 and abs(p[1]-cx) <= 1), key=lambda p: p[3])
y, x, i, v = best
cube = Ds[i-1:i+2, y-1:y+2, x-1:x+2]
md.append(f"### 26 соседей точки (y={y}, x={x}), σ={sigmas[i]:.2f}, D={v:.4f}\n")
for lvl, name in zip(range(3), [f"масштаб ниже σ={sigmas[i-1]:.2f}", f"тот же σ={sigmas[i]:.2f}", f"масштаб выше σ={sigmas[i+1]:.2f}"]):
    md.append(f"**{name}** (×1000)\n\n" + mat_md(cube[lvl] * 1000, "{:.2f}", hl=(1, 1) if lvl == 1 else None))

fig, ax = plt.subplots(1, 3, figsize=(10, 3.8))
lim = np.abs(cube).max()
for lvl in range(3):
    ax[lvl].imshow(cube[lvl], cmap="RdBu_r", vmin=-lim, vmax=lim)
    for (a_, b_), val in np.ndenumerate(cube[lvl]):
        ax[lvl].text(b_, a_, f"{val*1000:.1f}", ha="center", va="center", fontsize=10,
                     color="white", fontweight="bold" if (lvl, a_, b_) == (1, 1, 1) else "normal")
    ax[lvl].set_title(["масштаб ниже", "текущий масштаб", "масштаб выше"][lvl] + f"\nσ={sigmas[i-1+lvl]:.2f}")
    ax[lvl].axis("off")
fig.suptitle("Проверка экстремума: центр (жирный) меньше всех 26 соседей, D×1000", y=1.06)
fig.savefig(OUT / "dog06_neighbors.png"); plt.close(fig)

# край: пример отброшенной точки
if edge:
    y, x, i, v = edge[0]
    d = Ds[i]
    dxx = d[y, x+1] + d[y, x-1] - 2*v; dyy = d[y+1, x] + d[y-1, x] - 2*v
    dxy = (d[y+1, x+1] - d[y+1, x-1] - d[y-1, x+1] + d[y-1, x-1]) / 4
    tr, det = dxx + dyy, dxx*dyy - dxy**2
    md.append(f"пример краевой точки (y={y},x={x},σ={sigmas[i]:.2f}): Dxx={dxx:.5f}, Dyy={dyy:.5f}, Dxy={dxy:.5f}, "
              f"Tr²/Det={'∞ (Det≤0)' if det <= 0 else f'{tr**2/det:.1f}'} vs порог {(rr+1)**2/rr:.1f}")
    # пример оставленной
    y, x, i, v = best
    d = Ds[i]
    dxx = d[y, x+1] + d[y, x-1] - 2*v; dyy = d[y+1, x] + d[y-1, x] - 2*v
    dxy = (d[y+1, x+1] - d[y+1, x-1] - d[y-1, x+1] + d[y-1, x-1]) / 4
    tr, det = dxx + dyy, dxx*dyy - dxy**2
    md.append(f"точка на диске: Dxx={dxx:.5f}, Dyy={dyy:.5f}, Dxy={dxy:.5f}, Tr²/Det={tr**2/det:.2f}")

(OUT / "numbers.md").write_text("\n\n".join(md), encoding="utf-8")
print("\n\n".join(md))
