# Строит графики docs/img/*.png из docs/data/*.csv и печатает таблицы для docs/benchmarks.md:
#   python docs/plot_benchmarks.py docs/data docs/img
import csv
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator

DATA, IMG = sys.argv[1], sys.argv[2]

# Цвет времени — сущность: наш CPU синий, оригинал оранжевый, GPU и ускорение — свой цвет у каждого
# эксперимента, чтобы соседние графики не сливались. Штрих — второе измерение (потоки или бэкенд).
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREEN, VIOLET, AQUA, RED, MAGENTA = "#008300", "#4a3aa7", "#1baf7a", "#e34948", "#d55181"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def comma(x):
    return f"{x:g}".replace(".", ",")


COMMA = FuncFormatter(lambda v, _: comma(v))
INT = FuncFormatter(lambda v, _: f"{int(v)}")


def style(ax, xlabel, ylabel, title=None, logx=False, logy=False, xs=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK2)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(COMMA)
    if xs is not None:
        ax.set_xticks(xs)
        ax.xaxis.set_major_formatter(INT)
        ax.xaxis.set_minor_locator(NullLocator())


def figure(title, panels=2):
    fig, axes = plt.subplots(1, panels, figsize=(5.1 * panels, 4.3), dpi=150)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    return fig, axes


def save(fig, name):
    fig.tight_layout(rect=(0, 0.1 if fig.legends else 0, 1, 1))
    fig.savefig(f"{IMG}/{name}")
    plt.close(fig)


def label_points(ax, xs, ys, fmt, dy=8):
    for x, y in zip(xs, ys):
        ax.annotate(fmt(y), (x, y), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=8,
                    color=INK2)


def bottom_legend(fig, ax, ncol=None):
    # Легенда под рисунком: у четырёх линий в панели нет свободного угла.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=ncol or len(labels),
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.25)


def median(values):
    return statistics.median(values)


# --------------------------------------------------------------------------- CPU против оригинала
cpu = defaultdict(list)
cpu_f = defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/cpu.csv")):
    key = (r["variant"], int(r["n"]), int(r["k"]), int(r["pop"]))
    cpu[key].append(float(r["seconds"]))
    cpu_f[key].append(int(r["f"]))
cpu = {key: median(v) for key, v in cpu.items()}

VARIANTS = [("orig_1", "оригинал, 1 поток", ORANGE, "--", "s"),
            ("orig_16", "оригинал, 16 потоков", ORANGE, "-", "o"),
            ("ours_1", "наш, 1 поток", BLUE, "--", "s"),
            ("ours_16", "наш, 16 потоков", BLUE, "-", "o")]

ns = sorted({n for (_, n, _, p) in cpu if p == 128})
fig, axes = figure("CPU: время 5 итераций PBILS, популяция 128")
for ax, k in zip(axes, (2, 3)):
    for key, label, color, ls, marker in VARIANTS:
        xs = [n for n in ns if (key, n, k, 128) in cpu]
        ax.plot(xs, [cpu[(key, n, k, 128)] for n in xs], ls, color=color, lw=2, marker=marker, ms=5,
                label=label)
    style(ax, "число вершин n", "время, с (лог. шкала)", f"k = {k}", logy=True, xs=ns)
    ax.tick_params(axis="x", labelrotation=45)
bottom_legend(fig, axes[0])
save(fig, "cpu_variants.png")

fig, axes = figure("CPU: во сколько раз наша версия быстрее оригинала, популяция 128")
for ax, k in zip(axes, (2, 3)):
    for t, ls, marker, label in (("1", "--", "s", "оба в 1 поток"), ("16", "-", "o", "оба на 16 потоках")):
        xs = [n for n in ns if (f"orig_{t}", n, k, 128) in cpu]
        ys = [cpu[(f"orig_{t}", n, k, 128)] / cpu[(f"ours_{t}", n, k, 128)] for n in xs]
        ax.plot(xs, ys, ls, color=GREEN, lw=2, marker=marker, ms=5, label=label)
        label_points(ax, xs, ys, lambda y: comma(round(y, 1)) + "×", dy=8 if t == "1" else -14)
    style(ax, "число вершин n", "ускорение, раз", f"k = {k}", xs=ns)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylim(0, None)
top = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(0, top * 1.1)
axes[1].legend(frameon=False, loc="lower left")
save(fig, "cpu_speedup.png")

pops = sorted({p for (_, n, k, p) in cpu if n == 1000 and k == 3})
fig, axes = figure("CPU: зависимость от размера популяции, n = 1 000, k = 3, 5 итераций")
for key, label, color, ls, marker in VARIANTS:
    xs = [p for p in pops if (key, 1000, 3, p) in cpu]
    axes[0].plot(xs, [cpu[(key, 1000, 3, p)] for p in xs], ls, color=color, lw=2, marker=marker, ms=5, label=label)
style(axes[0], "размер популяции (лог. шкала)", "время, с (лог. шкала)", "время", logx=True, logy=True, xs=pops)
bottom_legend(fig, axes[0])
for t, ls, marker, label in (("1", "--", "s", "оба в 1 поток"), ("16", "-", "o", "оба на 16 потоках")):
    ys = [cpu[(f"orig_{t}", 1000, 3, p)] / cpu[(f"ours_{t}", 1000, 3, p)] for p in pops]
    axes[1].plot(pops, ys, ls, color=VIOLET, lw=2, marker=marker, ms=5, label=label)
    label_points(axes[1], pops, ys, lambda y: comma(round(y, 1)) + "×", dy=8 if t == "1" else -14)
style(axes[1], "размер популяции (лог. шкала)", "ускорение, раз", "ускорение", logx=True, xs=pops)
axes[1].set_ylim(0, max(axes[1].get_ylim()[1], 1) * 1.1)
axes[1].legend(frameon=False, loc="lower left")
save(fig, "cpu_pop.png")

# --------------------------------------------------------------------------- GPU
gpu = defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/gpu.csv")):
    assert r["gpu_f"] == r["cpu_f"], r
    gpu[r["series"]].append(r)


def series(name, col):
    by = defaultdict(list)
    for r in gpu[name]:
        by[int(r[col])].append(r)
    out = []
    for x in sorted(by):
        rows = by[x]
        assert len(rows) == 3, (name, x, len(rows))
        out.append((x, median(float(r["gpu_s"]) for r in rows), median(float(r["cpu_s"]) for r in rows)))
    return out


def gpu_vs_cpu(title, name, s_name, col, xlabel, accent):
    fig, (a1, a2) = figure(title)
    data = series(s_name, col)
    xs = [d[0] for d in data]
    a1.plot(xs, [d[2] for d in data], "--", color=BLUE, lw=2, marker="s", ms=5, label="cc_cpu, 16 потоков")
    a1.plot(xs, [d[1] for d in data], "-", color=accent, lw=2, marker="o", ms=5, label="cc_gpu")
    sp = [d[2] / d[1] for d in data]
    a2.plot(xs, sp, "-", color=accent, lw=2, marker="o", ms=5)
    label_points(a2, xs, sp, lambda y: f"{y:.0f}×")
    style(a1, xlabel, "время, с (лог. шкала)", "время", logx=True, logy=True, xs=xs)
    style(a2, xlabel, "ускорение GPU, раз", "ускорение", logx=True, xs=xs)
    a2.set_ylim(0, a2.get_ylim()[1] * 1.12)
    a1.legend(frameon=False, loc="upper left")
    for a in (a1, a2):
        a.tick_params(axis="x", labelrotation=45)
    save(fig, name)


gpu_vs_cpu("GPU против CPU на 16 потоках: 20 итераций, k = 3, популяция 128", "gpu_n.png", "n128", "n",
           "число вершин n (лог. шкала)", AQUA)
gpu_vs_cpu("GPU против CPU на 16 потоках: 20 итераций, n = 6 000, k = 3", "gpu_pop.png", "pop6000", "pop",
           "размер популяции (лог. шкала)", RED)
gpu_vs_cpu("GPU против CPU на 16 потоках: 20 итераций, n = 2 000, популяция 128", "gpu_k.png", "k128", "k",
           "число кластеров k (лог. шкала)", MAGENTA)

# --------------------------------------------------------------------------- оптимальная популяция
# Рекорд к моменту budget_s; отставание считается от лучшего решения, найденного на этом графе
# любой популяцией, и усредняется медианой по графам и сидам.
POP_CFGS = [(1000, 3, "n = 1 000, k = 3", VIOLET), (2000, 3, "n = 2 000, k = 3", GREEN),
            (6000, 3, "n = 6 000, k = 3", "#c98500"), (2000, 8, "n = 2 000, k = 8", MAGENTA)]
recs = defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/popopt.csv")):
    if r["record"]:
        recs[(int(r["n"]), int(r["k"]), int(r["graph_seed"]), int(r["pop"]), float(r["budget_s"]))].append(
            int(r["record"]))
best_known = defaultdict(lambda: None)
for (n, k, g, p, b), vals in recs.items():
    cur = best_known[(n, k, g)]
    best_known[(n, k, g)] = min(vals) if cur is None else min(cur, min(vals))
popopt_pops = sorted({key[3] for key in recs})


def gap(n, k, p, b):
    vals = [(v - best_known[(n, k, g)]) / best_known[(n, k, g)] * 100
            for g in (7, 8) for v in recs.get((n, k, g, p, b), [])]
    return median(vals) if len(vals) == 6 else None


def popopt_budgets(n, k):
    return sorted({key[4] for key in recs if key[0] == n and key[1] == k})


fig, (a1, a2) = figure("Какая популяция лучше за заданное время: GPU, 2 графа × 3 сида на точку")
for n, k, label, color in POP_CFGS:
    xs, ys = [], []
    for b in popopt_budgets(n, k):
        cand = {p: gap(n, k, p, b) for p in popopt_pops}
        cand = {p: v for p, v in cand.items() if v is not None}
        xs.append(b)
        ys.append(min(cand, key=cand.get))
    a1.plot(xs, ys, "-", color=color, lw=2, marker="o", ms=5, label=label)
a1.set_xscale("log")
a1.set_yscale("log", base=2)
a1.set_yticks(popopt_pops)
a1.yaxis.set_major_formatter(INT)
a1.yaxis.set_minor_locator(NullLocator())
all_b = sorted({key[4] for key in recs})
style(a1, "бюджет времени, с (лог. шкала)", "лучшая популяция (лог. шкала)", "лучшая популяция")
a1.set_xticks(all_b)
a1.xaxis.set_major_formatter(COMMA)
a1.xaxis.set_minor_locator(NullLocator())
a1.legend(frameon=False, loc="upper left")
# Справа — как популяции обгоняют друг друга на одной конфигурации.
shown = [64, 256, 1024, 4096]
ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
bs = popopt_budgets(2000, 3)
for p, color in zip(shown, ramp):
    pts = [(b, gap(2000, 3, p, b)) for b in bs]
    pts = [(b, v) for b, v in pts if v is not None]
    a2.plot([b for b, _ in pts], [v for _, v in pts], "-", color=color, lw=2, marker="o", ms=5,
            label=f"популяция {p}")
style(a2, "бюджет времени, с (лог. шкала)", "отставание от лучшего, % (лог. шкала)", "n = 2 000, k = 3",
      logy=True)
a2.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2])
a2.yaxis.set_major_formatter(COMMA)
a2.yaxis.set_minor_locator(NullLocator())
a2.set_xscale("log")
a2.set_xticks(bs)
a2.xaxis.set_major_formatter(COMMA)
a2.xaxis.set_minor_locator(NullLocator())
a2.legend(frameon=False, loc="lower left")
save(fig, "pop_optimum.png")

# --------------------------------------------------------------------------- таблицы для md
if "--tables" in sys.argv:
    def s(x):
        return comma(float(f"{x:.3g}")) + " с"

    for k in (2, 3):
        print(f"\nCPU k={k}\n| n | оригинал, 1 поток | оригинал, 16 потоков | наш, 1 поток | наш, 16 потоков |")
        for n in ns:
            print(f"| {n} | " + " | ".join(s(cpu[(v, n, k, 128)]) for v, *_ in VARIANTS) + " |")
    print("\nCPU pop\n| популяция | оригинал, 1 поток | оригинал, 16 потоков | наш, 1 поток | наш, 16 потоков |")
    for p in pops:
        print(f"| {p} | " + " | ".join(s(cpu[(v, 1000, 3, p)]) for v, *_ in VARIANTS) + " |")
    for name, col in (("n128", "n"), ("pop6000", "pop"), ("k128", "k")):
        print(f"\n{name}\n| {col} | cc_gpu | cc_cpu --threads 16 | GPU / CPU |")
        for x, g, c in series(name, col):
            print(f"| {x} | {s(g)} | {s(c)} | {c / g:.0f}× |")
    for n, k, label, _ in POP_CFGS:
        print(f"\npopopt {label}\n| бюджет | " + " | ".join(str(p) for p in popopt_pops) + " |")
        for b in popopt_budgets(n, k):
            vals = {p: gap(n, k, p, b) for p in popopt_pops}
            top = min((p for p in vals if vals[p] is not None), key=lambda p: vals[p])
            cells = ["—" if vals[p] is None else (f"**{comma(round(vals[p], 3))}**" if p == top
                                                   else comma(round(vals[p], 3))) for p in popopt_pops]
            print(f"| {comma(b)} с | " + " | ".join(cells) + " |")
