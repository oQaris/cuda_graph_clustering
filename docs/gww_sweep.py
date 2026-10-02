# GWW против PBILS без него, GPU. Готовые точки пропускаются, так что прерванный прогон можно просто запустить снова.
#   python docs/gww_sweep.py
# Точка — один процесс bin/cc_gpu на несколько сидов подряд (--runs). Прогон 0 отбрасывается: в него входит создание
# контекста CUDA. Сиды у всех долей GWW одни, поэтому прогоны сравниваются попарно.
#   docs/data/gww.csv       фиксированное число итераций без раннего останова: рекорд и время к итерациям 1, 2, 3, 5,
#                           7, 10, 15, 20, 30, ... и к последней
#   docs/data/gww_stop.csv  параметры по умолчанию, ранний останов после 6 итераций без рекорда: итог прогона
import csv
import os
import re
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GPU = os.path.join(ROOT, 'bin', 'cc_gpu')
CURVES = os.path.join(ROOT, 'docs', 'data', 'gww.csv')
STOPS = os.path.join(ROOT, 'docs', 'data', 'gww_stop.csv')
EDGES = os.path.join(ROOT, 'data', 'real', 'graphs', '{}.edges')

SHARES = ['0', '0.1', '0.25']
SEEDS = 8
DENSITY = '0.33'
GRAPH_SEEDS = [7, 8]
# (n, k, популяция, итераций): случайные G(n, p)
RANDOM = [(1000, 5, 128, 1000), (2000, 3, 128, 1000), (2000, 8, 128, 1000), (4000, 2, 128, 700), (6000, 3, 128, 500),
          (2000, 3, 1024, 300)]
# При k >= 5 GWW к длинным бюджетам отстаёт; там же проверена и доля мягче.
MORE_SHARES = {(1000, 5, 128): ['0.05'], (2000, 8, 128): ['0.05']}
# (граф, k, итераций): реальные графы из docs/real_graphs, k — число групп разметки, как в run.py. Короткая проверка:
# три быстрых графа, меньше сидов и одна доля.
REAL = [('cora', 7, 300), ('citeseer', 6, 300), ('amazon_photo', 8, 100)]
REAL_SEEDS = 4
REAL_SHARES = ['0', '0.1']

CHECKPOINTS = sorted({round(m * 10**e) for e in range(4) for m in (1, 1.5, 2, 3, 5, 7)})  # 1, 2, 3, 5, 7, 10, 15...
WARMUP = 1  # отбрасываемых прогонов в начале процесса
CURVE_FIELDS = ['set', 'graph', 'n', 'k', 'pop', 'gww', 'seed', 'iter', 'record', 't']
STOP_FIELDS = ['set', 'graph', 'n', 'k', 'pop', 'gww', 'seed', 'f', 'iters', 'seconds']
ITER = re.compile(r'^  iter\s+(\d+)\s+record (\d+)\s+stall \d+\s+([\d.]+)s')
RUN = re.compile(r'^run (\d+)\s+f=(\d+)\s+clusters=\d+\s+iters=(\d+)\s+([\d.]+)s')
INSTANCE = re.compile(r'^instance\s+n=(\d+)', re.M)


def done(path):
    if not os.path.exists(path):
        return set()
    return {(r['set'], r['graph'], r['k'], r['pop'], r['gww']) for r in csv.DictReader(open(path))}


def append(path, fields, rows):
    fresh = not os.path.exists(path)
    with open(path, 'a', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def solve(instance, k, pop, share, runs, extra):
    cmd = [GPU] + instance + ['--k', str(k), '--pop', str(pop), '--gww', share, '--seed', '1', '--runs', str(runs),
                              '--verbose', '--no-verify'] + extra
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    curves, finals, current = {}, {}, []
    for line in out.splitlines():
        if m := ITER.match(line):
            current.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
        elif m := RUN.match(line):
            curves[int(m.group(1))] = current
            finals[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
            current = []
    if len(finals) != runs:
        raise RuntimeError(' '.join(cmd) + '\n' + out[-800:])
    return int(INSTANCE.search(out).group(1)), curves, finals


def point(kind, graph, instance, k, pop, share, iters, seeds):
    key = (kind, graph, str(k), str(pop), share)
    base = dict(set=kind, graph=graph, k=k, pop=pop, gww=share)
    runs = seeds + WARMUP
    if key not in done(CURVES):
        n, curves, _ = solve(instance, k, pop, share, runs, ['--iters', str(iters), '--early-stop', str(iters)])
        rows = []
        for run in range(WARMUP, runs):
            for it, record, t in curves[run]:  # record — уже лучшее к этой итерации
                if it in CHECKPOINTS or it == len(curves[run]):
                    rows.append(dict(base, n=n, seed=run + 1, iter=it, record=record, t=t))
        append(CURVES, CURVE_FIELDS, rows)
        print('curves', key, flush=True)
    if key not in done(STOPS):
        n, _, finals = solve(instance, k, pop, share, runs, [])
        rows = [dict(base, n=n, seed=run + 1, f=f, iters=it, seconds=s) for run, (f, it, s) in finals.items()
                if run >= WARMUP]
        append(STOPS, STOP_FIELDS, rows)
        print('stop', key, flush=True)


for n, k, pop, iters in RANDOM:
    for g in GRAPH_SEEDS:
        for share in SHARES + MORE_SHARES.get((n, k, pop), []):
            instance = ['--n', str(n), '--density', DENSITY, '--graph-seed', str(g)]
            point('random', f'gnp{n}_{g}', instance, k, pop, share, iters, SEEDS)
for graph, k, iters in REAL:
    for share in REAL_SHARES:
        point('real', graph, ['--edges', EDGES.format(graph)], k, 128, share, iters, REAL_SEEDS)
