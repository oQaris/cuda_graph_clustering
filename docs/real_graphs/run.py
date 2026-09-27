# Решает подготовленные графы (prepare.py) бинарником bin/cc_gpu (или --backend=cpu) и сверяет с разметкой групп.
#   python docs/real_graphs/run.py [граф ...] [--k=K] [--backend=cpu]
# k по умолчанию — число групп первой разметки (не больше 64), без разметки 2. Популяция 128, --seed 5. До 32 767
# вершин прогонов два, время — второго: в первый входит создание контекста CUDA; f и метки — лучшие из двух.
# Результаты дописываются в docs/data/real_graphs.json под ключом "граф/k=K/бэкенд", метки — в data/real/out.
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GRAPHS = os.path.join(ROOT, 'data', 'real', 'graphs')
OUT = os.path.join(ROOT, 'data', 'real', 'out')
RESULTS = os.path.join(ROOT, 'docs', 'data', 'real_graphs.json')
NARROW = 32767  # до этого размера время одного прогона сравнимо с созданием контекста CUDA

options = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
backend = options.get('backend', 'gpu')
graphs = json.load(open(os.path.join(GRAPHS, 'graphs.json')))
names = [a for a in sys.argv[1:] if not a.startswith('--')] or list(graphs)
results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
os.makedirs(OUT, exist_ok=True)


def objective(adj, labels):
    sizes = np.bincount(labels)
    a = adj.tocoo()
    intra = int(np.sum(labels[a.row] == labels[a.col])) // 2
    return int(adj.nnz // 2 + np.sum(sizes * (sizes - 1) // 2) - 2 * intra)


for name in names:
    info = graphs[name]
    groups = [v for key, v in info.items() if key.startswith('groups_')]
    k = int(options['k']) if 'k' in options else min(groups[0], 64) if groups else 2
    runs = 2 if info['n'] <= NARROW else 1
    labels_path = os.path.join(OUT, f'{name}.{backend}.k{k}.labels')
    cmd = [os.path.join(ROOT, 'bin', f'cc_{backend}'), '--edges', os.path.join(GRAPHS, name + '.edges'), '--k', str(k),
           '--pop', '128', '--seed', '5', '--runs', str(runs), '--time-limit', '3600', '--verbose', '--labels-out',
           labels_path] + (['--threads', '16'] if backend == 'cpu' else [])
    started = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    wall = time.time() - started
    open(os.path.join(OUT, f'{name}.{backend}.k{k}.log'), 'w').write(out)
    found = re.findall(r'^run (\d+)\s+f=(\d+)\s+clusters=(\d+)\s+iters=(\d+)\s+([\d.]+)s', out, re.M)
    if len(found) < runs or 'verified' not in out:
        print(name, 'FAILED', out[-500:], flush=True)
        continue
    last = found[-1]
    record = [(int(i), int(f)) for i, f in re.findall(r'^  iter\s+(\d+)\s+record (\d+)', out, re.M)]
    record = record[-int(last[3]):]  # итерации последнего прогона
    last_improvement = next(i for i, f in record if f == int(last[1]))
    r = dict(n=info['n'], m=info['m'], k=k, backend=backend, f=min(int(x[1]) for x in found), iters=int(last[3]),
             seconds=float(last[4]), wall=round(wall, 1), last_improvement=last_improvement)
    labels = np.array(open(labels_path).read().split(), dtype=np.int64)
    adj = sp.load_npz(os.path.join(GRAPHS, name + '.adj.npz'))
    assert objective(adj, labels) == r['f'], name
    sizes = np.bincount(labels, minlength=k)
    if k == 2:  # f = C(S0,2) + C(S1,2) - m + 2 * разрез
        r['cut_share'] = round(float((r['f'] - np.sum(sizes * (sizes - 1) // 2) + r['m']) / 2 / r['m']), 4)
    for path in glob.glob(os.path.join(GRAPHS, f'{name}.*.labels.npy')):
        key = os.path.basename(path).split('.')[1]
        truth = np.load(path)
        r[f'f_{key}'] = objective(adj, truth)
        r[f'ari_{key}'] = round(adjusted_rand_score(truth, labels), 3)
        r[f'nmi_{key}'] = round(normalized_mutual_info_score(truth, labels), 3)
    results[f'{name}/k={k}/{backend}'] = r
    json.dump(results, open(RESULTS, 'w'), indent=1, ensure_ascii=False)
    print(name, r, flush=True)
