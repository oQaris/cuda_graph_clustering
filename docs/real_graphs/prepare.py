# Строит из скачанного (download.sh) простые неориентированные графы без петель для cc_gpu --edges:
#   data/real/graphs/<граф>.edges              ребро один раз, вершина без рёбер — петлёй, чтобы не выпала из рангов
#   data/real/graphs/<граф>.<разметка>.labels.npy  группы вершин в том же порядке, что у --edges
#   data/real/graphs/<граф>.adj.npz            матрица смежности (scipy) для проверки f и сравнения с разметкой
#   data/real/graphs/graphs.json               n, m, плотность, объём битовой матрицы, число групп, доля близнецов
# Запуск: python docs/real_graphs/prepare.py [граф ...]; без аргументов — все графы.
import csv
import gzip
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'real', 'raw')
OUT = os.path.join(ROOT, 'data', 'real', 'graphs')


# Каждый источник возвращает (рёбра по номерам, {разметка: pd.Series по номеру}, число вершин или None). При None
# вершины — номера, встреченные в рёбрах, по возрастанию: так нумерует и Graph::LoadEdgeList.
def netzschleuder(name, label=None):
    z = zipfile.ZipFile(os.path.join(RAW, name + '.csv.zip'))
    rows = list(csv.reader(z.read('nodes.csv').decode('utf-8').splitlines()))
    header = [c.strip().lstrip("# ") for c in rows[0]]
    nodes = rows[1:]
    edges = pd.read_csv(io.BytesIO(z.read('edges.csv')), comment='#', header=None, usecols=[0, 1]).to_numpy()
    labels = {label[1]: pd.Series([r[header.index(label[0])] for r in nodes])} if label else {}
    return edges, labels, len(nodes)


def snap_email():
    edges = np.loadtxt(gzip.open(os.path.join(RAW, 'email-Eu-core.txt.gz')), dtype=np.int64)
    dep = np.loadtxt(gzip.open(os.path.join(RAW, 'email-Eu-core-department-labels.txt.gz')), dtype=np.int64)
    return edges, {'department': pd.Series(dep[:, 1], index=dep[:, 0])}, len(dep)


def musae(archive, prefix, column, key):
    z = zipfile.ZipFile(os.path.join(RAW, archive))
    edges = pd.read_csv(io.BytesIO(z.read(prefix + '_edges.csv'))).to_numpy()
    target = pd.read_csv(io.BytesIO(z.read(prefix + '_target.csv'))).set_index('id')
    return edges, {key: target[column]}, len(target)


def gnn_benchmark(name):
    d = np.load(os.path.join(RAW, name + '.npz'), allow_pickle=True)
    a = sp.csr_matrix((d['adj_data'], d['adj_indices'], d['adj_indptr']), shape=d['adj_shape']).tocoo()
    return np.stack([a.row, a.col], 1), {'class': pd.Series(d['labels'])}, int(d['adj_shape'][0])


def snap_text(file):
    edges = pd.read_csv(os.path.join(RAW, file), sep=r'\s+', comment='#', header=None, usecols=[0, 1], dtype=np.int64)
    return edges.to_numpy(), {}, None


def twitch_gamers():
    z = zipfile.ZipFile(os.path.join(RAW, 'twitch_gamers.zip'))
    edges = pd.read_csv(io.BytesIO(z.read(next(f for f in z.namelist() if f.endswith('edges.csv'))))).to_numpy()
    features = pd.read_csv(io.BytesIO(z.read(next(f for f in z.namelist() if f.endswith('features.csv')))))
    features = features.set_index('numeric_id')
    return edges, {key: features[key] for key in ('language', 'mature', 'affiliate')}, None


def ogbn_arxiv():
    z = zipfile.ZipFile(os.path.join(RAW, 'arxiv.zip'))
    edges = pd.read_csv(io.BytesIO(gzip.decompress(z.read('arxiv/raw/edge.csv.gz'))), header=None).to_numpy()
    subject = pd.read_csv(io.BytesIO(gzip.decompress(z.read('arxiv/raw/node-label.csv.gz'))), header=None)[0]
    return edges, {'subject': subject}, None


def reddit():
    z = zipfile.ZipFile(os.path.join(RAW, 'reddit.zip'))
    g = np.load(io.BytesIO(z.read('reddit_graph.npz')))
    data = np.load(io.BytesIO(z.read('reddit_data.npz')))
    return np.stack([g['row'], g['col']], 1), {'subreddit': pd.Series(data['label'])}, None


def gplus():  # номера — 21-значные id Google+, в 64 бита не помещаются: перенумеровываем по строкам
    e = pd.read_csv(os.path.join(RAW, 'gplus_combined.txt.gz'), sep=' ', header=None, dtype=str)
    codes, _ = pd.factorize(pd.concat([e[0], e[1]]), sort=True)
    return np.stack([codes[:len(e)], codes[len(e):]], 1), {}, None


SOURCES = {
    # малые и средние, с разметкой групп
    'karate': lambda: netzschleuder('karate', ('groups', 'club')),
    'dolphins': lambda: netzschleuder('dolphins'),
    'football': lambda: netzschleuder('football', ('value', 'conference')),
    'polbooks': lambda: netzschleuder('polbooks', ('value', 'leaning')),
    'polblogs': lambda: netzschleuder('polblogs', ('value', 'leaning')),  # берётся наибольшая компонента, см. LCC
    'email-Eu-core': snap_email,
    'sp_primary_school': lambda: netzschleuder('sp_primary_school', ('class', 'class')),
    'sp_high_school': lambda: netzschleuder('sp_high_school', ('class', 'class')),
    'cora': lambda: gnn_benchmark('cora'),
    'citeseer': lambda: gnn_benchmark('citeseer'),
    'amazon_photo': lambda: gnn_benchmark('amazon_electronics_photo'),
    'lastfm_asia': lambda: musae('lastfm_asia.zip', 'lasftm_asia/lastfm_asia', 'target', 'country'),
    'amazon_computers': lambda: gnn_benchmark('amazon_electronics_computers'),
    'coauthor_cs': lambda: gnn_benchmark('ms_academic_cs'),
    'pubmed': lambda: gnn_benchmark('pubmed'),
    'cora_full': lambda: gnn_benchmark('cora_full'),
    'deezer_europe': lambda: musae('deezer_europe.zip', 'deezer_europe/deezer_europe', 'target', 'gender'),
    'coauthor_physics': lambda: gnn_benchmark('ms_academic_phy'),
    'github': lambda: musae('git_web_ml.zip', 'git_web_ml/musae_git', 'ml_target', 'developer'),
    # больше 100 тысяч вершин
    'gplus': gplus,
    'twitch_gamers': twitch_gamers,
    'ogbn_arxiv': ogbn_arxiv,
    'loc_gowalla': lambda: snap_text('loc-gowalla_edges.txt.gz'),
    'reddit': reddit,
    'amazon0302': lambda: snap_text('amazon0302.txt.gz'),
    'email_euall': lambda: snap_text('email-EuAll.txt.gz'),
    'web_stanford': lambda: snap_text('web-Stanford.txt.gz'),
    'com_dblp': lambda: snap_text('com-dblp.ungraph.txt.gz'),
}
LCC = {'polblogs'}  # как в литературе: 1222 блога наибольшей компоненты из 1490


def adjacency(edges, n):
    e = edges[edges[:, 0] != edges[:, 1]]
    a = sp.coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return ((a + a.T) > 0).astype(np.int8).tocsr()


def twins(a):
    """Истинные близнецы — вершины с одинаковыми замкнутыми окрестностями: доля вершин в классах от двух и больший."""
    closed = (a + sp.eye(a.shape[0], dtype=np.int8, format='csr')).tocsr()
    counts = {}
    for v in range(closed.shape[0]):
        key = hash(closed.indices[closed.indptr[v]:closed.indptr[v + 1]].tobytes())  # индексы строки CSR отсортированы
        counts[key] = counts.get(key, 0) + 1
    sizes = np.array(list(counts.values()))
    return float(sizes[sizes > 1].sum() / closed.shape[0]), int(sizes.max())


def prepare(name):
    edges, labels, n = SOURCES[name]()
    ids = np.arange(n) if n is not None else np.unique(edges)
    a = adjacency(np.searchsorted(ids, edges), len(ids))
    if name in LCC:
        _, component = sp.csgraph.connected_components(a, directed=False)
        keep = np.flatnonzero(component == np.bincount(component).argmax())
        a, ids = a[keep][:, keep].tocsr(), ids[keep]
    n, m = a.shape[0], a.nnz // 2
    upper = sp.triu(a, 1).tocoo()
    lonely = np.flatnonzero(np.diff(a.indptr) == 0)
    pairs = np.concatenate([np.stack([upper.row, upper.col], 1), np.stack([lonely, lonely], 1)])
    pd.DataFrame(pairs).to_csv(os.path.join(OUT, name + '.edges'), sep=' ', header=False, index=False)
    sp.save_npz(os.path.join(OUT, name + '.adj.npz'), a)
    twin_share, twin_max = twins(a)
    info = dict(n=int(n), m=int(m), density=m / (n * (n - 1) / 2), avg_degree=2 * m / n,
                bits_gib=n * ((n + 31) // 32) * 4 / 2**30, twin_share=round(twin_share, 4), twin_max=twin_max)
    for key, series in labels.items():
        values = series.reindex(ids).to_numpy().astype(str)
        _, groups = np.unique(values, return_inverse=True)
        np.save(os.path.join(OUT, f'{name}.{key}.labels.npy'), groups)
        info[f'groups_{key}'] = int(groups.max() + 1)
    return info


os.makedirs(OUT, exist_ok=True)
summary_path = os.path.join(OUT, 'graphs.json')
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
for name in sys.argv[1:] or list(SOURCES):
    summary[name] = prepare(name)
    json.dump(summary, open(summary_path, 'w'), indent=1)
    print(name, {k: round(v, 5) if isinstance(v, float) else v for k, v in summary[name].items()}, flush=True)
