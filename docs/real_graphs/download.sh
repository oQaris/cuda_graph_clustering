#!/bin/bash
# Скачивает реальные графы в data/real/raw (около 2 ГБ, в git не попадают). Уже скачанные файлы пропускаются.
# Источники: Netzschleuder (networks.skewed.de), SNAP (snap.stanford.edu), gnn-benchmark (Shchur et al.), OGB, DGL.
cd "$(dirname "$0")/../.."
RAW=data/real/raw
mkdir -p $RAW
get() {  # url [имя файла]
  local file=$RAW/${2:-$(basename "$1")}
  [ -s "$file" ] || curl -sfL -o "$file" "$1" || { rm -f "$file"; echo "FAIL $1"; }
}

NZ=https://networks.skewed.de/net
get $NZ/karate/files/78.csv.zip karate.csv.zip
for net in dolphins football polbooks polblogs sp_primary_school; do get $NZ/$net/files/$net.csv.zip; done
get $NZ/sp_high_school/files/proximity.csv.zip sp_high_school.csv.zip

SNAP=https://snap.stanford.edu/data
get $SNAP/email-Eu-core.txt.gz
get $SNAP/email-Eu-core-department-labels.txt.gz
for set in lastfm_asia deezer_europe git_web_ml twitch_gamers; do get $SNAP/$set.zip; done
for file in loc-gowalla_edges amazon0302 email-EuAll web-Stanford gplus_combined; do get $SNAP/$file.txt.gz; done
get $SNAP/bigdata/communities/com-dblp.ungraph.txt.gz

GNN=https://github.com/shchur/gnn-benchmark/raw/master/data/npz
for set in cora citeseer pubmed cora_full amazon_electronics_photo amazon_electronics_computers ms_academic_cs \
  ms_academic_phy; do get $GNN/$set.npz; done

get http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
get https://data.dgl.ai/dataset/reddit.zip
ls -la $RAW
