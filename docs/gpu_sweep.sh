#!/bin/bash
# GPU (--ls-kernel global) против cc_cpu --threads 16 на одной траектории: ровно 20 итераций.
# У GPU четыре прогона (сиды 5..8), первый отбрасывается из-за создания контекста CUDA; CPU идёт с
# сидами 6..8, поэтому f обязаны совпасть построчно.
# Готовые точки пропускаются, так что прерванный прогон можно просто запустить снова.
cd "$(dirname "$0")/.."
OUT=docs/data/gpu.csv
[ -f $OUT ] || echo "series,n,k,pop,run,gpu_f,gpu_s,cpu_f,cpu_s" > $OUT
runs() { awk '/^run /{sub("f=","",$3); sub("s$","",$6); print $2","$3","$6}'; }
point() {  # series n k pop
  [ "$(grep -c "^$1,$2,$3,$4," $OUT)" = 3 ] && return
  sed -i "/^$1,$2,$3,$4,/d" $OUT
  local c="--n $2 --density 0.33 --graph-seed 7 --k $3 --pop $4 --iters 20 --early-stop 1000 --no-verify"
  local g=$(./bin/cc_gpu $c --seed 5 --runs 4 --ls-kernel global | runs | awk -F, '$1>0')
  local p=$(./bin/cc_cpu $c --seed 6 --runs 3 --threads 16 | runs | awk -F, '{print $1+1","$2","$3}')
  join -t, <(echo "$g") <(echo "$p") | awk -F, -v q="$1,$2,$3,$4" '{print q","$1","$2","$3","$4","$5}' >> $OUT
  echo "$1 n=$2 k=$3 pop=$4 $(date +%T)"
}
# В отчёте: n128, pop6000, k128. Остальные серии лежат в CSV как дополнительные данные.
for n in 1000 2000 4000 6000 8000 12000 16000 24000 32000; do point n128 $n 3 128; done
for n in 1000 2000 4000 6000 8000 12000; do point n1024 $n 3 1024; done
for p in 32 64 128 256 512 1024 2048 4096; do point pop2000 2000 3 $p; done
for p in 32 64 128 256 512 1024 2048; do point pop6000 6000 3 $p; done
for k in 2 3 4 6 8 12 16 24 32; do point k128 2000 $k 128; done
for k in 2 3 4 8 16 32; do point k1024 2000 $k 1024; done
