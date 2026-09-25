# cuda_graph_clustering — корреляционная кластеризация графа на GPU

Пакет решает задачу **нестрогой k-корреляционной кластеризации**: разбить вершины
графа не более чем на k кластеров так, чтобы число «разногласий» было минимально.
Разногласие — пара вершин, которая соединена ребром, но разнесена по разным
кластерам, или не соединена, но попала в один кластер.

Алгоритм — **PBILS** (population based iterated local search) из
[BIGADIL/graph_correlation_clustering](https://github.com/BIGADIL/graph_correlation_clustering),
перенесённый на CUDA. Есть и многопоточная CPU-версия с тем же интерфейсом. Пакет
собирается только из файлов этого репозитория.

## Постановка

| Что             | Как                                                                   |
|-----------------|-----------------------------------------------------------------------|
| Число кластеров | не более k, пустые кластеры допустимы; k любое до 64                  |
| Граф            | невзвешенный, неориентированный, без петель, до 32 767 вершин на GPU  |
| Целевая функция | число разногласий, минимизация                                        |

## Как это устроено

Целевая функция считается не по парам вершин, а через матрицу связей
G = A·Z (G[v][c] — число соседей v в кластере c):

```
f = m + Σ_c C(n_c, 2) − tr(ZᵀAZ)
Δf(v: a → c) = (n_c − n_a + 1) − 2·(G[v][c] − G[v][a])
```

Выигрыш любого хода считается за O(1), после хода G правится за O(n). Саму G
пакет считает побитово, через popcount по упакованным строкам матрицы смежности.
Арифметика общая для CPU и GPU и лежит в `include/cc_math.hpp`.

**GPU:** один блок CUDA на особь популяции, потоки блока делят между собой вершины
и вместе ищут лучший ход. **CPU:** та же схема, особь получает поток из пула
(`--threads`).

У каждой особи свой поток ГПСЧ, поэтому ответ не зависит от числа потоков, а при
одинаковых параметрах и `--seed` **CPU и GPU проходят одну и ту же траекторию** и
выдают одно и то же решение.

## Сборка

```bash
make            # CPU-часть; GPU-часть, если найден nvcc
make cpu        # CPU-версия и тесты, CUDA не нужна
make gpu        # CUDA-версия (GPU_ARCH=sm_89 make gpu)
make test       # проверки корректности
```

`GPU_ARCH` должен соответствовать карте, иначе драйвер будет компилировать ядра из
PTX при каждом запуске: `sm_75` Turing, `sm_80` A100, `sm_86` Ampere GeForce,
`sm_89` Ada, `sm_90` Hopper.

### Windows

Нужны **VS Build Tools** (нагрузка «Разработка классических приложений на C++») и
**CUDA Toolkit**. Из окружения `vcvars64.bat`:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

mkdir bin obj

cl /nologo /O2 /std:c++17 /EHsc /arch:AVX2 /DNDEBUG /Iinclude ^
   src\cc_graph.cpp src\cc_pbils_cpu.cpp src\main_cpu.cpp ^
   /Fe:bin\cc_cpu.exe /Foobj\

cl /nologo /O2 /std:c++17 /EHsc /arch:AVX2 /DNDEBUG /Iinclude ^
   src\cc_graph.cpp src\cc_pbils_cpu.cpp src\cc_verify.cpp ^
   /Fe:bin\cc_verify.exe /Foobj\

nvcc -O3 -std=c++17 -Iinclude --generate-code arch=compute_89,code=sm_89 ^
   -Xcompiler "/O2 /std:c++17 /EHsc /arch:AVX2 /DNDEBUG" ^
   -o bin\cc_gpu.exe cuda\cc_pbils_gpu.cu cuda\main_gpu.cu ^
   src\cc_graph.cpp src\cc_pbils_cpu.cpp
```

## Запуск

```bash
# сгенерировать G(n, p) по сиду и решить
./bin/cc_gpu --n 3000 --density 0.33 --graph-seed 7 --k 3 --pop 128 --runs 10
./bin/cc_cpu --n 3000 --density 0.33 --graph-seed 7 --k 3 --pop 128 --threads 0

# решить граф из файла результатов оригинала или сохранить инстанс для него
./bin/cc_gpu --graph-json path/to/result.json
./bin/cc_gpu --n 600 --density 0.33 --graph-seed 1 --save-graph-json inst.json
```

Все опции — `--help`. Найденное значение целевой функции по умолчанию перепроверяется
пересчётом по парам вершин; при расхождении программа завершается с кодом 2.

## Что проверено

* `make test` (`bin/cc_verify.exe` на Windows): 6676 проверок формулы и инкрементальных
  обновлений против определения по парам, под GCC/Clang и MSVC.
* CPU и GPU при одном `--seed` дают одинаковые f и число итераций, так же и
  `--ls-kernel shared` против `--ls-kernel global`.
* Ответ CPU-версии не зависит от `--threads`.
* Значение целевой функции совпадает с тем, что выдаёт оригинал на его же инстансах.
* Ядра собраны и запущены на RTX 4070 Ti SUPER (compute 8.9), CUDA 13.3.

## Производительность

Замеры против оригинала и GPU против CPU — в [docs/benchmarks.md](docs/benchmarks.md).

## Структура

```
include/cc_math.hpp     целевая функция, дельта хода, ГПСЧ; общий код хоста и устройства
include/cc_graph.hpp    граф: биты, генератор по сиду, ввод-вывод, чтение JSON оригинала
include/cc_pbils.hpp    параметры, состояние решения, PBILS
include/cc_gpu.hpp      объявление GPU-решателя
include/cc_cli.hpp      общий интерфейс командной строки
src/                    CPU-версия и тесты
cuda/                   ядра CUDA
docs/                   замеры: отчёт, графики, данные, скрипты
```
