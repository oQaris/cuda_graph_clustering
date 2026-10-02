# Реальные графы

Графы реальных систем, на которых решатель проверяется помимо случайных G(n, p): 19 графов до 40 тысяч вершин
с разметкой групп и 9 графов от 107 до 317 тысяч вершин. Ещё два больших графа, по тегам NUS-WIDE и Stack Overflow
из данных оригинала, описаны в [разделе 4 отчёта о замерах](../benchmarks.md#4-большие-графы).

Под постановку подходит любой невзвешенный неориентированный граф без петель. Ограничения задаёт устройство
решателя:

* **размер** — битовая матрица смежности n²/8 байт и состояние популяции должны поместиться в память карты: на 16 ГБ
  это около 340 тысяч вершин (граф в 317 тысяч вершин занял 12,6 ГиБ);
* **число кластеров** — не больше 64; на графах больше 32 767 вершин быстрые ядра есть только для k = 2 и k = 3,
  при k ≥ 4 работает общее ядро с 32-битной G, и такой прогон идёт часами.

## Как воспроизвести

```bash
docs/real_graphs/download.sh                  # ~2 ГБ в data/real/raw, уже скачанное пропускается
python docs/real_graphs/prepare.py            # списки рёбер, разметка и сводка в data/real/graphs
python docs/real_graphs/run.py karate reddit  # решить и сверить с разметкой; --k=K, --backend=cpu
./bin/cc_gpu --edges data/real/graphs/com_dblp.edges --k 2   # или напрямую
```

`data/real/` в git не попадает. Сводка по графам — [`docs/data/real_graphs_stats.json`](../data/real_graphs_stats.json),
результаты — [`docs/data/real_graphs.json`](../data/real_graphs.json). Для `prepare.py` и `run.py` нужны numpy,
scipy, pandas и scikit-learn.

## Как построены

Из каждого источника берётся только структура связей, дальше одни и те же правила:

* направление рёбер отбрасывается: ребро {u, v} есть, если в исходных данных есть u → v или v → u;
* повторы сливаются, петли удаляются, веса и времена не читаются;
* вершины — все объекты источника, включая изолированные: в файле `.edges` изолированная вершина записана петлёй,
  которую загрузчик отбрасывает, но вершину сохраняет;
* нумерация: вершина — ранг номера объекта среди всех номеров в файле, так же нумерует `--edges`, и в том же
  порядке лежит разметка `<граф>.<разметка>.labels.npy`.

Отступления от этих правил:

* **polblogs** — взята наибольшая компонента связности (1 222 блога из 1 490), как принято в литературе;
* **школы SocioPatterns** — исходно это временные сети контактов лицом к лицу, записанных датчиками с шагом 20 с;
  ребро проводится, если за всё время был хотя бы один контакт;
* **Google+** — номера пользователей 21-значные и в 64 бита не помещаются, поэтому перед записью они
  перенумерованы по возрастанию строки.

Доля близнецов (вершин с одинаковыми замкнутыми окрестностями) в графах до 40 тысяч вершин не больше 4,5 %, кроме
Cora (8 %) и CiteSeer (20 %); у ogbn-arxiv — 0,9 %. Графы по тегам устроены иначе: на NUS-WIDE 193 734 вершины
образуют всего 4 487 классов близнецов.

## Разметка групп

Там, где у вершин есть известная принадлежность к группе (фракция, конференция, класс школы, тематика статьи), она
сохранена как разметка. С ней найденное разбиение сравнивается двумя мерами: ARI и NMI. Кроме того, для разметки
считается f: сколько разногласий дало бы разбиение ровно по группам. Разметка — это метаданные, а не оптимум задачи:
на всех графах, решённых с k, равным числу групп, у найденного разбиения f ниже, чем у разметки. Поэтому низкий ARI
говорит о том, что группы графа не похожи на плотные кластеры, а не о слабости решателя.

## Графы до 40 тысяч вершин

| граф | вершины и рёбра | n | m | плотность | разметка (групп) |
|---|---|---:|---:|---:|---|
| karate [1] | члены клуба карате, общение вне клуба | 34 | 78 | 13,9 % | фракция после раскола клуба (2) |
| dolphins [2] | афалины залива Даутфул-Саунд, частые совместные появления | 62 | 159 | 8,4 % | — |
| football [3] | команды NCAA Division IA, игры сезона 2000 года | 115 | 613 | 9,4 % | конференция (12) |
| polbooks [4] | книги о политике США (Amazon, 2004), частые совместные покупки | 105 | 441 | 8,1 % | либеральная, консервативная, нейтральная (3) |
| polblogs [5] | политические блоги США (2004), гиперссылки | 1 222 | 16 714 | 2,2 % | либеральный или консервативный (2) |
| email-Eu-core [6] | сотрудники европейского научного института, хотя бы одно письмо | 1 005 | 16 064 | 3,2 % | отдел (42) |
| sp_primary_school [7] | ученики и учителя начальной школы в Лионе (2009), контакты | 242 | 8 181 | 28,1 % | класс, учителя отдельно (11) |
| sp_high_school [8] | ученики девяти классов лицея в Марселе (2013), контакты | 329 | 5 613 | 10,4 % | класс (9) |
| cora [9] | статьи по машинному обучению, цитирование | 2 708 | 5 278 | 0,14 % | тема (7) |
| citeseer [9] | научные статьи, цитирование | 3 312 | 4 536 | 0,08 % | тема (6) |
| amazon_photo [10] | товары Amazon, часто покупаются вместе | 7 650 | 119 081 | 0,41 % | категория товара (8) |
| lastfm_asia [11] | пользователи LastFM из Азии (2020), взаимная подписка | 7 624 | 27 806 | 0,10 % | страна (18) |
| amazon_computers [10] | товары Amazon, часто покупаются вместе | 13 752 | 245 861 | 0,26 % | категория товара (10) |
| coauthor_cs [10] | авторы по информатике (Microsoft Academic Graph), соавторство | 18 333 | 81 894 | 0,05 % | основная область (15) |
| pubmed [9] | статьи PubMed о диабете, цитирование | 19 717 | 44 324 | 0,02 % | экспериментальный, 1-го или 2-го типа (3) |
| cora_full [12] | расширенная Cora, цитирование | 19 793 | 63 421 | 0,03 % | тема (70) |
| deezer_europe [11] | пользователи Deezer из Европы (2020), взаимная подписка | 28 281 | 92 752 | 0,02 % | пол по имени (2) |
| coauthor_physics [10] | авторы-физики (Microsoft Academic Graph), соавторство | 34 493 | 247 962 | 0,04 % | основная область (5) |
| github [13] | разработчики GitHub (2019), взаимная подписка | 37 700 | 289 003 | 0,04 % | веб или машинное обучение (2) |

У football конференции в исходных данных относятся к сезону 2001 года, а игры — к 2000-му; три пары команд сыграли
дважды, здесь такие рёбра слиты [14]. Исправленная разметка есть в файлах Evans
([figshare](https://figshare.com/articles/American_College_Football_Network_Files/93179)), здесь взята исходная, как
в большинстве работ. У email-Eu-core 25 571 направленное ребро сливается в 16 064. У cora_full групп 70, больше
предела в 64 кластера, поэтому решать её можно только с k ≤ 64.

## Графы больше 100 тысяч вершин

«Битовая матрица» — объём одной только матрицы смежности; память карты при k = 2 и популяции 128 см. в результатах.

| граф | вершины и рёбра | n | m | средняя степень | битовая матрица | разметка (групп) |
|---|---|---:|---:|---:|---:|---|
| gplus [15] | пользователи Google+, поделившиеся кругами; связи из кругов | 107 614 | 12 238 285 | 227 | 1,35 ГиБ | — |
| twitch_gamers [16] | пользователи Twitch (2018), взаимная подписка | 168 114 | 6 797 557 | 81 | 3,29 ГиБ | язык (21), контент 18+ (2), партнёрство (2) |
| ogbn_arxiv [17] | статьи arXiv по информатике, цитирование | 169 343 | 1 157 799 | 14 | 3,34 ГиБ | раздел arXiv cs.* (40) |
| loc_gowalla [18] | пользователи геосоциальной сети Gowalla, дружба | 196 591 | 950 327 | 10 | 4,50 ГиБ | — |
| reddit [19] | посты Reddit (сентябрь 2014), общий комментатор | 232 965 | 57 307 946 | 492 | 6,32 ГиБ | сабреддит (41) |
| amazon0302 [20] | товары Amazon (2 марта 2003), часто покупаются вместе | 262 111 | 899 792 | 7 | 8,00 ГиБ | — |
| email_euall [21] | адреса почты европейского института (2003–2005), хотя бы одно письмо | 265 214 | 364 481 | 3 | 8,19 ГиБ | — |
| web_stanford [22] | страницы stanford.edu (2002), гиперссылки | 281 903 | 1 992 636 | 14 | 9,25 ГиБ | — |
| com_dblp [23] | авторы DBLP, хотя бы одна общая статья | 317 080 | 1 049 866 | 7 | 11,70 ГиБ | — |

Направленные в источнике рёбра после слияния: Google+ 13 673 453 → 12 238 285, ogbn-arxiv 1 166 243 → 1 157 799,
amazon0302 1 234 877 → 899 792, email-EuAll 420 045 → 364 481, web-Stanford 2 312 497 → 1 992 636. У Reddit в файле
DGL каждое ребро записано в обе стороны (114 615 892 записи). У DBLP есть сообщества по местам публикации, но они
пересекаются и разбиением не являются, поэтому разметкой не взяты.

## Результаты

Популяция 128, `--seed 5`, остальные параметры по умолчанию (ранний останов после 6 итераций без рекорда); RTX 4070
Ti SUPER. Результаты сняты до появления GWW, то есть с `--gww 0`; `run.py` теперь идёт с GWW 0,1 по умолчанию.
Малые графы — k по числу групп разметки, время второго из двух прогонов (в первый входит создание контекста CUDA).
Большие — k = 2, один прогон.

| граф | k | ARI | NMI | время |
|---|---:|---:|---:|---:|
| sp_high_school | 9 | 0,981 | 0,981 | 3 мс |
| football | 12 | 0,906 | 0,931 | 2 мс |
| karate | 2 | 0,772 | 0,677 | 1 мс |
| polblogs | 2 | 0,772 | 0,675 | 3 мс |
| sp_primary_school | 11 | 0,758 | 0,876 | 2 мс |
| polbooks | 3 | 0,535 | 0,542 | 1 мс |
| amazon_photo | 8 | 0,389 | 0,488 | 1,6 с |
| email-Eu-core | 42 | 0,313 | 0,590 | 71 мс |
| coauthor_cs | 15 | 0,307 | 0,470 | 196 с |
| lastfm_asia | 18 | 0,272 | 0,446 | 6,2 с |
| amazon_computers | 10 | 0,203 | 0,409 | 7,2 с |
| cora | 7 | 0,200 | 0,289 | 0,23 с |
| pubmed | 3 | 0,163 | 0,150 | 5,6 с |
| citeseer | 6 | 0,088 | 0,101 | 0,37 с |

| граф | итерация | итераций | последний рекорд | до останова | разрезано рёбер | память карты |
|---|---:|---:|---:|---:|---:|---:|
| gplus | 1,7 с | 21 | 15 | 37 с | 4,1 % | 1,7 ГиБ |
| twitch_gamers | 5,2 с | 16 | 10 | 85 с | 11,5 % | 3,8 ГиБ |
| ogbn_arxiv | 4,9 с | 7 | 1 | 36 с | 3,3 % | 3,8 ГиБ |
| loc_gowalla | 6,8 с | 17 | 11 | 119 с | 5,8 % | 5,1 ГиБ |
| reddit | 12,6 с | 7 | 1 | 92 с | 4,9 % | 7,0 ГиБ |
| amazon0302 | 16,8 с | 35 | 29 | 591 с | 6,4 % | 8,7 ГиБ |
| email_euall | 11,3 с | 8 | 2 | 92 с | 12,8 % | 8,9 ГиБ |
| web_stanford | 20,2 с | 8 | 2 | 167 с | 1,6 % | 10,1 ГиБ |
| com_dblp | 28,9 с | 50 | 44 | 1450 с | 7,6 % | 12,6 ГиБ |

* Там, где группы — плотные подграфы (классы школы, конференции, два лагеря блогов), решение их почти
  восстанавливает. На разреженных графах цитирования и покупок группы плотными кластерами не являются, и совпадение
  слабое, хотя f решения ниже, чем у разметки.
* При k = 2 на разреженном графе f = C(S₀, 2) + C(S₁, 2) − m + 2 · разрез: решатель ищет почти равные половины
  (размеры отличаются не больше чем на 61 вершину) с малым разрезом. Случайное деление пополам разрезало бы около
  половины рёбер.
* В отличие от графа по тегам NUS-WIDE, на большинстве больших графов рекорд растёт до 10–44-й итерации.
* На Google+ две итерации заняли 4,4 с на GPU и 802 с на CPU в 16 потоков, с одинаковыми рекордами.
* Не посчитаны: cora_full, deezer_europe, coauthor_physics, github и k = 3 на больших графах; `run.py` для них готов.

## Что не вошло

* **Facebook100** (сети дружбы 100 университетов, используются в работе о LambdaCC [24]) открыто распространять
  нельзя: набор выложен только архивом на archive.org.
* **Facebook page-page** (SNAP, MUSAE): файл `facebook_large.zip` сейчас отдаёт 404.
* Графы больше примерно 340 тысяч вершин (com-Amazon, LiveJournal, Pokec) не помещаются на 16 ГБ.

## Источники

Лицензии и просьбы о цитировании — на страницах источников.

1. W. W. Zachary. An information flow model for conflict and fission in small groups. *J. Anthropological Research*
   33(4), 1977. Netzschleuder, [karate/78](https://networks.skewed.de/net/karate).
2. D. Lusseau et al. The bottlenose dolphin community of Doubtful Sound features a large proportion of long-lasting
   associations. *Behav. Ecol. Sociobiol.* 54, 2003. [Netzschleuder](https://networks.skewed.de/net/dolphins).
3. M. Girvan, M. E. J. Newman. Community structure in social and biological networks. *PNAS* 99(12), 2002.
   [Netzschleuder](https://networks.skewed.de/net/football).
4. V. Krebs. The political books network, не опубликовано; разметка M. E. J. Newman.
   [Netzschleuder](https://networks.skewed.de/net/polbooks).
5. L. A. Adamic, N. Glance. The political blogosphere and the 2004 U.S. election: divided they blog. *LinkKDD*, 2005.
   [Netzschleuder](https://networks.skewed.de/net/polblogs).
6. H. Yin, A. R. Benson, J. Leskovec, D. F. Gleich. Local higher-order graph clustering. *KDD*, 2017.
   [SNAP](https://snap.stanford.edu/data/email-Eu-core.html).
7. J. Stehlé et al. High-resolution measurements of face-to-face contact patterns in a primary school. *PLoS ONE*
   6(8), 2011. [Netzschleuder](https://networks.skewed.de/net/sp_primary_school).
8. R. Mastrandrea, J. Fournet, A. Barrat. Contact patterns in a high school: a comparison between data collected using
   wearable sensors, contact diaries and friendship surveys. *PLoS ONE* 10(9), 2015; подсеть proximity.
   [Netzschleuder](https://networks.skewed.de/net/sp_high_school).
9. P. Sen et al. Collective classification in network data. *AI Magazine* 29(3), 2008 (Cora, CiteSeer); G. Namata et al.
   Query-driven active surveying for collective classification. *MLG*, 2012 (PubMed).
   [gnn-benchmark](https://github.com/shchur/gnn-benchmark).
10. O. Shchur, M. Mumme, A. Bojchevski, S. Günnemann. Pitfalls of graph neural network evaluation. *R2L @ NeurIPS*,
    2018. [gnn-benchmark](https://github.com/shchur/gnn-benchmark).
11. B. Rozemberczki, R. Sarkar. Characteristic functions on graphs: birds of a feather, from statistical descriptors to
    parametric models. *CIKM*, 2020. SNAP: [LastFM Asia](https://snap.stanford.edu/data/feather-lastfm-social.html),
    [Deezer Europe](https://snap.stanford.edu/data/feather-deezer-social.html).
12. A. Bojchevski, S. Günnemann. Deep Gaussian embedding of graphs. *ICLR*, 2018.
    [gnn-benchmark](https://github.com/shchur/gnn-benchmark).
13. B. Rozemberczki, C. Allen, R. Sarkar. Multi-scale attributed node embedding. arXiv:1909.13021, 2019.
    [SNAP](https://snap.stanford.edu/data/github-social.html).
14. T. S. Evans. Clique graphs and overlapping communities. *J. Stat. Mech.*, P12037, 2010.
15. J. McAuley, J. Leskovec. Learning to discover social circles in ego networks. *NIPS*, 2012.
    [SNAP](https://snap.stanford.edu/data/ego-Gplus.html).
16. B. Rozemberczki, R. Sarkar. Twitch Gamers: a dataset for evaluating proximity preserving and structural role-based
    node embeddings. arXiv:2101.03091, 2021. [SNAP](https://snap.stanford.edu/data/twitch_gamers.html).
17. W. Hu et al. Open Graph Benchmark: datasets for machine learning on graphs. *NeurIPS*, 2020.
    [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv).
18. E. Cho, S. A. Myers, J. Leskovec. Friendship and mobility: user movement in location-based social networks. *KDD*,
    2011. [SNAP](https://snap.stanford.edu/data/loc-Gowalla.html).
19. W. L. Hamilton, R. Ying, J. Leskovec. Inductive representation learning on large graphs. *NIPS*, 2017.
    [DGL](https://data.dgl.ai/dataset/reddit.zip).
20. J. Leskovec, L. A. Adamic, B. A. Huberman. The dynamics of viral marketing. *ACM TWEB* 1(1), 2007.
    [SNAP](https://snap.stanford.edu/data/amazon0302.html).
21. J. Leskovec, J. Kleinberg, C. Faloutsos. Graph evolution: densification and shrinking diameters. *ACM TKDD* 1(1),
    2007. [SNAP](https://snap.stanford.edu/data/email-EuAll.html).
22. J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney. Community structure in large networks: natural cluster sizes and the
    absence of large well-defined clusters. *Internet Mathematics* 6(1), 2009.
    [SNAP](https://snap.stanford.edu/data/web-Stanford.html).
23. J. Yang, J. Leskovec. Defining and evaluating network communities based on ground-truth. *ICDM*, 2012.
    [SNAP](https://snap.stanford.edu/data/com-DBLP.html).
24. N. Veldt, D. F. Gleich, A. Wirth. A correlation clustering framework for community detection. *WWW*, 2018.
