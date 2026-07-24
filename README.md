# Greedy-Routing-Poincare

A deterministic Python experiment for comparing Euclidean greedy routing,
Poincare-disk greedy routing, and one-step repaired hyperbolic routing against
an unweighted Dijkstra shortest-path benchmark. The experiment controls graph
generation, coordinates, source-destination pairs, stopping rules, and
tie-breaking so that the routing methods are compared on the same inputs.

## Current status

Development Stages 1-13 are implemented: validated experiment settings,
distance and disk utilities, connected ER and BA generation, network metrics,
embedding distortion, Dijkstra and all three routing variants, deterministic
pair sampling, and a small integration smoke runner. The approved co-equal
embeddings are deterministic standard two-dimensional Hydra at curvature -1
and classical two-dimensional MDS.

The development smoke and embedding-feasibility pilot are pipeline validation
only. They are excluded from the full experiment, produce no scientific
results, and must not be used to claim that one routing metric is superior.

## Methods

- Dijkstra shortest path: global unweighted benchmark.
- Euclidean greedy routing: local forwarding by Euclidean distance.
- Hyperbolic greedy routing: the same local rule using Poincare distance.
- Repaired hyperbolic routing: one route-history-based backtracking repair.

Each graph has five coordinate conditions: one standard two-dimensional Hydra
embedding (`kappa = 1`, curvature -1) centred by a Poincare-disk isometry, and
four uniform rescalings of one classical two-dimensional MDS base embedding to
maximum radii 0.50, 0.70, 0.85, and 0.95. The MDS radii are nested sensitivity
conditions, not graph or embedding replicates.

Both embedding families reuse one all-pairs shortest-path distance matrix. The
same sampled ordered pairs are evaluated under every coordinate condition;
Dijkstra runs once per pair, while all three greedy methods run under every
condition. For Euclidean routing on the nested MDS radii, the absolute
tie/progress tolerance scales by `radius / 0.95`, preserving the same decisions
under uniform scaling. The older `dense_fruchterman_reingold_rescaled_v1`
layout remains available only for the development smoke and is not an approved
experimental condition.

## Experimental design and reproducibility

ER and BA settings are matched by finite-size expected average degree. For the
default NetworkX BA construction, the exact edge count is `m(n - m)`, so

```text
BA exact average degree = 2m(n - m) / n
ER expected average degree = p(n - 1)
p = 2m(n - m) / [n(n - 1)].
```

`2m` is only the large-`n` BA approximation. ER samples are retained only when
connected, so accepted graphs come from `G(n, p)` conditional on connectedness.
That conditioning can shift realised average degree; final comparisons must
record and analyse measured realised degree rather than relying only on the
matched expectation.

Graph generation, embedding provenance, and source-destination sampling use
separate master-seed streams. Hydra and MDS are deterministic given the shared
distance-matrix input. Derived seeds and the ordered-pair sampler use versioned,
domain-separated BLAKE2s identities; Python's process-randomised `hash()` and
library sampling behaviour are not used. The frozen data-generation,
analysis-plan, and combined methodology hashes are exposed by
`code/experiment_config.py`.

Approved experimental graphs use non-boolean integer node IDs from `0` through
`n - 1`. Low-level embedding fixtures may instead use homogeneous string
labels. Mixed labels, booleans, and custom object labels are rejected before
embedding-input fingerprinting, so unstable object representations cannot
silently affect experimental provenance.

## Repository structure

- `code/`: experiment configuration and implementation modules.
- `tests/`: deterministic unit and integration tests.
- `figures/`: existing explanatory figure generators and assets.
- `data/`: policy-controlled location for small intentional inputs only; the
  current pipeline needs no input dataset.

`code/graph_generation.py` is the experimental ER/BA generator.
`code/graph_construction.py` is preserved only as an introductory figure/demo
generator and is not used by the experiment. Generated experiment data belongs
in the ignored `results/` or `outputs/` directories, not in `data/`.

## Environment setup

Python 3.11 or newer is required by the pinned dependencies; NetworkX 3.6.1
explicitly excludes Python 3.14.1. The audited development environment uses
Python 3.14.0. From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Tests

Run the complete automated test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

## Development smoke experiment

Run the small deterministic Stage 11 integration experiment with:

```powershell
.\.venv\Scripts\python.exe code\run_development_smoke.py
```

It uses only development settings, at most five fixed pairs per graph, and
writes no results or plots.

## Approved-embedding feasibility pilot

Run the small in-memory Hydra/MDS feasibility pilot with:

```powershell
.\.venv\Scripts\python.exe -B code\run_embedding_feasibility.py
```

It uses three excluded development graphs and five pairs per graph. It prints
diagnostics and workload counts, writes no outputs or plots, and does not run
the full configuration.

The frozen full configuration has 9 matched parameter settings, 2 graph
models, and 20 repetitions, producing 360 graphs. At 1,000
ordered pairs per graph, that is 360,000 Dijkstra runs and 1,800,000 runs of
each greedy method across the five coordinate conditions: 5,760,000 routing
executions in total. It also entails 360 Hydra runs, 360 MDS base runs, 1,440
nested MDS radius transformations, 65,916,000 unordered-pair distortion
evaluations per metric condition, and 461,412,000 across the seven prescribed
metric conditions. The grid
contains 180 ER and 180 BA replicates; the 50-attempt ER cap implies at most
9,000 ER candidate draws, or 9,180 total generator calls including one call per
BA graph.
