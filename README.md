# Greedy-Routing-Poincare

A deterministic Python experiment for comparing Euclidean greedy routing,
Poincare-disk greedy routing, and one-step repaired hyperbolic routing against
an unweighted Dijkstra shortest-path benchmark. The experiment controls graph
generation, coordinates, source-destination pairs, stopping rules, and
tie-breaking so that the routing methods are compared on the same inputs.

## Current status

Development Stages 1-11 are implemented: validated experiment settings,
distance and disk utilities, connected ER and BA generation, network metrics,
a provisional development embedding, embedding distortion, Dijkstra and all
three routing variants, deterministic pair sampling, and a small integration
smoke runner.

The smoke run is pipeline validation only. It is not the full experiment, does
not produce paper results, and must not be used to claim that one routing metric
is superior.

## Methods

- Dijkstra shortest path: global unweighted benchmark.
- Euclidean greedy routing: local forwarding by Euclidean distance.
- Hyperbolic greedy routing: the same local rule using Poincare distance.
- Repaired hyperbolic routing: one route-history-based backtracking repair.

Both greedy metrics use exactly the same coordinate mapping. The current
development coordinates come from the project's deterministic dense
Fruchterman-Reingold force layout, using NumPy and a NetworkX adjacency matrix,
then centring and uniform rescaling into the unit disk. This is not a canonical
hyperbolic embedding. Its versioned identifier is
`dense_fruchterman_reingold_rescaled_v1`. Because the layout is Euclidean and
force-directed, it may favour Euclidean routing. It is suitable for pipeline
validation, but the final scientific embedding still requires approval.

## Experimental design and reproducibility

ER and BA settings are matched by finite-size expected average degree. For the
default NetworkX BA construction, the exact edge count is `m(n - m)`, so

```text
BA exact average degree = 2m(n - m) / n
ER expected average degree = p(n - 1)
p = [2m(n - m) / n] / (n - 1).
```

`2m` is only the large-`n` BA approximation. ER samples are retained only when
connected, so accepted graphs come from `G(n, p)` conditional on connectedness.
That conditioning can shift realised average degree; final comparisons must
record and analyse measured realised degree rather than relying only on the
matched expectation.

Graph generation, embedding, and source-destination sampling use separate
master-seed streams. Derived 32-bit seeds use versioned, domain-separated
BLAKE2s experiment identities; Python's process-randomised `hash()` is never
used. Every configured development and provisional-full seed use is checked for
collisions. The experiment also fixes node ordering, pair ordering, numerical
tolerances, and routing tie breaking. Exact settings and seed-derivation
metadata have canonical JSON serialization and a SHA-256 configuration
fingerprint in `code/experiment_config.py`. Structured smoke provenance also
records a SHA-256 fingerprint over the exact Stage 1-11 source modules.

## Repository structure

- `code/`: experiment configuration and implementation modules.
- `tests/`: deterministic unit and integration tests.
- `research/`: mathematical background, implementation methodology, and the
  pre-full-experiment audit.
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

The proposed full configuration remains provisional: 9 matched parameter
settings, 2 graph models, and 20 repetitions produce 360 graphs. At 1,000
ordered pairs per graph, that is 360,000 pair evaluations for each of Dijkstra,
Euclidean greedy, hyperbolic greedy, and repaired hyperbolic routing, or
1,440,000 total method executions. It also entails 360 embedding runs and
65,916,000 unordered-pair distortion evaluations. The grid contains 180 ER and
180 BA replicates; the 50-attempt ER cap implies at most 9,000 ER candidate
draws, or 9,180 total generator calls including one call per BA graph. The full
experiment must not run until the embedding method and full settings receive
explicit approval.

See `research/methodology_implementation.md` for formulas, validation rules,
repair semantics, reproducibility controls, and limitations. See
`research/pre_full_experiment_review.md` for the bounded performance benchmark
and pre-commit audit. The unresolved scientific embedding choice is compared in
`research/embedding_method_decision.md`; full-setting classifications,
connected-ER options, checkpointing, timing, and the manual review checklist
are in `research/full_experiment_approval.md`.
