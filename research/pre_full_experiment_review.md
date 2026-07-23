# Pre-full-experiment review

This review covers implemented Stages 1-11 only. The development smoke run and
the benchmark below validate software behaviour; they are not experiment
results and support no comparison claim about routing methods. No full run,
result dataset, or plot was produced.

## Audit findings and resolutions

### Critical

- The previous embedding delegated to NetworkX's size-dependent spring-layout
  dispatcher. At `n >= 500`, even the force path imports SciPy, which is not a
  project dependency, so the provisional `n=1000` grid would fail. The code now
  uses the versioned project-owned
  `dense_fruchterman_reingold_rescaled_v1` implementation for every graph size.
  A regression executes that same helper at `n=30`, `100`, `300`, and `1000`
  while forbidding SciPy imports. This is still only a development embedding
  and is not approved as the final scientific embedding.

### High

- ER probabilities had been matched to the asymptotic BA degree `2m`.
  NetworkX's star-initialised BA construction instead has exactly
  `m(n-m)` edges and finite-size average degree `2m(n-m)/n`. Configuration,
  labels, provenance, tests, and documentation now use the exact finite-size
  match. The BA initial star is passed explicitly rather than relying on a
  library default.
- The `acosh(1 + z)` Poincare formula could round a small positive distance to
  zero. Evaluation now uses the equivalent stable `2*asinh` form and factored
  boundary terms. Tests cover underflow-scale separation and near-boundary
  points without weakening open-disk validation.
- The smoke runner previously accepted any configuration named `development`.
  It now accepts only the exact canonical development configuration, with a
  regression against a relabelled full configuration.

### Medium

- Random streams now use versioned, domain-separated BLAKE2s identities that
  include setting, model, replicate, and ER attempt. The complete development
  and provisional-full grids are enumerated, including all permitted retries;
  no collision occurs among 10,204 combined configured seed uses. A 32-bit hash
  cannot promise collision freedom for arbitrary future configurations, so the
  explicit grid audit remains required whenever settings change.
- Configuration is immutable, canonically JSON-serializable, fingerprinted,
  and records embedding controls, numerical policy, tie-breaking, provisional
  values, seed derivation, and computed workloads. Graph metadata records the
  complete replicate identity and can regenerate identical nodes and edges.
- Diameter and average shortest-path length now share one all-pairs traversal;
  the smoke runner reuses the same topology-bound distance data for distortion.
  Dijkstra length is derived from its returned path instead of executing a
  second shortest-path search.
- Coordinates are validated and immutably snapshotted once per graph and metric
  for repeated routing. Euclidean gets one context; ordinary and repaired
  hyperbolic routing fairly share one Poincare context. No forwarding choices
  or graph distances are precomputed for a greedy method.
- Routing rejects boolean/non-integral node aliases, validates public result
  state combinations, and has deterministic generated-graph tests for walk
  edges, step limits, tolerance boundaries, input immutability, single repair,
  and absence of shortest-path leakage.
- Distortion rejects duplicate coordinates and tests its scale minimiser,
  determinism, pair count, topology coverage, and immutability.
- Structured smoke provenance records configuration, fingerprint, runtime
  versions, an exact implementation-source fingerprint, graph, embedding, pair
  sampler, coordinates, pairs, and method results. Reproducibility is tested in
  separate processes with different `PYTHONHASHSEED` values.
- Repository hygiene now includes UTF-8 requirements, a minimal line-ending and
  binary policy, a data-directory policy, ignored generated outputs and local
  AI-tool metadata, and documentation that separates the introductory graph
  demo from experimental generation.

### Low

- The mathematical-background placeholder, source-level repair semantics,
  embedding bias, connected-ER caveat, and development/full distinction are now
  documented. Repeated test `sys.path` bootstrapping remains deliberately
  unchanged because the flat `code/` layout does not justify packaging solely
  for aesthetics.

## Corrected provisional workload

The proposed full grid contains 9 `(n, m)` settings, 2 graph models, and 20
replicates: 360 graphs. Its computed workload is:

- 360 embeddings;
- 360,000 unique ordered source-destination pairs per method;
- 360,000 executions each of Dijkstra, Euclidean greedy, hyperbolic greedy, and
  repaired hyperbolic routing;
- 1,440,000 total method executions;
- 65,916,000 unordered coordinate pairs for distortion;
- at most 9,000 connected-ER candidate draws plus 180 BA draws.

Connectedness rejection means accepted ER graphs follow `G(n,p)` conditional on
connectedness. Exact finite-size matching is therefore nominal before
conditioning; final analysis must record acceptance rates and compare measured
realised average degree in every cell.

## Bounded performance benchmark

Environment: CPython 3.14.0, Windows 11 build 26200, AMD64 Family 25 Model 116,
NetworkX 3.6.1, and NumPy 2.4.6. Timings are local wall-clock observations, not
portable performance guarantees. The benchmark generated no files. It used the
first development ER and BA cells with 25 pairs each, plus one permitted
provisional medium BA graph (`n=100`, `m=4`) with 50 pairs. Routing medians used
10 repetitions for development and 7 for the medium case.

| Stage | Dev ER, n=30 | Dev BA, n=30 | Medium BA, n=100 |
|---|---:|---:|---:|
| Graph generation | 1.407 ms | 0.744 ms | 0.667 ms |
| Shared all-pairs distances | 0.565 ms | 0.350 ms | 3.369 ms |
| Network metrics after shared paths | 0.549 ms | 0.284 ms | 1.199 ms |
| Embedding setup | 14.110 ms | 3.977 ms | 41.687 ms |
| Distortion after shared paths | 2.614 ms | 2.602 ms | 27.516 ms |
| Pair sampling | 0.060 ms | 0.054 ms | 0.079 ms |
| Euclidean coordinate setup | 0.294 ms | 0.443 ms | 0.968 ms |
| Poincare coordinate setup | 0.258 ms | 0.426 ms | 0.955 ms |

Median route time, excluding the separately reported graph/coordinate setup:

| Method | Dev ER, n=30 | Dev BA, n=30 | Medium BA, n=100 |
|---|---:|---:|---:|
| Dijkstra | 0.0326 ms | 0.0339 ms | 0.1301 ms |
| Euclidean greedy | 0.0872 ms | 0.0836 ms | 0.1348 ms |
| Hyperbolic greedy | 0.1066 ms | 0.1075 ms | 0.1751 ms |
| Repaired hyperbolic greedy | 0.1278 ms | 0.1310 ms | 0.2871 ms |
| Repaired minus ordinary hyperbolic | 0.0212 ms | 0.0234 ms | 0.1121 ms |

The final row is an observed path-dependent difference, not a constant repair
cost: repaired routing may follow a different physical walk.

On the medium graph, one shared all-pairs computation plus both consumers took
31.755 ms versus 34.266 ms for two independent computations (1.08x faster),
with identical outputs. Twenty-five prepared hyperbolic routes took 4.212 ms;
raw per-route coordinate revalidation took 17.252 ms (4.10x route-only speedup,
or about 3.34x including the one-time 0.955 ms Poincare setup). These bounded
measurements confirm the repeated-work corrections but do not establish that
the full grid is affordable. The dense force layout, all-pairs graph distances,
and all-pairs distortion remain scaling risks.

### Required medium and large workload benchmarks

Two additional provisional BA workload benchmarks used the middle configured
attachment value, `m=8`, replicate 0, the full configured 200 embedding
iterations, and 20 deterministic ordered pairs. Each graph ran in a fresh
process so graph-level data from one case could not be retained by the next.
Route entries are medians of seven repetitions. These are workload
measurements only, not experimental results.

| Stage | BA, n=300, m=8 | BA, n=1000, m=8 |
|---|---:|---:|
| Graph generation | 3.737 ms | 8.751 ms |
| Shared all-pairs distances | 30.293 ms | 377.842 ms |
| Network metrics after shared paths | 9.467 ms | 42.066 ms |
| Embedding | 537.930 ms | 7.471 s |
| Distortion | 242.753 ms | 2.755 s |
| Euclidean coordinate preparation | 3.778 ms | 21.638 ms |
| Poincare coordinate preparation | 3.498 ms | 13.194 ms |
| Pair sampling | 0.083 ms | 0.162 ms |
| Total bounded benchmark | 1.098 s | 11.143 s |

| Route-only median per pair | BA, n=300, m=8 | BA, n=1000, m=8 |
|---|---:|---:|
| Dijkstra | 0.5246 ms | 1.8849 ms |
| Euclidean greedy | 0.3034 ms | 0.3212 ms |
| Hyperbolic greedy | 0.4055 ms | 0.3879 ms |
| Repaired hyperbolic greedy | 0.6423 ms | 0.6500 ms |
| Repaired minus ordinary hyperbolic | 0.2368 ms | 0.2621 ms |

The benchmark seed identities were derived through the configured seed
derivation: graph/embedding/pair seeds were
`1030712552/1166833787/667531295` at `n=300` and
`3195717173/1351316406/3028879141` at `n=1000`. Both graphs were generated on
their first BA attempt. Connected-ER benchmarks were optional and were not run:
the two required BA cases safely characterised the principal dense-layout,
all-pairs, and route-timing costs, while ER acceptance behaviour remains a
separate pilot requirement before approval.

Windows process counters observed a peak physical working set of about 58 MiB
for the `n=300` repeat and 159 MiB for `n=1000`. At `n=1000`, the working set
was about 48 MiB after graph generation, 86 MiB after stored all-pairs
distances, 88 MiB after embedding, and 87 MiB after distortion. Peak private
committed address space was about 564 MiB, much of it runtime reservation
rather than resident memory. These figures are approximate and machine
specific; processing one graph at a time remains mandatory.

## Full-workload estimate

The exact configured counts are 360 graphs, 360,000 ordered pairs per method,
1,440,000 total method executions, at most 9,000 connected-ER candidate calls
plus 180 BA calls, and 65,916,000 unordered distortion-pair calculations.

A direct size-stratified projection assigns 120 graphs to each of `n=100`,
`n=300`, and `n=1000`. It uses the measured time at each size rather than a
linear extrapolation across sizes. The resulting raw serial point estimate is
about 34.9 minutes: 16.1 minutes embedding, 6.0 minutes distortion, 11.7 minutes
route-only work, and about 1.1 minutes for the remaining measured setup. It does
not include final validation, checkpoint serialization, ER rejection retries,
or contention from other processes.

| Scenario | Approximate serial time | Interpretation |
|---|---:|---|
| Optimistic | about 30 minutes | favourable run-to-run variation and minimal output overhead |
| Central | about 40-45 minutes | measured point estimate plus checkpoint, validation, and serialization overhead |
| Conservative | about 1.5-2 hours | slower topology cases, ER retries, OS contention, and substantially heavier output overhead |

These are planning ranges, not guarantees. The dense force-directed embedding
costs `O(I n^2)` time and `O(n^2)` working memory for fixed iteration count
`I`. Unweighted all-pairs shortest paths cost `O(n(n+e))` time and `O(n^2)`
stored distances. Exhaustive distortion costs `O(n^2)` time and `O(n^2)` ratio
storage in the present implementation. One Dijkstra route costs
`O(e + n log n)` with a binary heap; greedy and repaired routing scan current
neighbours at each hop and are bounded by the step limit, with work dependent
on topology and walk length. These differences explain why extrapolation is
not assumed linear.

The measured peak resident memory supports a planning allowance of roughly
250-300 MiB when graphs are processed serially, while allowing more address
space for the Python runtime. A transparent pair-method JSONL dataset would
contain 1.44 million records. At roughly 0.6-1.2 KiB per record it would occupy
about 0.9-1.7 GiB, plus comparatively small manifests, graph records, failure
logs, and completion markers. Reserving 3-4 GiB of free disk allows partial
atomic writes and validation.

The current Option-A grid appears computationally practical on this machine,
but this does not approve it scientifically. There is no present reason to
reduce it automatically. If a pilot reveals pressure, reducing pairs from
1,000 to 500 preserves the number of independent graph replicates but does not
reduce embedding or exhaustive-distortion cost. Reducing the 20 graph
replicates should require a statistical-power review. Option C would at least
add a second embedding and routing comparison, and its cost cannot be estimated
reliably until the hyperbolic objective and optimiser are specified.

## Requirements and encoding review

The 12 dependency pins before and after the `requirements.txt` conversion are
textually identical; no package was added, removed, or repinned. The current
file is valid UTF-8, has a final newline, and retains NetworkX 3.6.1, the version
whose layout behaviour was inspected. An offline
`pip install --dry-run --no-index -r requirements.txt` succeeds in the current
environment. Git may display the historical UTF-16-to-UTF-8 change as binary;
that is expected for this one-time conversion and is not a reason to restore
UTF-16.

The `.gitattributes` policy keeps Python, Markdown, and text/requirements files
reviewable with LF endings and marks PNG and PDF as binary. No renormalization
or repository-wide line-ending rewrite is part of this review.

## Readiness and remaining decisions

Stages 1-11 are suitable for a human-controlled commit after the final clean
verification recorded in the audit handoff. The full experiment is not ready
to run. It remains blocked on explicit approval of:

1. Option A, B, or C in `embedding_method_decision.md`;
2. the full-grid values classified as needing methodological approval in
   `full_experiment_approval.md`;
3. the conditional connected-ER policy and a bounded acceptance pilot for the
   attempt limit; and
4. implementation and validation of the designed checkpointing runner before
   any full execution.
