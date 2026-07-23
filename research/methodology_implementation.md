# Methodology implementation

This document records the development implementation through Stage 11. The
development smoke run validates the pipeline; it does not produce final paper
evidence or support conclusions about which routing metric is better.

## Distances and unit-disk handling

For two finite two-dimensional coordinates `x` and `y`, Euclidean distance is

```text
d_E(x, y) = ||x - y||_2.
```

For coordinates strictly inside the open unit disk, Poincare distance is

```text
d_D(x, y) = arcosh(
    1 + 2 ||x - y||_2^2 /
        ((1 - ||x||_2^2) (1 - ||y||_2^2))
).
```

Distance evaluation rejects non-finite coordinates, coordinates that are not
two-dimensional, and points on or outside the unit circle. It does not modify
inputs. The implementation evaluates the mathematically equivalent stable form

```text
d_D(x, y) = 2 asinh(
    ||x - y||_2 /
    sqrt((1 - ||x||_2^2) (1 - ||y||_2^2))
).
```

This avoids the cancellation in `arcosh(1 + z)` that can otherwise round a
small positive distance to zero. Boundary factors are evaluated as
`(1 - ||x||)(1 + ||x||)` for additional near-boundary stability; invalid
inputs are still rejected rather than clamped or projected.

Projection is an embedding operation. A finite two-dimensional point whose norm
is at or beyond the configured safe radius `1 - disk_epsilon` is moved radially
to that radius. A zero vector remains zero, avoiding division by zero.

## Graph generation and connectedness

The implementation generates simple undirected NetworkX graphs with integer
nodes `0` through `n - 1`.

- Erdos-Renyi graphs use `G(n, p)`. An attempt is accepted only when the graph is
  connected. Each retry receives its own deterministic, domain-separated seed
  from the configured graph master seed and complete replicate/attempt identity.
  Generation stops at the configured attempt limit and never substitutes the
  largest connected component.
- Barabasi-Albert graphs use `BA(n, m)` with the replicate's deterministic graph
  seed. Connectedness is checked as an invariant even though valid BA generation
  should produce a connected graph.

Graph metadata records the model, `n`, `p` or `m`, replicate index, base graph
seed, accepted attempt seed, and attempt count.

### Finite-size ER/BA degree matching

NetworkX's default `barabasi_albert_graph(n, m)` starts from a star on `m + 1`
nodes and then adds `n - m - 1` nodes with `m` edges each. It therefore produces
exactly

```text
E_BA = m(n - m)
```

edges, giving exact finite-size average degree

```text
d_BA = 2m(n - m) / n.
```

The experiment matches the ER expectation `d_ER = p(n - 1)` to this exact
finite-size BA value:

```text
p = [2m(n - m) / n] / (n - 1).
```

The familiar `2m` value is retained only as a labelled large-`n`
approximation. Because disconnected ER draws are rejected, the accepted ER
sample is conditional on connectedness. This can shift realised average degree
away from the unconditional expectation. Every generated graph is therefore
measured, and the final analysis must report and use realised average degree.

### Seed domains and provenance

Graph generation, embedding initialization, and source-destination sampling use
separate master-seed domains. A replicate identity includes the configuration,
setting index, graph model, and replicate index; ER attempt identity additionally
includes the zero-based attempt index. Stable domain-separated derivation is
used instead of Python's process-randomised `hash()`: canonical JSON is hashed
with the versioned `blake2s-32-domain-separated-v1` scheme. Every returned seed
is within NetworkX's accepted 32-bit range. No 32-bit hash can guarantee
collision freedom for every possible future identity, so the complete configured
development and provisional-full seed sets, including every permitted ER retry,
are enumerated and checked explicitly. An identical identity reproduces the
identical seed.

Configuration serialization records the schema version, matched settings,
provisional status, numerical controls, tie-breaking metadata, embedding
metadata, workload counts, and seed-derivation metadata. Per-graph provenance
adds the setting label, all three replicate seeds, the accepted ER attempt seed,
and the coordinate and pair-sampling controls. A SHA-256 fingerprint identifies
the canonical configuration JSON. Software versions and benchmark environment
are recorded in the smoke result and pre-full-experiment review.

## Network measurements

Measurements require a non-empty, connected, undirected graph. Stable output
fields use these definitions:

- `number_of_vertices`: node count.
- `number_of_edges`: undirected edge count.
- `average_degree`: arithmetic mean of all node degrees.
- `maximum_degree`: largest node degree.
- `population_degree_variance`: population variance of node degree (`ddof=0`).
- `average_clustering_coefficient`: NetworkX average clustering.
- `diameter`: largest graph shortest-path distance.
- `average_shortest_path_length`: mean shortest-path distance over unordered
  distinct node pairs.

Embedding distortion is calculated separately after coordinates exist.
The smoke pipeline computes one validated all-pairs unweighted-distance table
per graph and reuses it for both network metrics and distortion. The cached data
is tied to an immutable node/edge topology snapshot and is rejected if reused
with a different graph. It is shared graph-level measurement work only and is
never exposed to any greedy-routing decision.

## Provisional development embedding

No documented coordinate generator existed in the repository. The Stage 5
fallback is therefore used:

1. Sort nodes into a stable order.
2. Run the project's deterministic dense Fruchterman-Reingold force layout,
   using NumPy and a NetworkX adjacency matrix, with the dedicated Part 1
   embedding seed and configured iteration count (100 in development).
3. Centre all coordinates at their arithmetic mean.
4. Uniformly rescale them so the largest norm is the configured embedding
   radius, `0.85` in the development configuration.
5. Apply the unit-disk projection safeguard with the configured epsilon.

Every node receives one finite two-dimensional coordinate. The graph is not
mutated. Coordinates depend on graph topology and the embedding seed, never on
source-destination pairs or routing outcomes. Euclidean and hyperbolic greedy
routing receive the same coordinate mapping.

The Fruchterman-Reingold layout is a force-directed development embedding, not
a canonical hyperbolic embedding. It is suitable for validating the software pipeline but
requires methodological approval before full-scale results can be treated as
paper evidence. Any later routing result is conditional on this coordinate
construction method. Because it optimises a Euclidean force-directed layout, it
may favour Euclidean greedy routing. Its versioned identifier is
`dense_fruchterman_reingold_rescaled_v1`; the algorithm, implementation method,
radius, iteration count, embedding seed, disk epsilon, and tolerance are
recorded in graph provenance. The project-owned dense implementation avoids an
otherwise implicit SciPy dependency for medium and large NetworkX force layouts.
It is the same implementation at `n=30`, `n=100`, `n=300`, and `n=1000`;
there is no graph-size dispatch threshold. NetworkX 3.6.1's explicit
`method="force"` still converts graphs with at least 500 nodes to a SciPy sparse
array, so it cannot provide the required dependency-free implementation at
every configured size. Euclidean and Poincare routing both consume the same
coordinate snapshot; only their distance functions differ.

## Embedding distortion

Distortion uses every unordered pair of distinct vertices `i < j`. Let `g_ij`
be graph shortest-path distance and `h_ij` be Poincare distance. Define

```text
q_ij = h_ij / g_ij
alpha = sum(q_ij) / sum(q_ij^2)
```

This scale is the unique minimiser of the relative squared-error objective
`J(alpha) = sum((alpha*q_ij - 1)^2)`: setting
`J'(alpha) = 2(alpha*sum(q_ij^2) - sum(q_ij))` to zero gives the stated
formula, while `J''(alpha) = 2*sum(q_ij^2) > 0` for valid distinct
coordinates.

The primary measure is

```text
mean_relative_distortion =
    mean(abs(alpha * h_ij - g_ij) / g_ij).
```

The supplementary root-mean-square measure is

```text
rmse_relative_distortion =
    sqrt(mean(((alpha * h_ij - g_ij) / g_ij)^2)).
```

The result also records `alpha` and the pair count `n(n - 1)/2`. Distortion is a
graph-level property and does not use sampled routing pairs.

## Dijkstra benchmark

The benchmark explicitly uses a NetworkX Dijkstra operation on the unweighted
graph. It returns a valid path and its edge count, defined as `len(path) - 1`.
Dijkstra has global graph information and is only the shortest-path reference;
its paths or distances never influence greedy forwarding choices.

## Shared greedy-routing rule

At a current node `v` with target `t`:

1. Evaluate the selected metric from every neighbour to `t`.
2. Find the minimum neighbour distance.
3. Treat values within `numerical_tolerance` of that minimum as tied.
4. Select the tied candidate with the smallest integer node ID.
5. If the selected node was visited, stop with `cycle`.
6. If it is not closer than the current node by more than the tolerance, stop
   with `local_minimum`.
7. Otherwise traverse the edge and continue.

A defensive step limit yields `step_limit`; it supplements rather than replaces
cycle detection. The Euclidean and hyperbolic wrappers use the same core,
coordinates, candidates, stopping rules, tolerance, tie-breaking rule, and step
limit. Only their metric differs.

Classification precedence is explicit: an attempted revisit is classified as
`cycle` before the lack-of-strict-progress test can classify `local_minimum`.

For repeated pair routing, the experiment runner validates and snapshots the
coordinate mapping once per graph and metric. One Euclidean context serves all
Euclidean routes; one Poincare context is shared by ordinary and repaired
hyperbolic routing. Each context is bound to the exact graph object, a frozen
topology snapshot, and its distance function, so incompatible reuse is rejected.
This shared work validates inputs but does not precompute forwarding choices or
give any routing method graph-distance information. Direct fixture calls may
still pass raw mappings and receive the same per-call validation semantics.

The paper's "misleading geometric move" is not a routing-time stopping rule.
Detecting it would require global graph information and, if later added, must be
labelled as post-hoc analysis.

## One-step hyperbolic repair

Repaired routing begins as ordinary strict hyperbolic greedy routing. Repair is
considered only on the first local minimum or attempted revisit.

1. Record the initial failure.
2. If there is no preceding walk vertex, mark repair unavailable.
3. Otherwise traverse the real edge back to the preceding vertex and count it.
4. Keep every explored vertex excluded, including the failed branch.
5. Rank the remaining valid neighbours by Poincare distance, using the same
   tolerance and smallest-node-ID tie break.
6. Traverse the best alternative even when it is not strictly closer. This is
   the sole escape step.
7. Resume ordinary strict hyperbolic greedy forwarding.
8. Never perform a second repair. A later failure is terminal.

The intentional backtrack is not classified as a cycle. The result separately
records the initial failure, whether repair was attempted, whether an
alternative existed, whether repair succeeded, the final failure, and the full
physical walk. Repair uses route history only, never Dijkstra information or
global lookahead.

If the initial failure occurs at the source, there is no previously traversed
edge to backtrack over and therefore no physical repair attempt can occur. That
case records `repair_attempted=False`, `repair_attempt_count=0`, and repair
`repair_alternative_existed=False`; its final failure is `repair_unavailable`.
It is distinct from an attempted backtrack for which no alternative neighbour
exists, which records one attempted repair and final failure `repair_failed`.

## Route length, stretch, and controls

Route length is the number of traversed graph edges, including a repair
backtrack:

```text
route_length = len(physical_walk) - 1.
```

For a successful greedy route only,

```text
stretch = greedy_route_length / Dijkstra_shortest_path_length.
```

Failed routes store no stretch. Development integration uses unique ordered
source-destination pairs with distinct endpoints. The sampler sorts integer
node IDs, maps the `n(n - 1)` possible ordered pairs to stable integer indices,
and samples those indices without replacement using Python's
`random.Random(pair_seed)`. This sampler is recorded as
`python_random_sample_without_replacement_v1`. Every method receives identical
pairs, and both geometric metrics receive the identical coordinate mapping.

Part 1 separates master seeds for graph generation, embedding initialization,
and pair sampling. Each graph replicate receives deterministic derived seeds;
ER retries additionally receive deterministic attempt seeds.

All pair-level methods receive the same sampled pair and the same coordinate
mapping. Successful greedy walks are checked to be no shorter than their
Dijkstra benchmark, and failed greedy routes have no stretch. The structured
smoke result has its own schema version, stable field ordering, JSON-safe
values, and a SHA-256 fingerprint over the exact Stage 1-11 source modules.
Reproducibility is tested both twice within one process and through separate
Python processes with different `PYTHONHASHSEED` values.

## Introductory demo versus experimental generation

`code/graph_generation.py` is the deterministic experimental ER/BA generator
described above. The older `code/graph_construction.py` is retained as an
introductory figure/demo generator. It is not imported by the experiment, and
its example graph or layout settings are not experimental settings.

## Development versus final evidence

The Stage 11 smoke runner uses only the development configuration, one ER graph,
one BA graph, and at most five pairs per graph. It checks implementation
invariants and prints descriptive outcomes without aggregation or statistical
claims. The full configuration is not executed. Before the final experiment,
the embedding choice, full workload, and final reporting protocol require
approval.

The provisional full grid has 9 matched settings (`n` in `100, 300, 1000` and
`m` in `4, 8, 16`), 2 models, and 20 graph repetitions: 360 graph replicates.
At 1,000 unique ordered pairs per graph this yields 360,000 pair evaluations per
method. Dijkstra, Euclidean greedy, hyperbolic greedy, and repaired hyperbolic
routing each run 360,000 times, for 1,440,000 total method executions. Pair count
and total method-execution count are deliberately reported separately. The grid
also requires 360 embeddings and 65,916,000 unordered-pair distortion
evaluations. There are 180 ER and 180 BA replicates. With at most 50 candidate
draws per ER replicate and one per BA replicate, the worst-case generator-call
count is `180 * 50 + 180 = 9,180`. These values, all full settings, and the
embedding are provisional until approved.

The supported pinned environment requires Python 3.11 or newer; NetworkX 3.6.1
explicitly excludes Python 3.14.1. The audit environment uses Python 3.14.0.
Exact package versions are in `requirements.txt`. Determinism is asserted within
that pinned software environment; bit-for-bit equivalence across arbitrary
Python, NumPy, or NetworkX versions is not claimed.

## Replication and statistical unit

Graph replicates are the independent experimental units. Ordered routing pairs
sampled within one graph are repeated observations conditional on that graph,
not additional independent graph replicates. Final confidence intervals and
method comparisons must preserve this hierarchy, for example by computing
graph-level summaries or by using a hierarchical model with graph as the
cluster. Treating all pair-level records as independent would understate
uncertainty.

The unresolved embedding choice, connected-ER policy, provisional setting
classifications, checkpoint record design, and timing protocol are recorded in
`embedding_method_decision.md` and `full_experiment_approval.md`. The full
experiment remains blocked until the decisions marked there receive explicit
approval.
