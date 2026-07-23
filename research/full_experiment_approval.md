# Full-experiment approval preparation

Status: **design only; no final runner or results have been created**.

## Connected Erdos-Renyi policy

The current generator repeatedly samples `G(n,p)` until it obtains a connected
graph. Consequently, accepted graphs are drawn from ER conditional on
connectedness. The finite ER/BA degree match is nominal before conditioning;
accepted ER graphs can have a different realised average degree. Every graph
record must retain its attempt count and accepted seed, and final analysis must
use measured graph properties and report acceptance rates by cell.

### Option A — retain conditional connected-ER rejection sampling

- Preserves fixed `n` and makes every ordered source-destination pair reachable.
- Keeps the Dijkstra benchmark and routing-pair protocol identical across graph
  families.
- Introduces explicit connectivity-selection bias, which must be measured and
  discussed.

**Recommendation:** retain this as the default. It provides the clearest
fixed-size routing experiment if realised degree and attempt counts are
analysed.

### Option B — raise p above the connectivity threshold and retain rejection

This improves acceptance but changes the degree regimes and breaks the current
exact nominal match unless BA settings are redesigned too. Use only if the
scientific design is explicitly changed and rematched.

### Option C — use the largest connected component

This changes `n`, degree distribution, and graph identity in a model-dependent
way. It weakens ER/BA comparability and must not be used without approval.

### Option D — allow disconnected graphs and sample only reachable pairs

This conditions pair selection on component structure and gives cells different
reachable-pair populations. It complicates the Dijkstra and failure protocols
and must not be used without approval.

## Provisional full-setting review

No provisional value is silently frozen.

| Setting | Current value | Classification | Reason/action |
|---|---|---|---|
| Graph sizes | `100, 300, 1000` | Needs methodological approval | Technically feasible in bounded benchmarks; scientific size range still requires approval. |
| BA attachment values | `4, 8, 16` | Needs methodological approval | Provide three useful degree regimes, but their relevance to the paper must be approved. |
| ER probabilities | exact `2m(n-m)/[n(n-1)]` | Needs methodological approval | Formula and tests are exact, but the values follow the still-unapproved `n`, `m`, and connected-ER policy. |
| Graph repetitions | `20` per cell | Needs methodological approval | Gives 20 independent graph units per model/setting; confirm power and graph-level interval plan. |
| Ordered routing pairs | `1000` per graph | Needs methodological approval | Computationally feasible, but pairs are repeated observations within a graph, not independent replicates. |
| Embedding method | `dense_fruchterman_reingold_rescaled_v1` | Needs methodological approval | Implementation is consistent; scientific bias remains unresolved. |
| Embedding radius | `0.85` | Needs methodological approval | Material to Poincare rankings even though Euclidean rankings are scale-invariant. |
| Embedding iteration limit | `200` | Needs benchmark confirmation | Runtime is feasible; add convergence diagnostics before freezing scientific use. |
| Connected-generation limit | `50` | Needs benchmark confirmation | Valid and collision-audited; confirm acceptance rates in a small ER pilot before final freeze. |
| Master seeds | three fixed 32-bit values | Ready to freeze | Domain-separated and collision-audited over the complete configured grid. |
| Numerical tolerance | `1e-12` | Ready to freeze | Validated consistently by distance, routing, embedding, and tests. |
| Unit-disk epsilon | `1e-6` | Ready to freeze | A validated numerical safety margin independent of the chosen radius, which must remain strictly smaller than `1-epsilon`. |

The exact provisional ER probabilities (in `m=4,8,16` order) are:

| n | Exact finite-size-matched p values |
|---:|---|
| 100 | `0.077575757575757576`, `0.14868686868686870`, `0.27151515151515149` |
| 300 | `0.026399108138238574`, `0.052084726867335562`, `0.10131549609810479` |
| 1000 | `0.0079759759759759751`, `0.015887887887887888`, `0.031519519519519520` |

### Replication hierarchy

Graphs are the independent experimental units. Routing pairs inside one graph
are repeated, correlated observations. Final confidence intervals, tests, and
models must not treat all 360,000 pairs as independent graph replicates.
Recommended summaries first aggregate or model outcomes within each graph, then
compare graph-level values using paired/blocked methods across matched settings.
A hierarchical or cluster-bootstrap analysis may use pair-level data while
resampling graphs as the top-level units.

## Future checkpoint and output design

Recommended transparent format: canonical UTF-8 JSONL, one atomic file per
graph replicate, plus a run manifest. JSONL represents null failure/stretch
fields and repair metadata more safely than a wide CSV; a validated CSV export
can be produced later for analysis.

### Stable identities and records

Use graph ID
`<configuration-fingerprint>/<setting-index>/<model>/<replicate-index>` and a
record key `(graph_id, pair_index, method)`. Each method record must contain:

- configuration and implementation-source fingerprints;
- embedding method/version and runtime package versions;
- graph ID, model, setting label, `n`, `p` or `m`, replicate and attempt data;
- graph, embedding, and pair seeds;
- network measurements and graph-level distortion;
- pair index, source, destination, and method;
- success, failure type, route length, Dijkstra length, and successful-route
  stretch;
- every repair state field; and
- setup, route-only, repair, and serialization timing fields as applicable.

Graph-level metadata and distortion may be repeated in flat records for simple
joins or stored once in a companion graph record referenced by graph ID. The
schema choice must be frozen before the run.

### Atomicity, resume, and validation

1. Write a canonical run manifest containing configuration JSON, both
   fingerprints, schema versions, package versions, and approved method IDs.
2. Refuse to resume if any manifest fingerprint or schema differs.
3. Write each graph to `<graph_id>.jsonl.partial` in deterministic record order.
4. Validate unique keys, expected record count, required fields, finite numeric
   values, pair/method coverage, and route invariants before completion.
5. Flush and close the partial file, then atomically rename it to `.jsonl`.
6. Write a small completion marker containing graph ID, record count, byte size,
   and SHA-256 only after the rename succeeds.
7. On resume, skip only files with a valid completion marker and matching
   manifest. Remove no partial automatically; quarantine or explicitly restart
   an incomplete graph after logging the decision.
8. Prevent duplicates through the stable record key and an expected-key set per
   graph. Treat duplicates as fatal rather than silently overwriting.
9. Write failures to a separate append-only JSONL log with graph ID, stage,
   exception type/message, and deterministic retry decision.
10. Provide a dry-run mode that validates settings, seed collisions, output
    paths, manifest compatibility, and expected counts without generating a
    graph or result record.

The future runner should process one graph at a time and release all-pairs and
embedding data before the next graph. No complete final runner is implemented
by this review.

## Timing policy

Record these non-overlapping categories:

1. graph generation, including ER rejected attempts;
2. network measurement excluding the shared all-pairs traversal;
3. embedding;
4. shared precomputation, including all-pairs distances and coordinate-context
   preparation, with metric-specific preparation reported separately;
5. pure per-route forwarding for each method;
6. repaired-route total and the descriptive difference from ordinary
   hyperbolic routing; and
7. result validation and serialization.

Timing must use prepared validated coordinates, identical graph/coordinate/pair
inputs, `perf_counter_ns`, no one-time imports, and no console output inside the
timed region. Warm up each method, time enough repeated pairs for stable medians,
and rotate or counterbalance method order across graph replicates. Shared setup
must never be charged to only one routing method. Report setup totals and
route-only distributions separately. “Repair overhead” is path-dependent, so
the repaired-minus-ordinary difference is descriptive; also stratify repaired
routes by whether repair was attempted.

## Manual pre-commit checklist

- [ ] Confirm the custom embedding is intentional and one implementation is
  used at `n=30,100,300,1000`.
- [ ] Approve an embedding option and exact method/version, or retain the
  full-run block.
- [ ] Confirm finite BA matching `m(n-m)` and ER probability formula.
- [ ] Review seed domains and configured-grid collision test.
- [ ] Review the stable `2*asinh` Poincare formula and disk checks.
- [ ] Review one-step repair, physical backtracking, and result-state tests.
- [ ] Confirm the 12 unchanged dependency pins and one-time UTF-16-to-UTF-8
  conversion.
- [ ] Review `.gitattributes` text/LF and PNG/PDF binary policy without
  renormalising the repository.
- [ ] Read the methodology, embedding memo, workload review, and README.
- [ ] Confirm the complete test suite and deterministic smoke output.
- [ ] Review the bounded `n=300` and `n=1000` workload timings and memory.
- [ ] Review every modified and expected untracked file before staging.
- [ ] Confirm `experiment_config.py` and its test are untracked only because
  they have never been committed; do not “fix” this outside an intentional
  checkpoint commit.
- [ ] Confirm no `results/`, `outputs/`, final plots, agent files, or automation
  files exist in the proposed commit.
