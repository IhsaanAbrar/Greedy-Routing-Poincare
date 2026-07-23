# Embedding method decision

Status: **methodological approval required before any full experiment**.

This memo separates implementation consistency from scientific approval. The
current code has one deterministic embedding implementation at every configured
graph size, but that fact does not make it the approved final scientific
embedding.

## Implementation consistency finding

The installed NetworkX 3.6.1 API exposes
`spring_layout(..., method="force")`, but its implementation still branches on
graph size. For fewer than 500 nodes it uses the dense force implementation; at
500 or more nodes it first calls `to_scipy_sparse_array` and then the sparse
force implementation. SciPy is not an approved or declared dependency, and a
direct `n=500`, one-iteration probe failed with `ModuleNotFoundError: scipy`.

The project therefore retains one project-owned dense Fruchterman-Reingold
implementation for every size. It has the versioned identifier
`dense_fruchterman_reingold_rescaled_v1` and method identifier
`project_dense_fruchterman_reingold_v1`. It uses:

- stable node and edge ordering;
- an unweighted NetworkX adjacency matrix;
- NumPy `RandomState`/MT19937 initialisation from the dedicated embedding seed;
- the same force update, linear cooling, and convergence threshold at every
  graph size;
- the configured iteration limit;
- arithmetic-mean centring and uniform maximum-radius rescaling; and
- projection only as a final open-unit-disk safeguard.

Tests execute the same helper at `n=30`, `100`, `300`, and `1000`, forbid any
SciPy import, reproduce identical coordinates from identical inputs, verify
that graphs are not mutated, and verify strict disk containment. NetworkX
version 3.6.1 and the explicit method identifiers are recorded in provenance;
the configuration and source fingerprints cover the method choice and its
implementation.

## Option A — rescaled force-directed embedding

### Scientific interpretation

The experiment compares routing metrics on coordinates constructed by a
Euclidean force-directed graph layout and then rescaled into the Poincare disk.
Any conclusion is conditional on that construction.

### Bias risk

The construction directly optimises Euclidean layout forces, so it may favour
Euclidean greedy routing. It is not a canonical hyperbolic embedding. Uniform
radius rescaling leaves Euclidean neighbour rankings unchanged but can change
Poincare rankings, making the approved radius part of the method.

### Complexity and cost

Implementation complexity is low because it already exists and is tested. Its
dense force step costs `O(iterations * n^2)` time and `O(n^2)` temporary memory.
The measured 200-iteration embedding times were approximately 0.042 s at
`n=100`, 0.538 s at `n=300`, and 7.471 s at `n=1000` on the audit machine.

### Reproducibility requirements

Freeze the method/version, NetworkX and NumPy versions, node/edge order, seed,
iteration limit, convergence threshold, radius, disk epsilon, and centring and
rescaling rules. Preserve the source fingerprint.

### Effect on the research question and methodology

This option answers: “How do the routing rules compare on this specific
Euclidean force-directed coordinate construction?” It cannot by itself support
a general claim that Euclidean geometry is intrinsically superior to
hyperbolic geometry. The paper must state that limitation prominently.

### Recommended use

Use for pipeline validation and, if explicitly approved, for a conditional
force-layout experiment. Do not present it as geometry-neutral evidence.

## Option B — hyperbolic-specific embedding

### Scientific interpretation

Coordinates are inferred or optimised using an objective designed for
hyperbolic geometry. This is closer to the project motivation and relevant
literature.

### Bias risk

A hyperbolic-designed objective may favour Poincare routing. Euclidean routing
would then be evaluated on coordinates not designed for Euclidean performance,
so its comparison would also remain conditional on the embedding.

### Complexity and cost

Implementation and validation complexity are high. Before code is written, the
method must specify the exact loss/objective, initialisation, optimiser,
learning-rate or step policy, convergence rule, restart policy, constraint or
projection rule, failure handling, and deterministic numerical environment.
Computational cost cannot be estimated credibly until those choices are fixed.

### Reproducibility requirements

Version and record every optimisation choice, package version, seed, stopping
condition, iteration trace summary, convergence/failure status, and final
coordinate validation. Determinism and graph/coordinate immutability require
new focused tests at every configured size.

### Effect on the research question and methodology

This option answers: “How do the routing rules compare on coordinates designed
for hyperbolic geometry?” It strengthens alignment with the motivation but does
not separate routing-metric effects from embedding-design effects.

### Recommended use

Use only after an exact literature-supported method is approved. Do not
implement or select an optimiser implicitly.

## Option C — sensitivity analysis with both embeddings

### Scientific interpretation

Run the controlled routing comparison once under the approved force-directed
construction and once under an approved hyperbolic-specific construction.
Assess whether graph-level conclusions persist, reverse, or weaken across
embedding methods.

### Bias risk

Each embedding retains its own directional bias, but making embedding an
explicit experimental factor exposes rather than hides that sensitivity. The
analysis must not pool the two embeddings as interchangeable replicates.

### Complexity and cost

This has the largest implementation and compute burden: Option B must first be
specified and validated, and embedding/routing work roughly doubles. Graphs and
source-destination pairs should be shared across embeddings so comparisons are
paired at the graph level.

### Reproducibility requirements

Maintain independent, versioned embedding configurations and seed domains,
while reusing the exact graph replicate and pair identities. Record the
embedding factor in every output key and checkpoint. Analyse paired graph-level
contrasts and interactions with graph family, size, and degree regime.

### Effect on the research question and methodology

This option answers the strongest question: “Are conclusions about routing
metrics robust to the coordinate-construction method?” It distinguishes a
general routing effect from an embedding-specific effect more credibly than
either single-embedding design.

### Recommended use

**Recommended for final scientific evidence**, subject to approval of one exact
hyperbolic-specific method and the expanded workload. If resources do not allow
Option C, Option A may be approved only with explicitly conditional claims;
Option B alone has the symmetric hyperbolic-design bias.

## Decision required

The full experiment remains blocked until the user approves Option A, B, or C.
Approval must also freeze the exact method version and all embedding controls.
This memo makes no approval on the user's behalf and introduces no new
embedding implementation.
