# Mathematical background

The experiment compares shortest-path and local geometric routing on simple,
connected, undirected graphs. Euclidean greedy routing forwards to the neighbour
closest to the destination in Euclidean distance. Hyperbolic greedy routing uses
the same forwarding rule and the same coordinates, but measures distance in the
Poincare disk. Dijkstra routing supplies the unweighted global shortest-path
benchmark; it does not influence greedy forwarding decisions.

The exact formulas, stopping rules, repair semantics, graph-model matching, and
development-embedding limitations used by the implementation are recorded in
[`methodology_implementation.md`](methodology_implementation.md).
