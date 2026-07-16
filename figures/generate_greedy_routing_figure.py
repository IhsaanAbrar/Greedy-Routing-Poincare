"""Generate a full-network illustration of greedy routing."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
from PIL import Image


FIGURES_DIR = Path(__file__).resolve().parent
OUTPUT_PNG = FIGURES_DIR / "greedy_routing_rule.png"
OUTPUT_PDF = FIGURES_DIR / "greedy_routing_rule.pdf"
SEED = 23

ROUTE = ["s", "r1", "r2", "v", "u", "r5", "r6", "t"]
ROUTE_EDGES = list(zip(ROUTE[:-1], ROUTE[1:]))

# The route positions are fixed; a seeded, very small jitter makes the surrounding
# network organic while keeping the mathematical routing decisions reproducible.
BASE_POSITIONS = {
    "s": (0.55, 2.75),
    "r1": (1.85, 3.55),
    "r2": (3.10, 2.85),
    "v": (4.30, 3.55),
    "u": (5.45, 2.95),
    "r5": (6.65, 3.45),
    "r6": (7.85, 2.80),
    "t": (9.15, 3.10),
    "a": (1.05, 0.75),
    "b": (0.95, 4.85),
    "c": (2.35, 5.15),
    "d": (2.35, 1.25),
    "e": (3.35, 4.75),
    "f": (3.55, 0.65),
    "w1": (4.55, 5.10),
    "w2": (4.65, 1.05),
    "g": (5.75, 4.70),
    "h": (5.85, 1.05),
    "i": (7.10, 4.85),
    "j": (7.25, 0.85),
}

EXTRA_EDGES = [
    # Alternatives at each greedy-routing decision.
    ("s", "a"), ("s", "b"),
    ("r1", "c"), ("r1", "d"),
    ("r2", "e"), ("r2", "f"),
    ("v", "w1"), ("v", "w2"),
    ("u", "g"), ("u", "h"),
    ("r5", "i"), ("r5", "j"),
    ("r6", "i"), ("r6", "j"),
    # Connections that make the background a coherent network.  The lower
    # chain is shorter in hop count than the highlighted greedy route.
    ("a", "d"), ("a", "f"), ("d", "f"),
    ("f", "w2"), ("f", "h"), ("w2", "h"),
    ("h", "j"), ("j", "t"),
    ("b", "c"), ("c", "e"), ("e", "w1"),
    ("w1", "g"), ("g", "i"), ("i", "t"),
    ("c", "r2"), ("e", "v"), ("g", "r5"),
]


def build_positions():
    """Return deterministic node coordinates."""
    rng = np.random.default_rng(SEED)
    positions = {}
    for node, point in BASE_POSITIONS.items():
        if node in ROUTE:
            positions[node] = np.asarray(point, dtype=float)
        else:
            positions[node] = np.asarray(point, dtype=float) + rng.uniform(-0.07, 0.07, 2)
    return positions


def build_graph():
    graph = nx.Graph()
    graph.add_nodes_from(BASE_POSITIONS)
    graph.add_edges_from(ROUTE_EDGES)
    graph.add_edges_from(EXTRA_EDGES)
    return graph


def distance_to_target(node, positions):
    return float(np.linalg.norm(positions[node] - positions["t"]))


def validate_greedy_route(graph, positions):
    """Assert that every drawn move is the local greedy choice."""
    decisions = []
    for current, selected in ROUTE_EDGES:
        neighbours = list(graph.neighbors(current))
        best = min(neighbours, key=lambda node: distance_to_target(node, positions))
        if best != selected:
            raise ValueError(
                f"Invalid greedy step at {current}: expected {selected}, but {best} is closer to t"
            )
        decisions.append(
            (current, selected, distance_to_target(selected, positions), len(neighbours))
        )

    greedy_hops = len(ROUTE_EDGES)
    shortest_hops = nx.shortest_path_length(graph, "s", "t")
    if shortest_hops >= greedy_hops:
        raise ValueError("The background graph should contain a shorter hop-count route")
    return decisions, shortest_hops


def draw_nodes(graph, positions, ax):
    background = [node for node in graph if node not in ROUTE and node not in {"w1", "w2"}]
    nx.draw_networkx_nodes(
        graph, positions, nodelist=background, node_size=165,
        node_color="#e3e5e6", edgecolors="#70757a", linewidths=0.75, ax=ax,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=ROUTE[1:-1], node_size=220,
        node_color="#d7e4ed", edgecolors="#315f7d", linewidths=1.0, ax=ax,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=["w1", "w2"], node_size=205,
        node_color="#eeeeee", edgecolors="#62676b", linewidths=0.9, ax=ax,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=["s"], node_size=330,
        node_color="#79a986", edgecolors="#315c3c", linewidths=1.2, ax=ax,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=["t"], node_size=350,
        node_color="#c96b67", edgecolors="#7c3532", linewidths=1.2, ax=ax,
    )
    # A stronger outline distinguishes the current and selected local vertices.
    nx.draw_networkx_nodes(
        graph, positions, nodelist=["v", "u"], node_size=255,
        node_color=["#c9dce8", "#a9c9dc"], edgecolors="#244f6c",
        linewidths=1.5, ax=ax,
    )


def draw_route_arrows(positions, ax):
    for start, end in ROUTE_EDGES:
        arrow = FancyArrowPatch(
            positions[start], positions[end],
            arrowstyle="-|>", mutation_scale=12.5,
            linewidth=2.8, color="#2f6f9f",
            shrinkA=9.5, shrinkB=10.5, zorder=5,
        )
        ax.add_patch(arrow)


def draw_figure(graph, positions):
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 12,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.8), facecolor="white")
    ax.set_facecolor("white")

    nx.draw_networkx_edges(
        graph, positions, edge_color="#c2c6c9", width=0.85, alpha=0.95, ax=ax,
    )
    local_edges = [("v", node) for node in graph.neighbors("v")]
    nx.draw_networkx_edges(
        graph, positions, edgelist=local_edges,
        edge_color="#8b9297", width=1.15, ax=ax,
    )
    draw_route_arrows(positions, ax)
    draw_nodes(graph, positions, ax)

    label_specs = {
        "s": (r"$s$", (-0.02, 0.34)),
        "t": (r"$t$", (0.02, 0.34)),
        "v": (r"$v$", (-0.02, 0.30)),
        "u": (r"$u$", (0.00, -0.34)),
        "w1": (r"$w_1$", (0.34, 0.02)),
        "w2": (r"$w_2$", (0.35, -0.02)),
    }
    for node, (label, offset) in label_specs.items():
        x, y = positions[node]
        ax.text(
            x + offset[0], y + offset[1], label,
            ha="center", va="center", fontsize=15,
            color="#1e1f20", zorder=8,
        )

    ax.annotate(
        r"At $v$, choose $u$: the neighbour of $v$ closest to $t$.",
        xy=positions["u"], xycoords="data",
        xytext=(5.42, 1.80), textcoords="data",
        ha="center", va="center", fontsize=11.5, color="#243d4d",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor="#8296a3", linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color="#55798f", linewidth=1.0,
                        shrinkA=5, shrinkB=10, connectionstyle="arc3,rad=-0.12"),
        zorder=9,
    )
    ax.text(
        4.58, 4.50, r"local neighbourhood $N(v)$",
        ha="center", va="center", fontsize=10.8, color="#555b5f",
    )
    ax.annotate(
        "greedy route",
        xy=(2.55, 3.12), xytext=(1.80, 2.25),
        ha="center", va="center", fontsize=11.5, color="#2f6f9f",
        arrowprops=dict(arrowstyle="->", color="#2f6f9f", linewidth=1.0),
    )

    ax.set_xlim(0.0, 9.65)
    ax.set_ylim(0.15, 5.65)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)

    # Store an opaque RGB PNG so viewers consistently show a white background.
    with Image.open(OUTPUT_PNG) as image:
        image.convert("RGB").save(OUTPUT_PNG, dpi=(600, 600))


def main():
    positions = build_positions()
    graph = build_graph()
    decisions, shortest_hops = validate_greedy_route(graph, positions)
    draw_figure(graph, positions)

    for current, selected, distance, degree in decisions:
        print(
            f"Greedy step {current} -> {selected}: "
            f"minimum distance to t is {distance:.3f} among {degree} neighbours"
        )
    print(f"Greedy route: {len(ROUTE_EDGES)} hops; graph shortest path: {shortest_hops} hops")
    print(f"Saved high-resolution figure to: {OUTPUT_PNG}")
    print(f"Saved vector figure to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
