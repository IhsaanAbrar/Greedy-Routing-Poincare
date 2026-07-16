"""Generate a two-panel, metric-agnostic greedy-routing illustration."""

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

ROUTE = ["s", "r1", "r2", "v", "u", "r5", "t"]
ROUTE_EDGES = list(zip(ROUTE[:-1], ROUTE[1:]))

# Values of the chosen metric d(., t).  They need not equal drawn Euclidean
# lengths: the construction is deliberately independent of a particular metric.
METRIC_TO_TARGET = {
    "s": 10.0,
    "a": 9.2,
    "b": 9.7,
    "r1": 8.0,
    "c": 7.4,
    "r2": 6.5,
    "w1": 5.8,
    "w2": 6.1,
    "v": 5.0,
    "g": 4.2,
    "u": 3.4,
    "h": 3.1,
    "i": 2.5,
    "r5": 1.8,
    "t": 0.0,
}

BASE_POSITIONS = {
    "s": (0.40, 2.55),
    "r1": (1.42, 3.42),
    "r2": (2.50, 2.78),
    "v": (3.55, 3.42),
    "u": (4.52, 2.76),
    "r5": (5.47, 3.26),
    "t": (6.45, 2.58),
    "a": (0.98, 0.92),
    "b": (1.05, 4.62),
    "c": (2.35, 4.63),
    "w1": (3.58, 4.83),
    "w2": (3.20, 1.75),
    "g": (4.73, 4.47),
    "h": (4.78, 0.88),
    "i": (5.78, 4.38),
}

EXTRA_EDGES = [
    ("s", "a"), ("s", "b"),
    ("r1", "b"), ("r1", "c"),
    ("r2", "c"), ("r2", "w2"),
    ("v", "w1"), ("v", "w2"),
    ("u", "g"), ("u", "h"),
    ("r5", "i"), ("r5", "h"),
    ("t", "i"), ("t", "h"),
    ("a", "w2"), ("b", "c"),
    ("c", "w1"), ("w1", "g"),
    ("g", "i"), ("w2", "h"),
]

LOCAL_NEIGHBOURS = ["r2", "w1", "u", "w2"]
LOCAL_LABELS = {"r2": r"$w_1$", "w1": r"$w_2$", "u": r"$u$", "w2": r"$w_3$"}

GREY_EDGE = "#c5c9cc"
GREY_NODE = "#e5e7e8"
BLUE = "#2f6f9f"
GREEN = "#79a986"
RED = "#c96b67"
V_COLOUR = "#9a8db1"
U_COLOUR = "#79a8a3"


def build_positions():
    """Return deterministic coordinates with subtle seeded jitter."""
    rng = np.random.default_rng(SEED)
    positions = {}
    for node, point in BASE_POSITIONS.items():
        if node in ROUTE:
            positions[node] = np.asarray(point, dtype=float)
        else:
            positions[node] = np.asarray(point, dtype=float) + rng.uniform(-0.045, 0.045, 2)
    return positions


def build_graph():
    graph = nx.Graph()
    graph.add_nodes_from(BASE_POSITIONS)
    graph.add_edges_from(ROUTE_EDGES)
    graph.add_edges_from(EXTRA_EDGES)
    return graph


def validate_greedy_route(graph):
    """Validate strict local minimisation, termination, and absence of cycles."""
    if ROUTE[-1] != "t":
        raise ValueError("The greedy route does not terminate at t")
    if len(ROUTE) != len(set(ROUTE)):
        raise ValueError("The greedy route contains a cycle")

    decisions = []
    for current, selected in ROUTE_EDGES:
        neighbours = list(graph.neighbors(current))
        values = [METRIC_TO_TARGET[node] for node in neighbours]
        if len(values) != len(set(values)):
            raise ValueError(f"Metric tie among neighbours of {current}")
        best = min(neighbours, key=METRIC_TO_TARGET.__getitem__)
        if best != selected:
            raise ValueError(
                f"Invalid greedy step at {current}: {best}, not {selected}, minimises d(w,t)"
            )
        decisions.append((current, selected, METRIC_TO_TARGET[selected], len(neighbours)))
    return decisions


def validate_local_decision(graph):
    """Assert that u is the unique metric-minimising neighbour of v."""
    neighbours = set(graph.neighbors("v"))
    if neighbours != set(LOCAL_NEIGHBOURS):
        raise ValueError(
            f"Panel (b) neighbours {set(LOCAL_NEIGHBOURS)} do not match N(v)={neighbours}"
        )
    values = {node: METRIC_TO_TARGET[node] for node in neighbours}
    if len(values.values()) != len(set(values.values())):
        raise ValueError("The neighbours of v do not have strict metric values")
    selected = min(values, key=values.__getitem__)
    if selected != "u":
        raise ValueError(f"u is not the unique closest neighbour of v; selected {selected}")
    return values


def build_local_positions(positions):
    """Enlarge N(v) using the same relative coordinates as panel (a)."""
    local_origin = np.array([-0.75, 0.18])
    scale = 1.02
    centre = positions["v"]
    local = {"v": local_origin}
    for node in LOCAL_NEIGHBOURS:
        local[node] = local_origin + scale * (positions[node] - centre)
    local["t"] = np.array([3.48, 0.18])
    return local


def draw_arrow(ax, start, end, *, width=2.9, scale=13.5, zorder=5):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=scale,
        linewidth=width, color=BLUE, shrinkA=10, shrinkB=11, zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_panel_a(ax, graph, positions):
    nx.draw_networkx_edges(
        graph, positions, edge_color=GREY_EDGE, width=0.85, alpha=0.95, ax=ax,
    )
    for start, end in ROUTE_EDGES:
        width = 3.6 if (start, end) == ("v", "u") else 2.9
        draw_arrow(ax, positions[start], positions[end], width=width)

    ordinary = [node for node in graph if node not in ROUTE]
    route_nodes = [node for node in ROUTE if node not in {"s", "v", "u", "t"}]
    nx.draw_networkx_nodes(
        graph, positions, nodelist=ordinary, node_size=125,
        node_color=GREY_NODE, edgecolors="#777d81", linewidths=0.7, ax=ax,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=route_nodes, node_size=150,
        node_color="#d6e4ed", edgecolors="#315f7d", linewidths=0.9, ax=ax,
    )
    for node, colour, edge, size in [
        ("s", GREEN, "#315c3c", 240),
        ("v", V_COLOUR, "#574b6a", 220),
        ("u", U_COLOUR, "#3f6965", 205),
        ("t", RED, "#7c3532", 250),
    ]:
        nx.draw_networkx_nodes(
            graph, positions, nodelist=[node], node_size=size,
            node_color=colour, edgecolors=edge, linewidths=1.1, ax=ax,
        )

    label_offsets = {
        "s": (-0.03, 0.33),
        "v": (-0.02, 0.33),
        "u": (0.02, -0.32),
        "t": (0.02, 0.34),
    }
    for node, offset in label_offsets.items():
        x, y = positions[node]
        ax.text(x + offset[0], y + offset[1], rf"${node}$",
                ha="center", va="center", fontsize=14, zorder=8)

    ax.text(0.02, 0.97, r"(a)", transform=ax.transAxes,
            ha="left", va="top", fontsize=13)
    ax.set_xlim(0.0, 6.78)
    ax.set_ylim(0.35, 5.20)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def draw_local_node(ax, local_positions, node, colour, edgecolour, size=230):
    x, y = local_positions[node]
    ax.scatter(x, y, s=size, facecolor=colour, edgecolor=edgecolour,
               linewidth=1.1, zorder=5)


def draw_panel_b(ax, positions):
    local_positions = build_local_positions(positions)
    v_position = local_positions["v"]
    t_position = local_positions["t"]

    for neighbour in LOCAL_NEIGHBOURS:
        point = local_positions[neighbour]
        ax.plot(
            [v_position[0], point[0]], [v_position[1], point[1]],
            color="#8c9296", linewidth=1.15, zorder=1,
        )
        ax.plot(
            [point[0], t_position[0]], [point[1], t_position[1]],
            color="#d0d4d6" if neighbour != "u" else "#abc0be",
            linewidth=0.75 if neighbour != "u" else 0.9,
            linestyle=(0, (3, 3)), zorder=1,
        )

    draw_arrow(ax, local_positions["r2"], v_position, width=2.9, scale=13.5, zorder=4)
    draw_arrow(ax, v_position, local_positions["u"], width=3.6, scale=14.5, zorder=4)

    draw_local_node(ax, local_positions, "v", V_COLOUR, "#574b6a", size=270)
    draw_local_node(ax, local_positions, "r2", GREY_NODE, "#666c70")
    draw_local_node(ax, local_positions, "w1", GREY_NODE, "#666c70")
    draw_local_node(ax, local_positions, "w2", GREY_NODE, "#666c70")
    draw_local_node(ax, local_positions, "u", U_COLOUR, "#3f6965", size=250)
    draw_local_node(ax, local_positions, "t", RED, "#7c3532", size=275)

    ax.text(v_position[0] - 0.02, v_position[1] + 0.33, r"$v$",
            ha="center", va="center", fontsize=15, zorder=7)
    ax.text(t_position[0], t_position[1] + 0.34, r"$t$",
            ha="center", va="center", fontsize=15, zorder=7)
    local_label_offsets = {
        "r2": (-0.17, 0.24),
        "w1": (-0.24, 0.01),
        "u": (-0.22, -0.03),
        "w2": (-0.20, -0.13),
    }
    for node, label in LOCAL_LABELS.items():
        point = local_positions[node]
        offset = local_label_offsets[node]
        ax.text(point[0] + offset[0], point[1] + offset[1], label,
                ha="right", va="center", fontsize=13.5, zorder=7)

    distance_labels = {
        "r2": (0.82, 0.42, r"$d(w_1,t)=6.5$"),
        "w1": (1.45, 1.34, r"$d(w_2,t)=5.8$"),
        "u": (2.12, -0.30, r"$d(u,t)=3.4$"),
        "w2": (1.38, -1.10, r"$d(w_3,t)=6.1$"),
    }
    for node, (x, y, text) in distance_labels.items():
        ax.text(
            x, y, text, ha="center", va="center", fontsize=10.5,
            color=BLUE if node == "u" else "#44484b",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.94),
            zorder=6,
        )

    ax.text(
        0.85, -2.12,
        r"$u=\underset{w\in N(v)}{\arg\min}\; d(w,t)$",
        ha="center", va="center", fontsize=13.5, color="#202224",
    )
    ax.text(0.85, -2.45, r"$d$ denotes the chosen metric",
            ha="center", va="center", fontsize=9.5, color="#666a6d")
    ax.text(0.02, 0.97, r"(b)", transform=ax.transAxes,
            ha="left", va="top", fontsize=13)
    ax.set_xlim(-2.05, 3.75)
    ax.set_ylim(-2.68, 2.02)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def draw_figure(graph, positions):
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 11,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_route, ax_local) = plt.subplots(
        1, 2, figsize=(11.2, 5.1), facecolor="white",
        gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.035},
    )
    ax_route.set_facecolor("white")
    ax_local.set_facecolor("white")
    draw_panel_a(ax_route, graph, positions)
    draw_panel_b(ax_local, positions)

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)

    with Image.open(OUTPUT_PNG) as image:
        image.convert("RGB").save(OUTPUT_PNG, dpi=(600, 600))


def main():
    positions = build_positions()
    graph = build_graph()
    decisions = validate_greedy_route(graph)
    local_values = validate_local_decision(graph)
    draw_figure(graph, positions)

    for current, selected, value, degree in decisions:
        print(
            f"Greedy step {current} -> {selected}: "
            f"strict minimum d(w,t)={value:.1f} among {degree} neighbours"
        )
    print(f"Route reaches t in {len(ROUTE_EDGES)} steps without cycling")
    print(
        "Local decision at v verified: "
        + ", ".join(f"d({node},t)={value:.1f}" for node, value in sorted(local_values.items()))
    )
    print(f"Saved high-resolution figure to: {OUTPUT_PNG}")
    print(f"Saved vector figure to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
