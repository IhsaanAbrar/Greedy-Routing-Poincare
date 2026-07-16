"""Generate a schematic of the local greedy-routing decision rule."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
from PIL import Image


FIGURES_DIR = Path(__file__).resolve().parent
OUTPUT_PNG = FIGURES_DIR / "greedy_routing_rule.png"
OUTPUT_PDF = FIGURES_DIR / "greedy_routing_rule.pdf"

CURRENT = "v"
TARGET = "t"
NEIGHBOURS = ["w_1", "w_2", "u", "w_3"]
POSITIONS = {
    CURRENT: (0.8, 2.5),
    "w_1": (3.1, 4.15),
    "w_2": (3.35, 3.05),
    "u": (3.55, 1.85),
    "w_3": (3.0, 0.65),
    TARGET: (7.15, 2.05),
}
DISTANCES = {"w_1": 4.1, "w_2": 3.0, "u": 1.8, "w_3": 3.4}


def midpoint(a, b, fraction=0.55):
    """Return a point at ``fraction`` of the segment from a to b."""
    return (
        a[0] + fraction * (b[0] - a[0]),
        a[1] + fraction * (b[1] - a[1]),
    )


def draw_figure():
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 13,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    graph = nx.Graph()
    graph.add_edges_from((CURRENT, neighbour) for neighbour in NEIGHBOURS)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor="white")
    ax.set_facecolor("white")

    # Local graph edges incident to the current vertex.
    nx.draw_networkx_edges(
        graph,
        POSITIONS,
        edgelist=[(CURRENT, neighbour) for neighbour in NEIGHBOURS],
        edge_color="#777777",
        width=1.35,
        ax=ax,
    )

    # Geometric distances used for the local decision (not graph edges).
    for neighbour in NEIGHBOURS:
        start, end = POSITIONS[neighbour], POSITIONS[TARGET]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#9a9a9a" if neighbour != "u" else "#4f6f88",
            linewidth=1.05 if neighbour != "u" else 1.45,
            linestyle=(0, (3, 3)),
            zorder=1,
        )
        label_xy = midpoint(start, end, 0.58)
        y_offset = {"w_1": 0.14, "w_2": 0.12, "u": -0.15, "w_3": -0.13}[neighbour]
        ax.text(
            label_xy[0],
            label_xy[1] + y_offset,
            rf"$d({neighbour},t)={DISTANCES[neighbour]:.1f}$",
            ha="center",
            va="center",
            fontsize=11.5,
            color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.7, alpha=0.92),
            zorder=4,
        )

    node_colours = ["#d9d9d9" if n != "u" else "#b7cbd8" for n in NEIGHBOURS]
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        nodelist=NEIGHBOURS,
        node_color=node_colours,
        edgecolors="#333333",
        linewidths=1.1,
        node_size=720,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        nodelist=[CURRENT],
        node_color="#ececec",
        edgecolors="#222222",
        linewidths=1.3,
        node_size=800,
        ax=ax,
    )
    ax.scatter(
        *POSITIONS[TARGET],
        s=820,
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.5,
        zorder=3,
    )

    labels = {CURRENT: r"$v$", TARGET: r"$t$", "u": r"$u$",
              "w_1": r"$w_1$", "w_2": r"$w_2$", "w_3": r"$w_3$"}
    for node, label in labels.items():
        ax.text(*POSITIONS[node], label, ha="center", va="center", fontsize=17,
                fontweight="bold" if node in {CURRENT, TARGET, "u"} else "normal", zorder=5)

    # Overlay the selected routing step as a directed, high-contrast arrow.
    selected_arrow = FancyArrowPatch(
        POSITIONS[CURRENT],
        POSITIONS["u"],
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=3.0,
        color="#2f6688",
        shrinkA=24,
        shrinkB=25,
        zorder=6,
    )
    ax.add_patch(selected_arrow)
    ax.text(2.0, 1.83, "selected step", color="#2f6688", fontsize=11.5,
            ha="center", va="bottom", rotation=-13)

    # Identify the candidate set without crowding the node labels.
    ax.text(
        2.85,
        4.70,
        r"candidate neighbours $N(v)$",
        ha="center",
        va="center",
        fontsize=13,
        color="#333333",
    )

    ax.text(
        4.25,
        -0.50,
        r"$u=\underset{w\in N(v)}{\arg\min}\; d(w,t)$"
        "\n" r"$d(u,t)=1.8$ is the smallest candidate distance",
        ha="center",
        va="center",
        fontsize=14,
        color="#1f1f1f",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5",
                  edgecolor="#777777", linewidth=0.9),
        zorder=7,
    )

    ax.set_xlim(0.1, 7.9)
    ax.set_ylim(-1.05, 5.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)

    # Store an opaque RGB PNG so viewers do not substitute a dark background.
    with Image.open(OUTPUT_PNG) as image:
        image.convert("RGB").save(OUTPUT_PNG, dpi=(600, 600))
    return OUTPUT_PNG, OUTPUT_PDF


def main():
    png_path, pdf_path = draw_figure()
    print(f"Saved high-resolution figure to: {png_path}")
    print(f"Saved vector figure to: {pdf_path}")


if __name__ == "__main__":
    main()
