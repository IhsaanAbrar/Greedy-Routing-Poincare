from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_PATH = PROJECT_ROOT / "figures" / "introductory_graph.png"

VERTICES = list(range(10))

EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (2, 4),
    (3, 5),
    (4, 5),
    (5, 6),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 9),
]


def build_graph():
    graph = nx.Graph()
    graph.add_nodes_from(VERTICES)
    graph.add_edges_from(EDGES)
    return graph


def print_graph_summary(graph):
    shortest_path = nx.shortest_path(graph, source=0, target=9)
    shortest_path_length = nx.shortest_path_length(graph, source=0, target=9)

    print(f"Number of vertices: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Graph is connected: {nx.is_connected(graph)}")
    print(f"Shortest path from 0 to 9: {shortest_path}")
    print(f"Shortest path length from 0 to 9: {shortest_path_length}")


def draw_graph(graph):
    FIGURE_PATH.parent.mkdir(exist_ok=True)

    layout = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(8, 6), facecolor="white")
    nx.draw_networkx_nodes(graph, layout, node_color="#8ecae6", node_size=900)
    nx.draw_networkx_edges(graph, layout, edge_color="#555555", width=1.8)
    nx.draw_networkx_labels(graph, layout, font_size=12, font_weight="bold")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300, facecolor="white")
    plt.close()

    with Image.open(FIGURE_PATH) as image:
        image.convert("RGB").save(FIGURE_PATH)

    return FIGURE_PATH


def main():
    graph = build_graph()
    print_graph_summary(graph)
    figure_path = draw_graph(graph)
    print(f"Saved graph figure to: {figure_path}")


if __name__ == "__main__":
    main()
