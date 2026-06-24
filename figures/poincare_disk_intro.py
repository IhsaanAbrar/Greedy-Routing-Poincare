from pathlib import Path
import shutil

import matplotlib
from matplotlib.transforms import Bbox

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PNG = PROJECT_ROOT / "figures" / "poincare_disk_intro.png"
OUTPUT_PDF = PROJECT_ROOT / "figures" / "poincare_disk_intro.pdf"
LABEL_SIZE = 14


def padded_bbox(bbox, pad):
    return Bbox.from_extents(
        bbox.x0 - pad,
        bbox.y0 - pad,
        bbox.x1 + pad,
        bbox.y1 + pad,
    )


def points_inside_bbox(points, bbox):
    return (
        (points[:, 0] >= bbox.x0)
        & (points[:, 0] <= bbox.x1)
        & (points[:, 1] >= bbox.y0)
        & (points[:, 1] <= bbox.y1)
    )


def segment_crosses_bbox(start, end, bbox):
    segment = np.linspace(start, end, 60)
    return np.any(points_inside_bbox(segment, bbox))


def boundary_point(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def geodesic_arc(start_angle, end_angle, samples=1000):
    """Return the Poincare geodesic arc with ideal endpoints on the unit circle."""
    p = boundary_point(start_angle)
    q = boundary_point(end_angle)

    # A circle orthogonal to the unit circle and passing through p, q has centre c
    # satisfying c . p = c . q = 1.
    center = np.linalg.solve(np.vstack([p, q]), np.ones(2))
    radius = np.sqrt(np.dot(center, center) - 1.0)

    start = np.arctan2(p[1] - center[1], p[0] - center[0])
    end = np.arctan2(q[1] - center[1], q[0] - center[0])

    def sample_between(a, b, first, last):
        if b < a:
            b += 2 * np.pi
        theta = np.linspace(a, b, samples)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        x[0], y[0] = first
        x[-1], y[-1] = last
        return x, y

    candidate_a = sample_between(start, end, p, q)
    candidate_b = sample_between(end, start, q, p)

    mean_norm_a = np.mean(candidate_a[0] ** 2 + candidate_a[1] ** 2)
    mean_norm_b = np.mean(candidate_b[0] ** 2 + candidate_b[1] ** 2)
    return candidate_a if mean_norm_a < mean_norm_b else candidate_b


def draw_poincare_disk():
    OUTPUT_PNG.parent.mkdir(exist_ok=True)
    latex_available = shutil.which("latex") is not None

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": LABEL_SIZE,
            "mathtext.fontset": "cm",
            "text.usetex": latex_available,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor="white")
    ax.set_facecolor("white")

    boundary = plt.Circle(
        (0, 0),
        1,
        fill=False,
        color="#222222",
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(boundary)

    ax.plot([-1, 1], [0, 0], color="#d7d7d7", linewidth=0.75, zorder=0)
    ax.plot([0, 0], [-1, 1], color="#d7d7d7", linewidth=0.75, zorder=0)

    geodesics = [
        (np.deg2rad(22), np.deg2rad(146)),
        (np.deg2rad(40), np.deg2rad(312)),
        (np.deg2rad(206), np.deg2rad(334)),
        (np.deg2rad(168), np.deg2rad(286)),
    ]

    representative_point = None
    geodesic_samples = []
    for index, (start, end) in enumerate(geodesics):
        x, y = geodesic_arc(start, end)
        geodesic_samples.append(np.column_stack([x, y]))
        line = ax.plot(
            x,
            y,
            color="#9f9f9f",
            linewidth=0.9,
            alpha=0.95,
            solid_capstyle="round",
            antialiased=True,
            zorder=1,
        )[0]
        if index == 0:
            representative_point = (x[int(len(x) * 0.20)], y[int(len(y) * 0.20)])

    example_points = np.array(
        [
            [-0.56, 0.34],
            [0.38, 0.54],
            [-0.36, -0.54],
            [0.58, -0.42],
        ]
    )
    ax.scatter(
        example_points[:, 0],
        example_points[:, 1],
        s=30,
        color="#3b78a0",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )

    ax.scatter([0], [0], s=46, color="#111111", zorder=6)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.20, 1.20)
    ax.set_ylim(-1.20, 1.20)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    boundary_samples = np.column_stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, 720)),
            np.sin(np.linspace(0, 2 * np.pi, 720)),
        ]
    )
    axis_samples = np.vstack(
        [
            np.column_stack([np.linspace(-1, 1, 400), np.zeros(400)]),
            np.column_stack([np.zeros(400), np.linspace(-1, 1, 400)]),
        ]
    )
    protected_points = np.vstack(
        geodesic_samples
        + [
            boundary_samples,
            axis_samples,
            example_points,
            np.array([[0.0, 0.0]]),
        ]
    )

    def add_clear_annotation(text, xy, candidates, arrowprops, color="#111111"):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        display_points = ax.transData.transform(protected_points)
        axes_bbox = padded_bbox(ax.get_window_extent(renderer), -28)
        placed_bboxes = getattr(add_clear_annotation, "placed_bboxes", [])

        chosen = None
        for candidate in candidates:
            xytext, ha, va = candidate
            probe = ax.text(
                *xytext,
                text,
                ha=ha,
                va=va,
                fontsize=LABEL_SIZE,
                color=color,
                alpha=0,
            )
            fig.canvas.draw()
            bbox = probe.get_window_extent(renderer)
            probe.remove()

            label_bbox = padded_bbox(bbox, 8)
            inside_axes = axes_bbox.contains(label_bbox.x0, label_bbox.y0) and axes_bbox.contains(
                label_bbox.x1, label_bbox.y1
            )
            clear_of_art = not np.any(points_inside_bbox(display_points, label_bbox))
            clear_of_labels = all(not label_bbox.overlaps(other) for other in placed_bboxes)
            arrow_start = ax.transData.transform(xy)
            arrow_end = ax.transData.transform(xytext)
            arrow_clear = all(
                not segment_crosses_bbox(arrow_start, arrow_end, padded_bbox(other, 4))
                for other in placed_bboxes
            )

            if inside_axes and clear_of_art and clear_of_labels and arrow_clear:
                chosen = candidate
                placed_bboxes.append(label_bbox)
                break

        if chosen is None:
            chosen = candidates[0]

        xytext, ha, va = chosen
        annotation = ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            ha=ha,
            va=va,
            fontsize=LABEL_SIZE,
            arrowprops=arrowprops,
            color=color,
            annotation_clip=False,
            clip_on=False,
            zorder=10,
        )
        add_clear_annotation.placed_bboxes = placed_bboxes
        return annotation

    add_clear_annotation.placed_bboxes = []
    add_clear_annotation(
        "Boundary of the unit disk",
        (np.cos(np.deg2rad(62)), np.sin(np.deg2rad(62))),
        [
            ((0.46, 1.08), "center", "center"),
            ((0.36, 1.06), "center", "center"),
            ((0.18, 1.07), "left", "center"),
            ((-0.04, 1.08), "left", "center"),
        ],
        dict(arrowstyle="->", color="#444444", linewidth=0.85),
    )
    add_clear_annotation(
        "Centre",
        (0, 0),
        [
            ((-0.30, -0.20), "right", "center"),
            ((-0.24, -0.26), "right", "center"),
            ((-0.38, -0.18), "right", "center"),
            ((-0.22, -0.32), "right", "center"),
        ],
        dict(arrowstyle="-", color="#444444", linewidth=0.8),
    )

    fig.tight_layout(pad=1.0)
    fig.savefig(
        OUTPUT_PNG,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.25,
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        pad_inches=0.25,
        facecolor="white",
    )
    plt.close(fig)

    return OUTPUT_PNG, OUTPUT_PDF


def main():
    png_path, pdf_path = draw_poincare_disk()
    print(f"Saved Poincare disk introduction figure to: {png_path}")
    print(f"Saved vector version to: {pdf_path}")


if __name__ == "__main__":
    main()
