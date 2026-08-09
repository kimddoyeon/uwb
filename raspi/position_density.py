#!/usr/bin/env python3
"""Render live 2-D position densities from GAT_DIST/RULE_DIST lines.

Usage:
  python3 gat_distance_filter.py --port /dev/ttyUSB0 --baud 115200 \
    | python3 position_density.py
"""

import argparse
import math
import queue
import re
import sys
import threading
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects


PEOPLE = ("C", "D", "E", "F")
COLORS = {
    "C": (0.90, 0.05, 0.05),  # red
    "D": (1.00, 0.78, 0.00),  # yellow
    "E": (0.00, 0.65, 0.12),  # green
    "F": (0.05, 0.25, 1.00),  # blue
}


def person_colormap(person):
    """White means zero density; the person's colour means maximum density."""
    return LinearSegmentedColormap.from_list(
        f"white_to_{person}", [(1.0, 1.0, 1.0), COLORS[person]]
    )


def combined_rgb(densities):
    """Blend four coloured density layers over a white background."""
    shape = next(iter(densities.values())).shape
    image = np.ones((*shape, 3), dtype=float)
    for person in PEOPLE:
        alpha = np.clip(densities[person] * 0.82, 0.0, 0.82)[..., None]
        colour = np.asarray(COLORS[person], dtype=float)
        image = image * (1.0 - alpha) + colour * alpha
    return np.clip(image, 0.0, 1.0)


def add_person_labels(axis, person, density, x_edges, y_edges):
    """Label the strongest solution in each mirrored half-plane."""
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    colour = COLORS[person]
    labelled = []
    for positive_side in (True, False):
        rows = np.where(y_centres >= 0)[0] if positive_side else np.where(y_centres < 0)[0]
        if rows.size == 0 or density[rows, :].max() <= 0:
            continue
        local_row, col = np.unravel_index(np.argmax(density[rows, :]), density[rows, :].shape)
        row = rows[local_row]
        position = (x_centres[col], y_centres[row])
        if any(math.hypot(position[0] - old[0], position[1] - old[1]) < 0.25 for old in labelled):
            continue
        labelled.append(position)
        label = axis.text(
            position[0], position[1], f" {person} ", color=colour,
            fontsize=12, weight="bold", ha="center", va="center", zorder=10
        )
        label.set_path_effects([
            path_effects.Stroke(linewidth=3.0, foreground="white"),
            path_effects.Normal(),
        ])


def parse_line(line):
    line = line.strip()
    if not (line.startswith("RULE_DIST,") or line.startswith("GAT_DIST,")):
        return None

    fields = {}
    for token in line.split(",")[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value

    distances = {}
    for label in ("AB", "AC", "BC", "AD", "BD", "AE", "BE", "AF", "BF"):
        try:
            value = float(fields[label])
            if math.isfinite(value) and value > 0:
                distances[label] = value
        except (KeyError, ValueError):
            pass

    probabilities = {}
    for item in fields.get("prob", "").split("|"):
        match = re.fullmatch(r"([A-F]{2}):([0-9]*\.?[0-9]+)", item)
        if match:
            probabilities[match.group(1)] = float(match.group(2))

    return distances, probabilities


def triangle_candidates(ab, ap, bp):
    """Return the two mirror positions P=(x,+/-y), or None if invalid."""
    if ab <= 0:
        return None
    x = (ap * ap - bp * bp + ab * ab) / (2.0 * ab)
    y_squared = ap * ap - x * x
    # Small negative values can result from measurement noise.
    tolerance = max(0.05, 0.02 * ap * ap)
    if y_squared < -tolerance:
        return None
    y = math.sqrt(max(0.0, y_squared))
    return (x, y), (x, -y)


def gaussian_kernel(sigma_bins):
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return kernel / kernel.sum()


def smooth_density(points, x_edges, y_edges, bandwidth):
    if not points:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    xy = np.asarray([(x, y) for x, y, _ in points], dtype=float)
    weights = np.asarray([weight for _, _, weight in points], dtype=float)
    density, _, _ = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=(y_edges, x_edges), weights=weights
    )
    bin_size = max(x_edges[1] - x_edges[0], 1e-6)
    kernel = gaussian_kernel(max(0.6, bandwidth / bin_size))
    density = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, density)
    density = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, density)
    peak = density.max()
    return density / peak if peak > 0 else density


def stdin_reader(input_queue):
    for line in sys.stdin:
        input_queue.put(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=1000, help="samples retained per person")
    parser.add_argument("--grid", type=int, default=180, help="density grid resolution")
    parser.add_argument("--bandwidth", type=float, default=0.18, help="density smoothing in metres")
    parser.add_argument("--refresh-ms", type=int, default=500)
    parser.add_argument("--output-dir", default="density_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = {person: deque(maxlen=args.history * 2) for person in PEOPLE}
    latest_ab = None
    line_queue = queue.Queue()
    threading.Thread(target=stdin_reader, args=(line_queue,), daemon=True).start()

    fig, axes_array = plt.subplots(2, 3, figsize=(15, 9))
    axes = list(axes_array.flat)
    axes[5].axis("off")
    fig.canvas.manager.set_window_title("UWB position probability density")

    def update(_frame):
        nonlocal latest_ab
        accepted = 0
        while accepted < 500:
            try:
                line = line_queue.get_nowait()
            except queue.Empty:
                break
            parsed = parse_line(line)
            if parsed is None:
                continue
            distances, probabilities = parsed
            ab = distances.get("AB")
            if ab is None:
                continue
            latest_ab = ab
            for person in PEOPLE:
                ap_label = "A" + person
                bp_label = "B" + person
                ap = distances.get(ap_label)
                bp = distances.get(bp_label)
                if ap is None or bp is None:
                    continue
                candidates = triangle_candidates(ab, ap, bp)
                if candidates is None:
                    print(f"invalid triangle skipped: AB={ab:.3f}, {ap_label}={ap:.3f}, {bp_label}={bp:.3f}")
                    continue
                # Both required links must be probable; geometric mean keeps the scale in [0, 1].
                weight = math.sqrt(
                    probabilities.get(ap_label, 1.0) * probabilities.get(bp_label, 1.0)
                ) / 2.0
                for x, y in candidates:
                    samples[person].append((x, y, weight))
            accepted += 1

        # Avoid redrawing and rewriting five PNG files when no new filtered row arrived.
        if accepted == 0:
            return []

        all_points = [point for person in PEOPLE for point in samples[person]]
        if not all_points or latest_ab is None:
            return []

        xs = [0.0, latest_ab] + [point[0] for point in all_points]
        ys = [0.0] + [point[1] for point in all_points]
        margin = max(0.75, 0.10 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
        x_min, x_max = min(xs) - margin, max(xs) + margin
        y_min, y_max = min(ys) - margin, max(ys) + margin
        x_edges = np.linspace(x_min, x_max, args.grid + 1)
        y_edges = np.linspace(y_min, y_max, args.grid + 1)
        extent = (x_min, x_max, y_min, y_max)

        person_points = {person: list(samples[person]) for person in PEOPLE}
        densities = {
            person: smooth_density(person_points[person], x_edges, y_edges, args.bandwidth)
            for person in PEOPLE
        }
        combined_image = combined_rgb(densities)

        plot_names = list(PEOPLE) + ["ALL"]
        for axis, name in zip(axes[:5], plot_names):
            axis.clear()
            axis.set_facecolor("white")
            if name == "ALL":
                axis.imshow(
                    combined_image, origin="lower", extent=extent,
                    aspect="equal", interpolation="bilinear"
                )
                for person in PEOPLE:
                    add_person_labels(axis, person, densities[person], x_edges, y_edges)
                axis.legend(
                    handles=[
                        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
                               markerfacecolor=COLORS[person], markeredgecolor="none", label=person)
                        for person in PEOPLE
                    ], title="Person", loc="upper right", framealpha=0.95
                )
                count = sum(len(person_points[person]) // 2 for person in PEOPLE)
            else:
                density = densities[name]
                axis.imshow(
                    density, origin="lower", extent=extent, cmap=person_colormap(name),
                    vmin=0.0, vmax=1.0, aspect="equal", interpolation="bilinear"
                )
                add_person_labels(axis, name, density, x_edges, y_edges)
                count = len(person_points[name]) // 2
            axis.scatter([0.0], [0.0], marker="^", s=90, c="black", label="A")
            axis.scatter([latest_ab], [0.0], marker="^", s=90, c="white", edgecolors="black", label="B")
            axis.text(0.0, 0.0, " A", weight="bold")
            axis.text(latest_ab, 0.0, " B", weight="bold")
            axis.set_title(f"{name} position density (N={count})")
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_min, y_max)
            axis.grid(alpha=0.2)

            save_name = f"density_{name.lower() if name == 'ALL' else name}.png"
            single_fig, single_ax = plt.subplots(figsize=(7, 6))
            single_ax.set_facecolor("white")
            if name == "ALL":
                single_ax.imshow(
                    combined_image, origin="lower", extent=extent,
                    aspect="equal", interpolation="bilinear"
                )
                for person in PEOPLE:
                    add_person_labels(single_ax, person, densities[person], x_edges, y_edges)
                single_ax.legend(
                    handles=[
                        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
                               markerfacecolor=COLORS[person], markeredgecolor="none", label=person)
                        for person in PEOPLE
                    ], title="Person", loc="upper right", framealpha=0.95
                )
            else:
                density = densities[name]
                single_image = single_ax.imshow(
                    density, origin="lower", extent=extent, cmap=person_colormap(name),
                    vmin=0.0, vmax=1.0, aspect="equal", interpolation="bilinear"
                )
                add_person_labels(single_ax, name, density, x_edges, y_edges)
                single_fig.colorbar(
                    single_image, ax=single_ax, label="relative probability density"
                )
            single_ax.scatter([0.0], [0.0], marker="^", s=90, c="black")
            single_ax.scatter([latest_ab], [0.0], marker="^", s=90, c="white", edgecolors="black")
            single_ax.text(0.0, 0.0, " A", weight="bold")
            single_ax.text(latest_ab, 0.0, " B", weight="bold")
            single_ax.set(title=f"{name} position probability density", xlabel="x (m)", ylabel="y (m)")
            single_ax.grid(alpha=0.2)
            single_fig.tight_layout()
            single_fig.savefig(output_dir / save_name, dpi=160)
            plt.close(single_fig)

        fig.suptitle(f"A=(0, 0), B=({latest_ab:.3f}, 0) — mirrored triangle solutions")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return []

    animation = FuncAnimation(
        fig, update, interval=args.refresh_ms, cache_frame_data=False
    )
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Five density images are in: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
