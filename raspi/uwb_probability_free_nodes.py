#!/usr/bin/env python3
"""Same pipeline as uwb_probability_pipeline.py, but with A and B free too.

The fixed version pins A at the origin and B at ``(AB, 0)``, so those two nodes
carry no uncertainty by construction and every measurement error is pushed onto
C-F.  Here all six nodes are sampled together, which is what the measurements
actually determine -- and what they do *not* determine.

Range measurements only fix a configuration up to a global rigid motion:
translating, rotating or reflecting every node at once changes no distance.
Those three-and-a-bit degrees of freedom are removed after sampling instead of
before it:

1.  Sample every active node's coordinates from the same posterior the fixed
    version uses, with AB now a real constraint rather than a definition.
2.  Take the lowest-energy sample as the reference and canonicalise it: A at the
    origin, B on the +x axis, chirality chosen so the people sit at y > 0 on
    average.
3.  Procrustes-align every sample to that reference, reflection allowed. The
    frame then matches the fixed version's plots, but A and B land in a cloud
    rather than on a point.

Reflection is allowed in the alignment because it is a gauge freedom here, not
a second solution: unlike the fixed version there is no external axis to mirror
against, so the two chiralities are the same configuration and get collapsed.

Everything else -- parsing, windows, the checkpoint, Top-K + alpha, the sampler,
the density grid -- is imported from uwb_probability_pipeline.py.

Examples
--------
::

    python3 uwb_probability_free_nodes.py \
        --input-log device-monitor-260812-214621.log --checkpoint best_model.pt

    python3 uwb_probability_free_nodes.py --mode last \
        --save-dir density_free --no-show
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from uwb_probability_pipeline import (
    Box,
    ConfigurationEnergy,
    PEOPLE,
    SCRIPT_DIR,
    SamplerConfig,
    Selection,
    WindowBank,
    density_peak,
    describe_checkpoint,
    iter_lines,
    load_checkpoint,
    newest_log,
    parse_uwb_line,
    person_density,
    sample_configurations,
    score_links,
    select_links,
)


# A and B need their own colours here; C-F keep the ones the other plots use.
NODE_COLOR = {
    "A": (0.20, 0.20, 0.20),
    "B": (0.55, 0.15, 0.75),
    "C": (0.90, 0.05, 0.05),
    "D": (1.00, 0.78, 0.00),
    "E": (0.00, 0.65, 0.12),
    "F": (0.05, 0.25, 1.00),
}
PANEL_ORDER = ("A", "B", "C", "D", "E", "F")


# ============================================================
# Gauge fixing
# ============================================================
def canonicalise(configuration: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Put one configuration into the frame the plots use.

    A at the origin, B on the +x axis, and the people on the +y side on
    average.  When A or B is missing the longest node pair stands in for the
    baseline, so the frame is still deterministic.
    """
    config = np.asarray(configuration, dtype=np.float64).copy()
    index = {name: position for position, name in enumerate(names)}

    origin = config[index["A"]] if "A" in index else config.mean(axis=0)
    config -= origin

    if "A" in index and "B" in index:
        direction = config[index["B"]] - config[index["A"]]
    else:
        # Longest pair: the most stable axis the configuration offers.
        deltas = config[:, None, :] - config[None, :, :]
        lengths = np.linalg.norm(deltas, axis=-1)
        i, j = np.unravel_index(int(np.argmax(lengths)), lengths.shape)
        direction = config[j] - config[i]

    if not np.any(direction):
        direction = np.asarray([1.0, 0.0])

    angle = math.atan2(direction[1], direction[0])
    cos, sin = math.cos(angle), math.sin(angle)
    config = config @ np.asarray([[cos, -sin], [sin, cos]])

    people_rows = [index[name] for name in PEOPLE if name in index]
    if people_rows and config[people_rows, 1].sum() < 0.0:
        config[:, 1] *= -1.0

    return config


def procrustes_align(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rigidly align every sample onto ``reference`` (reflection allowed).

    ``samples``: (S, N, 2); ``reference``: (N, 2).
    """
    reference_mean = reference.mean(axis=0)
    reference_centred = reference - reference_mean

    centred = samples - samples.mean(axis=1, keepdims=True)
    cross = np.einsum("sni,nj->sij", centred, reference_centred)
    left, _, right = np.linalg.svd(cross)
    rotation = left @ right

    return centred @ rotation + reference_mean


# ============================================================
# Frame
# ============================================================
@dataclass
class FreeFrame:
    seq: int
    ms: float
    selection: Selection
    nodes: List[str]
    degrees: Dict[str, int]
    densities: Dict[str, np.ndarray]
    peaks: Dict[str, Optional[Tuple[float, float]]]
    spreads: Dict[str, float]
    box: Box
    acceptance: float
    residual_rms: float
    posterior_sigma: float
    sample_count: int
    architecture: str


def view_box(samples: np.ndarray, margin: float,
             override: Dict[str, Optional[float]]) -> Box:
    flat = samples.reshape(-1, 2)
    x_low, x_high = np.percentile(flat[:, 0], [0.5, 99.5])
    y_low, y_high = np.percentile(flat[:, 1], [0.5, 99.5])

    box = Box(
        x_min=float(x_low - margin),
        x_max=float(x_high + margin),
        y_min=float(y_low - margin),
        y_max=float(y_high + margin),
    )
    for name, value in override.items():
        if value is not None:
            setattr(box, name, float(value))
    return box


def build_frame(
    model,
    bank: WindowBank,
    seq: int,
    ms: float,
    runtime,
    device,
    args,
    rng: np.random.Generator,
    previous_state: Optional[np.ndarray],
) -> Tuple[Optional[FreeFrame], Optional[np.ndarray]]:
    links = bank.active_links(seq, args.min_samples, args.max_age_scans)
    if not links:
        return None, previous_state

    scores, distances = score_links(model, bank, links, runtime, device)
    selection = select_links(
        scores,
        distances,
        runtime.k_ratio,
        runtime.confidence_threshold,
        args.min_person_degree,
        not args.no_anchor_links,
    )

    # Every node that a selected link touches becomes a free variable.
    nodes = [
        name
        for name in PANEL_ORDER
        if any(name in label for label in selection.links)
    ]
    if len(nodes) < 3:
        return None, previous_state

    # ConfigurationEnergy treats each name in `nodes` as a free coordinate, so
    # passing all six leaves nothing pinned and AB becomes a real constraint.
    unused = np.zeros(2, dtype=np.float64)
    energy = ConfigurationEnergy(nodes, selection, unused, unused)
    degrees = energy.degree(nodes)

    sampler = SamplerConfig(
        posterior_sigma=args.posterior_sigma,
        chains=args.chains,
        samples=args.samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        proposal_std=args.proposal_std,
        anneal_start=args.anneal,
        jump_prob=args.jump_prob,
        jump_scale=args.jump_scale,
        # Reflection is removed by the alignment below, so adding a mirrored
        # copy here would only duplicate every sample.
        mirror=False,
    )

    # Nothing anchors the configuration, so the prior box only has to be large
    # enough that it never squeezes the shape; the alignment recentres it.
    reach = max(
        [selection.distances[label] for label in selection.links], default=10.0
    )
    span = 1.5 * reach + args.box_margin
    prior = Box(x_min=-span, x_max=span, y_min=-span, y_max=span)

    samples, acceptance = sample_configurations(
        energy, len(nodes), prior, sampler, rng,
        initial_state=(
            previous_state
            if previous_state is not None
            and previous_state.shape == (len(nodes), 2)
            else None
        ),
    )
    if not samples.size:
        return None, previous_state

    previous_state = samples[-1].astype(np.float64)

    reference = canonicalise(samples[int(np.argmin(energy.raw(samples)))], nodes)
    aligned = procrustes_align(samples.astype(np.float64), reference)

    box = view_box(
        aligned,
        args.box_margin,
        {
            "x_min": args.x_min,
            "x_max": args.x_max,
            "y_min": args.y_min,
            "y_max": args.y_max,
        },
    )
    x_edges = np.linspace(box.x_min, box.x_max, args.grid_bins + 1)
    y_edges = np.linspace(box.y_min, box.y_max, args.grid_bins + 1)

    densities: Dict[str, np.ndarray] = {}
    peaks: Dict[str, Optional[Tuple[float, float]]] = {}
    spreads: Dict[str, float] = {}
    for position, name in enumerate(nodes):
        node_samples = aligned[:, position, :]
        density = person_density(node_samples, x_edges, y_edges, args.smoothing)
        densities[name] = density
        peaks[name] = density_peak(density, x_edges, y_edges, upper_half_only=False)
        spreads[name] = float(np.sqrt(node_samples.var(axis=0).sum()))

    frame = FreeFrame(
        seq=seq,
        ms=ms,
        selection=selection,
        nodes=nodes,
        degrees=degrees,
        densities=densities,
        peaks=peaks,
        spreads=spreads,
        box=box,
        acceptance=acceptance,
        residual_rms=energy.residual_rms(samples),
        posterior_sigma=args.posterior_sigma,
        sample_count=int(samples.shape[0]),
        architecture=runtime.architecture,
    )
    return frame, previous_state


# ============================================================
# Rendering
# ============================================================
def node_colormap(name: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        f"white_to_{name}", [(1.0, 1.0, 1.0), NODE_COLOR[name]]
    )


def draw_frame(figure, frame: FreeFrame) -> None:
    from matplotlib.lines import Line2D

    figure.clear()
    axes = figure.subplots(2, 4, squeeze=False)
    flat = list(axes.flat)
    extent = (frame.box.x_min, frame.box.x_max, frame.box.y_min, frame.box.y_max)

    def finish(axis):
        axis.set_xlim(frame.box.x_min, frame.box.x_max)
        axis.set_ylim(frame.box.y_min, frame.box.y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")

    overview = flat[0]
    for name in frame.nodes:
        density = frame.densities[name]
        if density.max() <= 0:
            continue
        overview.contour(
            density,
            levels=[0.15, 0.35, 0.60, 0.82],
            extent=extent,
            origin="lower",
            colors=[NODE_COLOR[name]],
            linewidths=1.3,
        )
    finish(overview)
    overview.set_title("All nodes free -- combined contours")
    overview.legend(
        handles=[
            Line2D([0], [0], color=NODE_COLOR[name], lw=2, label=name)
            for name in frame.nodes
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    for panel, name in enumerate(PANEL_ORDER, start=1):
        axis = flat[panel]
        density = frame.densities.get(name)
        if density is None:
            axis.axis("off")
            continue
        axis.imshow(
            density,
            origin="lower",
            extent=extent,
            cmap=node_colormap(name),
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="bilinear",
        )
        peak = frame.peaks.get(name)
        if peak is not None:
            axis.scatter(
                peak[0], peak[1], marker="x", s=60,
                c=[NODE_COLOR[name]], linewidths=2.0, zorder=8,
            )
        finish(axis)
        axis.set_title(
            f"{name}  (links={frame.degrees.get(name, 0)}, "
            f"spread={frame.spreads.get(name, float('nan')):.2f} m)"
        )

    info = flat[7]
    info.axis("off")
    lines = [
        f"model:     {frame.architecture}",
        f"scan seq:  {frame.seq}   t={frame.ms / 1000.0:.1f} s",
        f"nodes:     {len(frame.nodes)} free ({', '.join(frame.nodes)})",
        f"K:         {frame.selection.k_min}",
        f"alpha:     {frame.selection.alpha}",
        f"forced:    {', '.join(frame.selection.forced) or '-'}",
        f"samples:   {frame.sample_count}  (accept {frame.acceptance:.2f})",
        f"fit RMS:   {frame.residual_rms:.3f} m  (sigma {frame.posterior_sigma:.2f})",
        "",
        "selected links:",
    ]
    for rank, label in enumerate(frame.selection.links, start=1):
        lines.append(
            f"{rank:2d}. {label}  d={frame.selection.distances[label]:6.2f} m"
            f"  s={frame.selection.scores[label]:.3f}"
        )
    lines.append("")
    lines.append("peak / spread per node:")
    for name in frame.nodes:
        peak = frame.peaks.get(name)
        lines.append(
            f"  {name}: ({peak[0]:6.2f},{peak[1]:6.2f}) "
            f"+/-{frame.spreads[name]:.2f} m"
            if peak
            else f"  {name}: -"
        )
    lines.extend(
        [
            "",
            "A and B are sampled like every",
            "other node. Distances fix the",
            "shape only, so each sample is",
            "rigidly aligned to the best-fit",
            "one: A near the origin, B near",
            "the +x axis, both with spread.",
        ]
    )
    info.text(
        0.02, 0.98, "\n".join(lines),
        va="top", ha="left", family="monospace", fontsize=8,
        transform=info.transAxes,
    )

    figure.suptitle(
        f"UWB position probability density, all nodes free | scan {frame.seq}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))


def frame_json(frame: FreeFrame) -> str:
    return json.dumps(
        {
            "type": "free_distribution_update",
            "seq": frame.seq,
            "ms": frame.ms,
            "model": frame.architecture,
            "nodes": frame.nodes,
            "k_min": frame.selection.k_min,
            "alpha": frame.selection.alpha,
            "fit_rms": round(frame.residual_rms, 4),
            "selected": [
                {
                    "link": label,
                    "distance": round(frame.selection.distances[label], 3),
                    "score": round(frame.selection.scores[label], 4),
                }
                for label in frame.selection.links
            ],
            "positions": {
                name: (
                    {
                        "peak": [round(peak[0], 3), round(peak[1], 3)],
                        "spread": round(frame.spreads[name], 3),
                    }
                    if (peak := frame.peaks.get(name))
                    else None
                )
                for name in frame.nodes
            },
        },
        ensure_ascii=False,
    )


# ============================================================
# Main
# ============================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original_DIST -> trained edge model -> density, all nodes free",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_argument_group("input")
    source.add_argument("--input-log", help="device-monitor log or gat_input.txt")
    source.add_argument("--port", help="serial port; overrides --input-log")
    source.add_argument("--baud", type=int, default=115200)
    source.add_argument(
        "--checkpoint", default=str(SCRIPT_DIR / "checkpoints" / "best_model.pt")
    )

    run = parser.add_argument_group("run")
    run.add_argument("--mode", choices=["replay", "last"], default="replay")
    run.add_argument("--stride", type=int, default=1)
    run.add_argument("--save-dir", help="write one PNG per rendered frame")
    run.add_argument("--dpi", type=int, default=130)
    run.add_argument("--json-out", help="selection JSON per frame; '-' for stdout")
    run.add_argument("--no-show", action="store_true")
    run.add_argument("--seed", type=int, default=7)

    links = parser.add_argument_group("link windows")
    links.add_argument("--window-size", type=int)
    links.add_argument("--min-samples", type=int, default=3)
    links.add_argument("--max-age-scans", type=int, default=15)
    links.add_argument("--min-person-degree", type=int, default=2)
    links.add_argument("--no-anchor-links", action="store_true")

    geometry = parser.add_argument_group("view")
    geometry.add_argument("--box-margin", type=float, default=2.0)
    geometry.add_argument("--x-min", type=float)
    geometry.add_argument("--x-max", type=float)
    geometry.add_argument("--y-min", type=float)
    geometry.add_argument("--y-max", type=float)

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--posterior-sigma", type=float, default=0.35)
    sampling.add_argument("--chains", type=int, default=12)
    sampling.add_argument("--samples", type=int, default=3000)
    sampling.add_argument("--burn-in", type=int, default=2000)
    sampling.add_argument("--thinning", type=int, default=2)
    sampling.add_argument("--proposal-std", type=float, default=0.30)
    sampling.add_argument("--anneal", type=float, default=20.0)
    sampling.add_argument("--jump-prob", type=float, default=0.15)
    sampling.add_argument("--jump-scale", type=float, default=10.0)

    grid = parser.add_argument_group("density grid")
    grid.add_argument("--grid-bins", type=int, default=220)
    grid.add_argument("--smoothing", type=float, default=0.30, help="metres")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.port and not args.input_log:
        found = newest_log()
        if found is None:
            print(
                "No input: pass --input-log or --port.",
                file=sys.stderr,
            )
            return 2
        args.input_log = str(found)
        print(f"[INPUT] newest log: {args.input_log}")

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import torch

    device = torch.device("cpu")
    model, runtime, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    describe_checkpoint(runtime, checkpoint)
    print("[MODE] A and B are free variables; gauge fixed by Procrustes alignment")

    bank = WindowBank(args.window_size or runtime.window_size)
    rng = np.random.default_rng(args.seed)

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    json_handle = None
    if args.json_out:
        json_handle = sys.stdout if args.json_out == "-" else open(
            args.json_out, "w", encoding="utf-8"
        )

    figure = plt.figure("UWB probability, all nodes free", figsize=(18.0, 9.0))
    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    streaming = bool(args.port)
    rendered = 0
    scans = 0
    seq = 0
    ms = 0.0
    previous_state: Optional[np.ndarray] = None
    last_frame: Optional[FreeFrame] = None
    started = time.monotonic()

    def render(frame: FreeFrame) -> None:
        draw_frame(figure, frame)
        if save_dir:
            path = save_dir / f"free_seq{frame.seq:05d}.png"
            figure.savefig(path, dpi=args.dpi)
            shutil.copyfile(path, save_dir / "free_latest.png")
        if json_handle:
            json_handle.write(frame_json(frame) + "\n")
            json_handle.flush()
        if not args.no_show:
            plt.pause(0.001)
        print(
            f"[FRAME] seq={frame.seq} nodes={len(frame.nodes)} "
            f"links={len(frame.selection.links)} "
            f"accept={frame.acceptance:.2f} "
            f"rms={frame.residual_rms:.3f}m"
        )

    try:
        for line in iter_lines(args):
            parsed = parse_uwb_line(line)
            if parsed is None:
                continue

            seq = bank.update(parsed)
            scans += 1
            ms = float(parsed.get("ms", 0) or 0.0)

            if scans % max(args.stride, 1) != 0:
                continue
            if args.mode == "last" and not streaming:
                continue

            frame, previous_state = build_frame(
                model, bank, seq, ms, runtime, device, args, rng, previous_state
            )
            if frame is None:
                continue

            last_frame = frame
            render(frame)
            rendered += 1

            if not args.no_show and not plt.fignum_exists(figure.number):
                break

        if args.mode == "last" and not streaming and scans:
            frame, previous_state = build_frame(
                model, bank, seq, ms, runtime, device, args, rng, previous_state
            )
            if frame is not None:
                last_frame = frame
                render(frame)
                rendered += 1

    except KeyboardInterrupt:
        print("\n[STOP]")
    finally:
        if json_handle and json_handle is not sys.stdout:
            json_handle.close()

    print(
        f"[DONE] {scans} scans parsed, {rendered} frames rendered "
        f"in {time.monotonic() - started:.1f} s"
    )
    if save_dir:
        print(f"[DONE] images in {save_dir}")
    if last_frame is None:
        print("[DONE] nothing rendered: no scan had enough links", file=sys.stderr)
        return 1

    if not args.no_show:
        plt.ioff()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
