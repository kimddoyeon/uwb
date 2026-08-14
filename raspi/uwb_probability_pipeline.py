#!/usr/bin/env python3
"""Collected UWB log or live serial -> trained edge model -> position density.

This is the Raspberry Pi pipeline for the data the B collector already prints.
It replaces the two-process chain (``gat_distance_filter.py | position_density.py``)
with a single process that runs the *trained* checkpoint instead of the rule
fallback, and that builds the density from feasible whole-node configurations
instead of per-person mirrored triangles.

Stages
------
1.  Parse ``Original_DIST`` rows (log file replay or live serial).
2.  Keep a rolling window of raw measurements per link and build the four edge
    features the model was trained on: mean distance, distance std, distance
    slope, mean RSS -- normalized with the checkpoint's own scales.
3.  Score every link that currently has data, then keep Top-K + alpha with the
    checkpoint's ``k_ratio`` / ``confidence_threshold``.  AB is always kept (it
    defines the A-B baseline) and every person keeps at least two incident
    links when it has them, matching ``selectTopLinksWithCoverage.m``.
4.  Sample whole-node configurations that satisfy the selected distances with
    Metropolis-Hastings (A at the origin, B on the +x axis at the measured AB),
    and turn the per-person marginals into smoothed probability densities.

Examples
--------
Replay the newest device-monitor log and watch the density evolve::

    python raspi/uwb_probability_pipeline.py

Only the final frame of one specific log, saved as PNG without a window::

    python raspi/uwb_probability_pipeline.py \
        --input-log logs/device-monitor-260812-214621.log \
        --mode last --save-dir density_output --no-show

Live on the Pi, also feeding the existing JSON consumer::

    python3 raspi/uwb_probability_pipeline.py --port /dev/ttyUSB0 --json-out -
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from gat_distance_filter import PAIR_LABELS, parse_uwb_line


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

NODES = ("A", "B", "C", "D", "E", "F")
NODE_INDEX = {name: idx for idx, name in enumerate(NODES)}
PEOPLE = ("C", "D", "E", "F")
DRONE_COUNT = 2

# Links to A or B anchor a person in the A-B frame.  gat_distance_filter.py
# already force-keeps exactly this set for position_density.py: a link that
# Top-K ranks below the cutoff still carries the only absolute position
# information a person has, and dropping it smears that person over a circle.
ANCHOR_LINKS = frozenset(
    label for label in PAIR_LABELS if label[0] in "AB" or label[1] in "AB"
)

# Same colours position_density.py already uses for the four firefighters.
PERSON_COLOR = {
    "C": (0.90, 0.05, 0.05),
    "D": (1.00, 0.78, 0.00),
    "E": (0.00, 0.65, 0.12),
    "F": (0.05, 0.25, 1.00),
}


# ============================================================
# Checkpoint
# ============================================================
@dataclass
class RuntimeConfig:
    architecture: str
    dist_scale: float
    std_scale: float
    slope_scale: float
    rss_mean: float
    rss_std: float
    k_ratio: float
    confidence_threshold: float
    window_size: int


def load_checkpoint(path: Path, device):
    """Load the checkpoint and instantiate whichever model it actually holds.

    ``train.py`` saves ``LightweightGAT`` and the ablation run saves
    ``EdgeOnlyMLP`` under the same file name and the same ``model_config``, so
    the architecture is decided by the state-dict keys, not by the config.
    """
    import torch

    from gat_model import EdgeOnlyMLP, LightweightGAT

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        checkpoint = torch.load(path, map_location=device)

    state = checkpoint["model_state_dict"]
    model_cfg = checkpoint["model_config"]
    topk_cfg = checkpoint["topk_config"]
    norm = checkpoint["normalization"]
    data_cfg = checkpoint.get("data_config", {})

    if any(key.startswith("net.") for key in state):
        architecture = "EdgeOnlyMLP"
        model = EdgeOnlyMLP(
            edge_feat_dim=int(model_cfg["edge_feat_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
        )
    else:
        architecture = "LightweightGAT"
        model = LightweightGAT(
            node_feat_dim=int(model_cfg["node_feat_dim"]),
            edge_feat_dim=int(model_cfg["edge_feat_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            leaky_slope=float(model_cfg["leaky_relu_slope"]),
        )

    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    rss_mean = norm.get("rss_mean", norm.get("signal_mean"))
    rss_std = norm.get("rss_std", norm.get("signal_std"))

    config = RuntimeConfig(
        architecture=architecture,
        dist_scale=float(norm["dist_scale"]),
        std_scale=max(float(norm["std_scale"]), 1e-6),
        slope_scale=max(float(norm["slope_scale"]), 1e-6),
        rss_mean=float(rss_mean),
        rss_std=max(float(rss_std), 1e-6),
        k_ratio=float(topk_cfg["k_ratio"]),
        confidence_threshold=float(topk_cfg["confidence_threshold"]),
        window_size=int(data_cfg.get("window_size", 8)),
    )
    return model, config, checkpoint


def describe_checkpoint(config: RuntimeConfig, checkpoint: dict) -> None:
    print(f"[MODEL] {config.architecture} from checkpoint")
    print(
        f"[MODEL] window={config.window_size} "
        f"k_ratio={config.k_ratio:.4f} "
        f"threshold={config.confidence_threshold:.2f}"
    )
    print(
        f"[MODEL] rss_mean={config.rss_mean:.2f} dBm "
        f"rss_std={config.rss_std:.2f} dB "
        f"dist_scale={config.dist_scale:.1f} m"
    )

    metrics = checkpoint.get("val_metrics") or {}
    if metrics:
        print(
            "[MODEL] val: "
            + " ".join(
                f"{key}={float(value):.3f}"
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            )
        )

    label = checkpoint.get("label_definition") or {}
    if label:
        print(f"[MODEL] label: {label}")

    if config.architecture == "EdgeOnlyMLP":
        print(
            "[WARNING] This checkpoint holds the edge-only MLP ablation, not the\n"
            "[WARNING] graph-attention model: link scores use each link's own four\n"
            "[WARNING] features and ignore the graph. Retrain with train.py to get\n"
            "[WARNING] a real GAT checkpoint; the rest of this pipeline is unchanged.",
            file=sys.stderr,
        )


# ============================================================
# Rolling per-link windows
# ============================================================
@dataclass
class Measurement:
    distance: float
    rssi_dbm: float
    seq: int
    ms: float


class LinkWindow:
    """Last ``maxlen`` raw measurements of one link."""

    def __init__(self, maxlen: int):
        self.samples: Deque[Measurement] = deque(maxlen=max(1, maxlen))

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, sample: Measurement) -> None:
        self.samples.append(sample)

    @property
    def last_seq(self) -> int:
        return self.samples[-1].seq

    def stats(self, fallback_rssi: float) -> Tuple[float, float, float, float]:
        """mean distance, distance std, distance slope, mean RSS.

        Identical to ``dataset.sample_window`` so the runtime features match the
        ones the checkpoint was trained on.
        """
        distances = np.asarray(
            [sample.distance for sample in self.samples], dtype=np.float32
        )
        mean_distance = float(np.mean(distances))
        std_distance = float(np.std(distances))

        if distances.size >= 4:
            half = distances.size // 2
            slope = float(np.mean(distances[half:]) - np.mean(distances[:half]))
        else:
            slope = 0.0

        rss = np.asarray(
            [sample.rssi_dbm for sample in self.samples], dtype=np.float32
        )
        rss = rss[np.isfinite(rss)]
        mean_rss = float(np.mean(rss)) if rss.size else float(fallback_rssi)

        return mean_distance, std_distance, slope, mean_rss


class WindowBank:
    def __init__(self, window_size: int):
        self.windows = {label: LinkWindow(window_size) for label in PAIR_LABELS}

    def update(self, parsed: dict) -> int:
        seq = int(float(parsed.get("seq", 0)))
        ms = float(parsed.get("ms", 0) or 0.0)
        powers = parsed.get("rx_power_dbm", {})

        for label in PAIR_LABELS:
            distance = parsed["distances"].get(label)
            if distance is None:
                continue
            rssi = powers.get(label)
            self.windows[label].add(
                Measurement(
                    distance=float(distance),
                    rssi_dbm=float(rssi) if rssi is not None else math.nan,
                    seq=seq,
                    ms=ms,
                )
            )
        return seq

    def active_links(
        self, seq: int, min_samples: int, max_age_scans: int
    ) -> List[str]:
        active = []
        for label in PAIR_LABELS:
            window = self.windows[label]
            if len(window) < min_samples:
                continue
            if seq - window.last_seq > max_age_scans:
                continue
            active.append(label)
        return active


# ============================================================
# Model scoring and link selection
# ============================================================
def score_links(
    model,
    bank: WindowBank,
    links: Sequence[str],
    config: RuntimeConfig,
    device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return per-link reliability score and the window mean distance."""
    import torch

    node_count = len(NODES)
    node_feat = np.zeros((node_count, 2), dtype=np.float32)
    node_feat[:DRONE_COUNT, 0] = 1.0
    node_feat[DRONE_COUNT:, 1] = 1.0

    edge_feat = np.zeros((node_count, node_count, 4), dtype=np.float32)
    adj_mask = np.zeros((node_count, node_count), dtype=bool)
    distances: Dict[str, float] = {}

    for label in links:
        mean_d, std_d, slope, mean_rss = bank.windows[label].stats(config.rss_mean)
        feature = np.asarray(
            [
                mean_d / config.dist_scale,
                std_d / config.std_scale,
                slope / config.slope_scale,
                (mean_rss - config.rss_mean) / config.rss_std,
            ],
            dtype=np.float32,
        )
        i, j = NODE_INDEX[label[0]], NODE_INDEX[label[1]]
        edge_feat[i, j] = feature
        edge_feat[j, i] = feature
        adj_mask[i, j] = True
        adj_mask[j, i] = True
        distances[label] = mean_d

    # A node with no measured link this frame is treated the way training
    # treated padding slots.
    node_mask = adj_mask.any(axis=1)

    with torch.no_grad():
        logits = model(
            torch.from_numpy(node_feat).unsqueeze(0).to(device),
            torch.from_numpy(edge_feat).unsqueeze(0).to(device),
            torch.from_numpy(adj_mask).unsqueeze(0).to(device),
            torch.from_numpy(node_mask).unsqueeze(0).to(device),
        )[0]
        probabilities = torch.sigmoid(logits).cpu().numpy()

    scores = {
        label: float(probabilities[NODE_INDEX[label[0]], NODE_INDEX[label[1]]])
        for label in links
    }
    return scores, distances


@dataclass
class Selection:
    links: List[str]
    #: every scored link this frame, not only the selected ones
    scores: Dict[str, float]
    distances: Dict[str, float]
    k_min: int
    alpha: int
    forced: List[str]
    dropped: List[str]


def select_links(
    scores: Dict[str, float],
    distances: Dict[str, float],
    k_ratio: float,
    threshold: float,
    min_person_degree: int,
    keep_anchor_links: bool,
) -> Selection:
    """Top-K + alpha, then the project's anchor and per-person coverage rules."""
    ranked = sorted(scores, key=lambda label: scores[label], reverse=True)
    total = len(ranked)
    k_min = max(1, min(int(round(total * k_ratio)), total)) if total else 0

    selected = set(ranked[:k_min])
    for label in ranked[k_min:]:
        if scores[label] >= threshold:
            selected.add(label)
        else:
            # ranked is sorted, so nothing further can pass either.
            break
    alpha = len(selected) - k_min

    forced: List[str] = []

    # B's position is defined by the AB baseline, so AB can never be dropped.
    if "AB" in scores and "AB" not in selected:
        selected.add("AB")
        forced.append("AB")

    # Keep every measured link to A or B, the same way gat_distance_filter.py
    # force-keeps them before position_density.py.
    if keep_anchor_links:
        for label in ranked:
            if label in ANCHOR_LINKS and label not in selected:
                selected.add(label)
                forced.append(label)

    # Every person keeps at least two incident links when it has them,
    # otherwise its marginal collapses into the uniform prior.
    for person in PEOPLE:
        incident = [label for label in ranked if person in label]
        kept = [label for label in incident if label in selected]
        for label in incident:
            if len(kept) >= min_person_degree:
                break
            if label not in selected:
                selected.add(label)
                kept.append(label)
                forced.append(label)

    ordered = sorted(selected, key=lambda label: scores[label], reverse=True)
    dropped = [label for label in ranked if label not in selected]

    return Selection(
        links=ordered,
        scores=dict(scores),
        distances=dict(distances),
        k_min=k_min,
        alpha=alpha,
        forced=forced,
        dropped=dropped,
    )


# ============================================================
# Feasible-configuration sampling
# ============================================================
@dataclass
class SamplerConfig:
    posterior_sigma: float
    chains: int
    samples: int
    burn_in: int
    thinning: int
    proposal_std: float
    anneal_start: float
    jump_prob: float
    jump_scale: float
    mirror: bool


@dataclass
class Box:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class ConfigurationEnergy:
    """Sum of squared range residuals over the selected links.

    ``raw`` leaves out the 1/sigma^2 factor so the annealing schedule can change
    sigma without recomputing anything.
    """

    def __init__(
        self,
        targets: Sequence[str],
        selection: Selection,
        a_position: np.ndarray,
        b_position: np.ndarray,
    ):
        person_index = {name: idx for idx, name in enumerate(targets)}
        fixed = {"A": a_position, "B": b_position}

        pp_i, pp_j, pp_d = [], [], []
        pf_i, pf_xy, pf_d = [], [], []

        for label in selection.links:
            distance = selection.distances[label]
            u, v = label[0], label[1]
            if u in person_index and v in person_index:
                pp_i.append(person_index[u])
                pp_j.append(person_index[v])
                pp_d.append(distance)
            elif u in person_index and v in fixed:
                pf_i.append(person_index[u])
                pf_xy.append(fixed[v])
                pf_d.append(distance)
            elif v in person_index and u in fixed:
                pf_i.append(person_index[v])
                pf_xy.append(fixed[u])
                pf_d.append(distance)
            # AB constrains no person: it already defines the frame.

        self.pp_i = np.asarray(pp_i, dtype=np.intp)
        self.pp_j = np.asarray(pp_j, dtype=np.intp)
        self.pp_d = np.asarray(pp_d, dtype=np.float64)
        self.pf_i = np.asarray(pf_i, dtype=np.intp)
        self.pf_xy = (
            np.asarray(pf_xy, dtype=np.float64)
            if pf_xy
            else np.zeros((0, 2), dtype=np.float64)
        )
        self.pf_d = np.asarray(pf_d, dtype=np.float64)

        # Per-person view of the same links, so a single-person move only has
        # to rescore the links incident to that person.
        self.people = len(targets)
        self.neighbour: List[np.ndarray] = []
        self.neighbour_d: List[np.ndarray] = []
        self.anchor_xy: List[np.ndarray] = []
        self.anchor_d: List[np.ndarray] = []

        for person in range(self.people):
            others, others_d = [], []
            for i, j, distance in zip(self.pp_i, self.pp_j, self.pp_d):
                if i == person:
                    others.append(int(j))
                    others_d.append(distance)
                elif j == person:
                    others.append(int(i))
                    others_d.append(distance)
            self.neighbour.append(np.asarray(others, dtype=np.intp))
            self.neighbour_d.append(np.asarray(others_d, dtype=np.float64))

            mask = self.pf_i == person
            self.anchor_xy.append(
                self.pf_xy[mask]
                if self.pf_xy.size
                else np.zeros((0, 2), dtype=np.float64)
            )
            self.anchor_d.append(self.pf_d[mask] if self.pf_d.size else self.pf_d)

    def local_raw(
        self, state: np.ndarray, person: int, coordinates: np.ndarray
    ) -> np.ndarray:
        """Energy of the links incident to ``person`` if it sat at ``coordinates``.

        ``state``: (chains, people, 2); ``coordinates``: (chains, 2).
        """
        energy = np.zeros(coordinates.shape[0], dtype=np.float64)

        others = self.neighbour[person]
        if others.size:
            delta = coordinates[:, None, :] - state[:, others, :]
            predicted = np.sqrt(np.einsum("cld,cld->cl", delta, delta))
            energy += 0.5 * np.sum((predicted - self.neighbour_d[person]) ** 2, axis=1)

        anchors = self.anchor_xy[person]
        if anchors.size:
            delta = coordinates[:, None, :] - anchors[None, :, :]
            predicted = np.sqrt(np.einsum("cld,cld->cl", delta, delta))
            energy += 0.5 * np.sum((predicted - self.anchor_d[person]) ** 2, axis=1)

        return energy

    @property
    def constraint_count(self) -> int:
        return int(self.pp_d.size + self.pf_d.size)

    def degree(self, targets: Sequence[str]) -> Dict[str, int]:
        counts = {name: 0 for name in targets}
        for idx in np.concatenate([self.pp_i, self.pp_j, self.pf_i]).astype(int):
            counts[targets[idx]] += 1
        return counts

    def residual_rms(self, samples: np.ndarray) -> float:
        """Median per-sample RMS range residual, in metres.

        Much larger than the posterior sigma means the sampler never reached a
        configuration that fits, or that the selected distances contradict each
        other -- in either case the density below is not trustworthy.
        """
        if not samples.size or self.constraint_count == 0:
            return float("nan")
        per_sample = 2.0 * self.raw(samples) / self.constraint_count
        return float(np.sqrt(np.median(per_sample)))

    def raw(self, state: np.ndarray) -> np.ndarray:
        """state: (chains, people, 2) -> (chains,) sum of 0.5 * residual^2."""
        energy = np.zeros(state.shape[0], dtype=np.float64)

        if self.pp_d.size:
            delta = state[:, self.pp_i, :] - state[:, self.pp_j, :]
            predicted = np.sqrt(np.einsum("cld,cld->cl", delta, delta))
            energy += 0.5 * np.sum((predicted - self.pp_d) ** 2, axis=1)

        if self.pf_d.size:
            delta = state[:, self.pf_i, :] - self.pf_xy[None, :, :]
            predicted = np.sqrt(np.einsum("cld,cld->cl", delta, delta))
            energy += 0.5 * np.sum((predicted - self.pf_d) ** 2, axis=1)

        return energy


def sample_configurations(
    energy: ConfigurationEnergy,
    people: int,
    box: Box,
    cfg: SamplerConfig,
    rng: np.random.Generator,
    initial_state: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Annealed random-walk Metropolis over the person coordinates.

    One sweep moves every person once, so a person starting 25 m from its
    solution still gets ``burn_in`` chances to walk there.  Only the links
    incident to the moved person are rescored, which makes a sweep no more
    expensive than a single joint proposal.

    Several independent chains run in lockstep: one chain cannot cross the
    energy barrier between mirror solutions, but a handful of chains started
    from different random configurations populate both.

    A fraction of the proposals are long jumps.  An under-constrained person --
    one link only -- lives on a circle, and a small-step random walk diffuses
    along that circle far too slowly to fill it within one frame.
    """
    chains = max(1, cfg.chains)
    per_chain = max(1, math.ceil(cfg.samples / chains))

    state = np.column_stack(
        (
            rng.uniform(box.x_min, box.x_max, chains * people),
            rng.uniform(box.y_min, box.y_max, chains * people),
        )
    ).reshape(chains, people, 2)

    if initial_state is not None and initial_state.shape == (people, 2):
        # Keep one chain anchored on the previous frame for temporal continuity.
        state[0] = initial_state

    kept = np.zeros((chains, per_chain, people, 2), dtype=np.float32)
    total_sweeps = cfg.burn_in + per_chain * cfg.thinning
    proposed = 0
    accepted = 0
    save_index = 0

    for sweep in range(total_sweeps):
        if sweep < cfg.burn_in and cfg.anneal_start > 1.0:
            progress = sweep / max(cfg.burn_in - 1, 1)
            temperature = cfg.anneal_start ** (1.0 - progress)
        else:
            temperature = 1.0

        inverse_variance = 1.0 / (cfg.posterior_sigma * temperature) ** 2
        step_std = cfg.proposal_std * min(math.sqrt(temperature), 6.0)

        for person in range(people):
            current = state[:, person, :]
            scale = np.where(
                rng.random(chains) < cfg.jump_prob, cfg.jump_scale, 1.0
            )[:, None]
            candidate = current + rng.normal(0.0, step_std, size=(chains, 2)) * scale

            inside = (
                (candidate[:, 0] >= box.x_min)
                & (candidate[:, 0] <= box.x_max)
                & (candidate[:, 1] >= box.y_min)
                & (candidate[:, 1] <= box.y_max)
            )

            delta_energy = energy.local_raw(
                state, person, candidate
            ) - energy.local_raw(state, person, current)

            log_ratio = -delta_energy * inverse_variance
            accept = inside & (
                np.log(rng.random(chains)) < np.minimum(0.0, log_ratio)
            )

            state[accept, person, :] = candidate[accept]
            proposed += chains
            accepted += int(np.count_nonzero(accept))

        if sweep >= cfg.burn_in and (sweep - cfg.burn_in) % cfg.thinning == 0:
            if save_index < per_chain:
                kept[:, save_index] = state
                save_index += 1

    samples = kept[:, :save_index].reshape(-1, people, 2)
    acceptance = accepted / max(proposed, 1)

    if cfg.mirror and samples.size:
        # A and B both sit on y = 0, so reflecting every person across the
        # x axis gives an exactly equally likely configuration.
        mirrored = samples.copy()
        mirrored[:, :, 1] *= -1.0
        samples = np.concatenate([samples, mirrored], axis=0)

    return samples, acceptance


def search_box(selection: Selection, a_position: np.ndarray, ab: float) -> Box:
    """Generous prior box: every person must be reachable inside it."""
    reach = max(
        [ab] + [selection.distances[label] for label in selection.links],
        default=10.0,
    )
    margin = 0.5 * reach + 2.0
    span = reach + margin
    return Box(
        x_min=float(a_position[0] - span),
        x_max=float(a_position[0] + ab + span),
        y_min=-span,
        y_max=span,
    )


def plot_box(
    samples: np.ndarray,
    a_position: np.ndarray,
    b_position: np.ndarray,
    margin: float,
    override: Dict[str, Optional[float]],
) -> Box:
    """Tight, y-symmetric view around the bulk of the samples plus A and B."""
    if samples.size:
        flat = samples.reshape(-1, 2)
        x_low, x_high = np.percentile(flat[:, 0], [0.5, 99.5])
        y_low, y_high = np.percentile(flat[:, 1], [0.5, 99.5])
    else:
        x_low = x_high = y_low = y_high = 0.0

    x_min = min(x_low, a_position[0], b_position[0]) - margin
    x_max = max(x_high, a_position[0], b_position[0]) + margin
    y_extent = max(abs(y_low), abs(y_high), 1.0) + margin

    box = Box(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(-y_extent),
        y_max=float(y_extent),
    )
    for name, value in override.items():
        if value is not None:
            setattr(box, name, float(value))
    return box


# ============================================================
# Density
# ============================================================
def gaussian_kernel(sigma_bins: float) -> np.ndarray:
    sigma_bins = max(sigma_bins, 0.6)
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    axis = np.arange(-radius, radius + 1, dtype=float)
    x, y = np.meshgrid(axis, axis)
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma_bins**2))
    return kernel / kernel.sum()


def fft_convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    shape = (
        image.shape[0] + kernel.shape[0] - 1,
        image.shape[1] + kernel.shape[1] - 1,
    )
    axes = (0, 1)
    full = np.fft.irfftn(
        np.fft.rfftn(image, shape, axes=axes)
        * np.fft.rfftn(kernel, shape, axes=axes),
        shape,
        axes=axes,
    )
    y0 = (kernel.shape[0] - 1) // 2
    x0 = (kernel.shape[1] - 1) // 2
    return full[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]]


def person_density(
    person_samples: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    smoothing_m: float,
) -> np.ndarray:
    counts, _, _ = np.histogram2d(
        person_samples[:, 1], person_samples[:, 0], bins=(y_edges, x_edges)
    )
    step = float(x_edges[1] - x_edges[0])
    density = fft_convolve_same(counts, gaussian_kernel(smoothing_m / max(step, 1e-6)))
    density = np.clip(density, 0.0, None)
    peak = float(density.max())
    return density / peak if peak > 0 else density


def density_peak(
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    upper_half_only: bool,
) -> Optional[Tuple[float, float]]:
    """Most likely cell.

    With mirroring on, the map is exactly symmetric about y = 0 and argmax
    would always land in the lower half by tie-break, so restrict the search to
    y >= 0 and let the caller report the peak as +/- y.
    """
    if density.max() <= 0:
        return None

    search = density
    offset = 0
    if upper_half_only:
        centres = 0.5 * (y_edges[:-1] + y_edges[1:])
        rows = np.flatnonzero(centres >= 0.0)
        if rows.size:
            offset = int(rows[0])
            search = density[offset:, :]

    row, col = np.unravel_index(int(np.argmax(search)), search.shape)
    row += offset
    x = 0.5 * (x_edges[col] + x_edges[col + 1])
    y = 0.5 * (y_edges[row] + y_edges[row + 1])
    return float(x), float(y)


# ============================================================
# Rendering
# ============================================================
def person_colormap(person: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        f"white_to_{person}", [(1.0, 1.0, 1.0), PERSON_COLOR[person]]
    )


@dataclass
class Frame:
    seq: int
    ms: float
    selection: Selection
    targets: List[str]
    degrees: Dict[str, int]
    densities: Dict[str, np.ndarray]
    peaks: Dict[str, Optional[Tuple[float, float]]]
    x_edges: np.ndarray
    y_edges: np.ndarray
    a_position: np.ndarray
    b_position: np.ndarray
    box: Box
    acceptance: float
    residual_rms: float
    posterior_sigma: float
    sample_count: int
    architecture: str
    mirrored: bool


def draw_frame(figure, frame: Frame) -> None:
    from matplotlib.lines import Line2D

    figure.clear()
    axes = figure.subplots(2, 3, squeeze=False)
    flat = list(axes.flat)
    extent = (frame.box.x_min, frame.box.x_max, frame.box.y_min, frame.box.y_max)

    def draw_anchors(axis):
        axis.scatter(
            frame.a_position[0], frame.a_position[1],
            marker="^", s=80, c="black", zorder=6,
        )
        axis.scatter(
            frame.b_position[0], frame.b_position[1],
            marker="s", s=70, c="white", edgecolors="black", zorder=6,
        )
        axis.annotate("A", frame.a_position, textcoords="offset points",
                      xytext=(6, 4), weight="bold", zorder=7)
        axis.annotate("B", frame.b_position, textcoords="offset points",
                      xytext=(6, 4), weight="bold", zorder=7)
        axis.set_xlim(frame.box.x_min, frame.box.x_max)
        axis.set_ylim(frame.box.y_min, frame.box.y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")

    overview = flat[0]
    for person in frame.targets:
        density = frame.densities[person]
        if density.max() <= 0:
            continue
        overview.contour(
            density,
            levels=[0.15, 0.35, 0.60, 0.82],
            extent=extent,
            origin="lower",
            colors=[PERSON_COLOR[person]],
            linewidths=1.3,
        )
    draw_anchors(overview)
    overview.set_title("Combined probability contours")
    overview.legend(
        handles=[
            Line2D([0], [0], color=PERSON_COLOR[person], lw=2, label=person)
            for person in frame.targets
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    for index, person in enumerate(PEOPLE, start=1):
        axis = flat[index]
        density = frame.densities.get(person)
        if density is None:
            axis.axis("off")
            continue
        axis.imshow(
            density,
            origin="lower",
            extent=extent,
            cmap=person_colormap(person),
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="bilinear",
        )
        peak = frame.peaks.get(person)
        if peak is not None:
            ys = [peak[1], -peak[1]] if frame.mirrored else [peak[1]]
            axis.scatter(
                [peak[0]] * len(ys), ys, marker="x", s=60,
                c=[PERSON_COLOR[person]] * len(ys), linewidths=2.0, zorder=8,
            )
        draw_anchors(axis)
        degree = frame.degrees.get(person, 0)
        axis.set_title(f"{person} position probability (links={degree})")
        if degree == 0:
            axis.text(
                0.5, 0.5, "no selected link", transform=axis.transAxes,
                ha="center", va="center", fontsize=11, color="0.4",
            )

    info = flat[5]
    info.axis("off")
    lines = [
        f"model:     {frame.architecture}",
        f"scan seq:  {frame.seq}   t={frame.ms / 1000.0:.1f} s",
        f"people:    {len(frame.targets)}",
        f"K:         {frame.selection.k_min}",
        f"alpha:     {frame.selection.alpha}",
        f"forced:    {', '.join(frame.selection.forced) or '-'}",
        f"samples:   {frame.sample_count}  (accept {frame.acceptance:.2f})",
        f"fit RMS:   {frame.residual_rms:.3f} m  (sigma {frame.posterior_sigma:.2f})",
        f"AB:        {np.linalg.norm(frame.b_position - frame.a_position):.3f} m",
        "",
        "selected links:",
    ]
    for rank, label in enumerate(frame.selection.links, start=1):
        lines.append(
            f"{rank:2d}. {label}  d={frame.selection.distances[label]:6.2f} m"
            f"  s={frame.selection.scores[label]:.3f}"
        )
    if frame.selection.dropped:
        lines.append("")
        lines.append(
            "dropped: "
            + ", ".join(
                f"{label}({frame.selection.scores[label]:.2f})"
                for label in frame.selection.dropped
            )
        )
    lines.append("")
    lines.append("peak of each marginal:")
    sign = "+/-" if frame.mirrored else ""
    for person in frame.targets:
        peak = frame.peaks.get(person)
        lines.append(
            f"  {person}: ({peak[0]:6.2f}, {sign}{abs(peak[1]):5.2f}) m"
            if peak
            else f"  {person}: -"
        )
    lines.extend(
        [
            "",
            "Bright area = more whole-node",
            "configurations fit every selected",
            "distance.",
        ]
    )
    if frame.mirrored:
        lines.extend(
            [
                "Mirrored across y=0: A and B",
                "alone cannot resolve the side.",
            ]
        )
    info.text(
        0.02, 0.98, "\n".join(lines),
        va="top", ha="left", family="monospace", fontsize=8,
        transform=info.transAxes,
    )

    figure.suptitle(
        f"UWB position probability density | scan {frame.seq}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))


# ============================================================
# Input
# ============================================================
def newest_log() -> Optional[Path]:
    candidates = sorted(
        (REPO_ROOT / "logs").glob("device-monitor-*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(
            (REPO_ROOT / "data_save").glob("*_gat_input.txt"),
            key=lambda path: path.stat().st_mtime,
        )
    return candidates[-1] if candidates else None


def iter_lines(args) -> Iterator[str]:
    if args.port:
        try:
            import serial
        except ImportError:
            print("Install pyserial first: pip3 install pyserial", file=sys.stderr)
            raise
        with serial.Serial(args.port, args.baud, timeout=1) as device:
            time.sleep(1.0)
            while True:
                raw = device.readline()
                if raw:
                    yield raw.decode("utf-8", errors="ignore")
        return

    with open(args.input_log, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            yield line


# ============================================================
# Frame processing
# ============================================================
def build_frame(
    model,
    bank: WindowBank,
    seq: int,
    ms: float,
    runtime: RuntimeConfig,
    device,
    args,
    rng: np.random.Generator,
    last_ab: Optional[float],
    previous_state: Optional[np.ndarray],
) -> Tuple[Optional[Frame], Optional[float], Optional[np.ndarray]]:
    links = bank.active_links(seq, args.min_samples, args.max_age_scans)
    if not links:
        return None, last_ab, previous_state

    scores, distances = score_links(model, bank, links, runtime, device)
    selection = select_links(
        scores,
        distances,
        runtime.k_ratio,
        runtime.confidence_threshold,
        args.min_person_degree,
        not args.no_anchor_links,
    )

    ab = selection.distances.get("AB", last_ab)
    if ab is None or not math.isfinite(ab) or ab <= 0:
        return None, last_ab, previous_state
    last_ab = ab

    a_position = np.asarray([args.a_x, args.a_y], dtype=np.float64)
    b_position = a_position + np.asarray([ab, 0.0], dtype=np.float64)

    targets = [
        person
        for person in PEOPLE
        if any(person in label for label in selection.links)
    ]
    if not targets:
        return None, last_ab, previous_state

    energy = ConfigurationEnergy(targets, selection, a_position, b_position)
    degrees = energy.degree(targets)

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
        mirror=not args.no_mirror,
    )
    prior = search_box(selection, a_position, ab)
    samples, acceptance = sample_configurations(
        energy, len(targets), prior, sampler, rng,
        initial_state=(
            previous_state
            if previous_state is not None and previous_state.shape == (len(targets), 2)
            else None
        ),
    )
    previous_state = samples[-1].astype(np.float64) if samples.size else previous_state

    view = plot_box(
        samples,
        a_position,
        b_position,
        args.box_margin,
        {
            "x_min": args.x_min,
            "x_max": args.x_max,
            "y_min": args.y_min,
            "y_max": args.y_max,
        },
    )
    x_edges = np.linspace(view.x_min, view.x_max, args.grid_bins + 1)
    y_edges = np.linspace(view.y_min, view.y_max, args.grid_bins + 1)

    densities: Dict[str, np.ndarray] = {}
    peaks: Dict[str, Optional[Tuple[float, float]]] = {}
    for index, person in enumerate(targets):
        density = person_density(
            samples[:, index, :], x_edges, y_edges, args.smoothing
        )
        densities[person] = density
        peaks[person] = density_peak(density, x_edges, y_edges, sampler.mirror)

    frame = Frame(
        seq=seq,
        ms=ms,
        selection=selection,
        targets=targets,
        degrees=degrees,
        densities=densities,
        peaks=peaks,
        x_edges=x_edges,
        y_edges=y_edges,
        a_position=a_position,
        b_position=b_position,
        box=view,
        acceptance=acceptance,
        residual_rms=energy.residual_rms(samples),
        posterior_sigma=args.posterior_sigma,
        sample_count=int(samples.shape[0]),
        architecture=runtime.architecture,
        mirrored=sampler.mirror,
    )
    return frame, last_ab, previous_state


def frame_json(frame: Frame) -> str:
    return json.dumps(
        {
            "type": "distribution_update",
            "seq": frame.seq,
            "ms": frame.ms,
            "model": frame.architecture,
            "people": len(frame.targets),
            "k_min": frame.selection.k_min,
            "alpha": frame.selection.alpha,
            "forced": frame.selection.forced,
            "selected": [
                {
                    "link": label,
                    "distance": round(frame.selection.distances[label], 3),
                    "score": round(frame.selection.scores[label], 4),
                }
                for label in frame.selection.links
            ],
            "peaks": {
                person: (
                    [round(peak[0], 3), round(peak[1], 3)]
                    if (peak := frame.peaks.get(person))
                    else None
                )
                for person in frame.targets
            },
        },
        ensure_ascii=False,
    )


# ============================================================
# Main
# ============================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original_DIST -> trained edge model -> position density",
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
    run.add_argument(
        "--mode",
        choices=["replay", "last"],
        default="replay",
        help="replay every scan, or only render the final one",
    )
    run.add_argument("--stride", type=int, default=1, help="render every Nth scan")
    run.add_argument("--save-dir", help="write one PNG per rendered frame")
    run.add_argument("--dpi", type=int, default=130)
    run.add_argument("--json-out", help="selection JSON per frame; '-' for stdout")
    run.add_argument("--no-show", action="store_true", help="headless, no window")
    run.add_argument("--seed", type=int, default=7)

    links = parser.add_argument_group("link windows")
    links.add_argument("--window-size", type=int, help="override checkpoint window")
    links.add_argument("--min-samples", type=int, default=3)
    links.add_argument(
        "--max-age-scans",
        type=int,
        default=15,
        help="drop a link that has not been measured for this many scans",
    )
    links.add_argument("--min-person-degree", type=int, default=2)
    links.add_argument(
        "--no-anchor-links",
        action="store_true",
        help="let Top-K drop links to A or B (shows raw model selection)",
    )

    geometry = parser.add_argument_group("geometry")
    geometry.add_argument("--a-x", type=float, default=0.0)
    geometry.add_argument("--a-y", type=float, default=0.0)
    geometry.add_argument("--box-margin", type=float, default=2.0)
    geometry.add_argument("--x-min", type=float)
    geometry.add_argument("--x-max", type=float)
    geometry.add_argument("--y-min", type=float)
    geometry.add_argument("--y-max", type=float)

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--posterior-sigma", type=float, default=0.35)
    sampling.add_argument("--chains", type=int, default=12)
    sampling.add_argument("--samples", type=int, default=3000)
    sampling.add_argument("--burn-in", type=int, default=1500)
    sampling.add_argument("--thinning", type=int, default=2)
    sampling.add_argument("--proposal-std", type=float, default=0.30)
    sampling.add_argument(
        "--anneal",
        type=float,
        default=20.0,
        help="starting sigma multiplier for the burn-in schedule; 1 disables it",
    )
    sampling.add_argument(
        "--jump-prob",
        type=float,
        default=0.15,
        help="fraction of proposals that are long jumps",
    )
    sampling.add_argument("--jump-scale", type=float, default=10.0)
    sampling.add_argument(
        "--no-mirror",
        action="store_true",
        help="do not add the exact y-mirrored copy of each sample",
    )

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
                "No input: pass --input-log or --port "
                "(no logs/device-monitor-*.log found).",
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

    window_size = args.window_size or runtime.window_size
    bank = WindowBank(window_size)
    rng = np.random.default_rng(args.seed)

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    json_handle = None
    if args.json_out:
        json_handle = sys.stdout if args.json_out == "-" else open(
            args.json_out, "w", encoding="utf-8"
        )

    figure = plt.figure("UWB probability distribution", figsize=(15.5, 9.0))
    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    streaming = bool(args.port)
    rendered = 0
    scans = 0
    last_ab: Optional[float] = None
    previous_state: Optional[np.ndarray] = None
    last_frame: Optional[Frame] = None
    started = time.monotonic()

    def render(frame: Frame) -> None:
        draw_frame(figure, frame)
        if save_dir:
            path = save_dir / f"density_seq{frame.seq:05d}.png"
            figure.savefig(path, dpi=args.dpi)
            # Rendering the same figure twice costs as much as the sampling,
            # so copy the file instead of saving it again.
            shutil.copyfile(path, save_dir / "density_latest.png")
        if json_handle:
            json_handle.write(frame_json(frame) + "\n")
            json_handle.flush()
        if not args.no_show:
            plt.pause(0.001)
        scored = len(frame.selection.links) + len(frame.selection.dropped)
        print(
            f"[FRAME] seq={frame.seq} "
            f"links={len(frame.selection.links)}/{scored} "
            f"K={frame.selection.k_min}+{frame.selection.alpha} "
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

            frame, last_ab, previous_state = build_frame(
                model, bank, seq, ms, runtime, device, args, rng,
                last_ab, previous_state,
            )
            if frame is None:
                continue

            last_frame = frame
            render(frame)
            rendered += 1

            if not args.no_show and not plt.fignum_exists(figure.number):
                break

        if args.mode == "last" and not streaming and scans:
            frame, last_ab, previous_state = build_frame(
                model, bank, seq, ms, runtime, device, args, rng,
                last_ab, previous_state,
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
        print(
            "[DONE] nothing rendered: no scan had enough links "
            "(try --min-samples 1 --max-age-scans 40)",
            file=sys.stderr,
        )
        return 1

    if not args.no_show:
        plt.ioff()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
