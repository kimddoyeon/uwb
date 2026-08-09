#!/usr/bin/env python3
import argparse
import math
import os
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple


PAIR_LABELS = [
    "AB", "BC", "BD", "BE", "BF", "CD", "DE", "EF",
    "CE", "CF", "DF", "AC", "AD", "AE", "AF",
]

# position_density.py needs the A-B baseline and both A/B links for each
# firefighter in the same output row.  Keep these valid links available for
# probability weighting even when threshold/top-k edge selection ranks one of
# them (most commonly the final AF link) below the cutoff.
POSITION_DENSITY_PAIR_LABELS = {
    "AB", "AC", "BC", "AD", "BD", "AE", "BE", "AF", "BF",
}
POSITION_DENSITY_PAIR_MASK = sum(
    1 << idx
    for idx, label in enumerate(PAIR_LABELS)
    if label in POSITION_DENSITY_PAIR_LABELS
)

NODE_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
FIREFIGHTER_TAG = {"A": 0.0, "B": 0.0, "C": 0.25, "D": 0.50, "E": 0.75, "F": 1.00}

NODE_FEATURE_DIM = 7
EDGE_FEATURE_DIM = 8


@dataclass
class LinkFilterConfig:
    min_distance: float = 0.05
    max_distance: float = 30.0
    stale_seconds: float = 1.0
    median_window: int = 5
    ema_alpha: float = 0.35
    max_rate_mps: float = 3.0
    jump_soft_limit: float = 2.0
    stable_mad_limit: float = 0.8


def parse_uwb_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line.startswith("Original_DIST,"):
        return None

    fields = {}
    for token in line.split(",")[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value

    try:
        fresh_mask = int(fields.get("fresh_mask", "0"), 16)
    except ValueError:
        fresh_mask = 0

    # Provenance only: deterministic geometry recovery has already happened
    # in the collector firmware and this mask is never a GAT feature.
    try:
        recovered_mask = int(fields.get("recovered_mask", "0"), 16)
    except ValueError:
        recovered_mask = 0

    distances: Dict[str, Optional[float]] = {}
    rx_power_dbm: Dict[str, Optional[float]] = {}
    fp_power_dbm: Dict[str, Optional[float]] = {}
    rssi_fp_diff_db: Dict[str, Optional[float]] = {}
    for idx, label in enumerate(PAIR_LABELS):
        value = fields.get(label, "nan")
        is_fresh = bool(fresh_mask & (1 << idx))
        try:
            distance = float(value)
        except ValueError:
            distance = math.nan

        if not is_fresh or not math.isfinite(distance) or distance <= 0.0:
            distances[label] = None
        else:
            distances[label] = distance

        try:
            power = float(fields.get(f"rssi_{label}", "nan"))
        except ValueError:
            power = math.nan
        rx_power_dbm[label] = power if is_fresh and math.isfinite(power) else None

        try:
            fp_power = float(fields.get(f"fp_{label}", "nan"))
        except ValueError:
            fp_power = math.nan
        fp_power_dbm[label] = (
            fp_power if is_fresh and math.isfinite(fp_power) else None
        )

        try:
            power_diff = float(fields.get(f"rssi_fp_diff_{label}", "nan"))
        except ValueError:
            power_diff = math.nan
        if not math.isfinite(power_diff) and math.isfinite(power) and math.isfinite(fp_power):
            power_diff = power - fp_power
        rssi_fp_diff_db[label] = (
            power_diff if is_fresh and math.isfinite(power_diff) else None
        )

    try:
        ms = float(fields.get("ms", "0"))
    except ValueError:
        ms = 0.0

    return {
        "seq": fields.get("seq", "0"),
        "ms": fields.get("ms", "0"),
        "ms_float": ms,
        "recovered_mask": recovered_mask,
        "distances": distances,
        "rx_power_dbm": rx_power_dbm,
        "fp_power_dbm": fp_power_dbm,
        "rssi_fp_diff_db": rssi_fp_diff_db,
    }


class LinkDistanceFilter:
    def __init__(self, config: LinkFilterConfig):
        self.config = config
        self.raw_history: Deque[float] = deque(maxlen=max(1, config.median_window))
        self.filtered: Optional[float] = None
        self.last_update_s: Optional[float] = None
        self.last_output_s: Optional[float] = None
        self.last_jump_m: float = 0.0
        self.last_mad_m: float = 0.0

    def _valid_raw(self, value: Optional[float]) -> bool:
        if value is None or not math.isfinite(value):
            return False
        return self.config.min_distance <= value <= self.config.max_distance

    def _history_median_and_mad(self) -> Tuple[float, float]:
        median = float(statistics.median(self.raw_history))
        deviations = [abs(value - median) for value in self.raw_history]
        mad = float(statistics.median(deviations)) if deviations else 0.0
        return median, mad

    def update(self, raw_value: Optional[float], now_s: float) -> Tuple[Optional[float], float]:
        if self._valid_raw(raw_value):
            raw = float(raw_value)
            previous = self.filtered
            self.raw_history.append(raw)
            median_value, self.last_mad_m = self._history_median_and_mad()

            if self.filtered is None:
                candidate = median_value
            else:
                candidate = (
                    self.config.ema_alpha * median_value +
                    (1.0 - self.config.ema_alpha) * self.filtered
                )

            if previous is not None and self.last_output_s is not None:
                dt = max(now_s - self.last_output_s, 0.001)
                max_delta = max(self.config.max_rate_mps * dt, 0.001)
                delta = candidate - previous
                if abs(delta) > max_delta:
                    candidate = previous + math.copysign(max_delta, delta)
                self.last_jump_m = abs(raw - previous)
            else:
                self.last_jump_m = 0.0

            self.filtered = candidate
            self.last_update_s = now_s
            self.last_output_s = now_s

        return self.current(now_s)

    def current(self, now_s: float) -> Tuple[Optional[float], float]:
        if self.filtered is None or self.last_update_s is None:
            return None, 0.0

        age_s = max(0.0, now_s - self.last_update_s)
        if age_s > self.config.stale_seconds:
            return None, 0.0

        freshness = max(0.0, 1.0 - age_s / max(self.config.stale_seconds, 0.001))
        distance_score = max(0.15, 1.0 - 0.35 * (self.filtered / self.config.max_distance))
        jump_score = max(0.0, 1.0 - self.last_jump_m / max(self.config.jump_soft_limit, 0.001))
        stability_score = max(0.0, 1.0 - self.last_mad_m / max(self.config.stable_mad_limit, 0.001))

        confidence = (
            0.45 * freshness +
            0.20 * distance_score +
            0.20 * stability_score +
            0.15 * jump_score
        )
        return self.filtered, max(0.0, min(1.0, confidence))


class DistanceFilterPipeline:
    def __init__(self, config: LinkFilterConfig):
        self.config = config
        self.links = {label: LinkDistanceFilter(config) for label in PAIR_LABELS}
        self.last_esp_ms: Optional[float] = None
        self.last_local_s: Optional[float] = None

    def _timestamp_s(self, parsed: dict, fallback_s: Optional[float]) -> float:
        esp_ms = float(parsed.get("ms_float", 0.0))
        if esp_ms > 0.0:
            if self.last_esp_ms is not None and esp_ms < self.last_esp_ms:
                self.last_local_s = None
            self.last_esp_ms = esp_ms
            return esp_ms / 1000.0

        if fallback_s is not None:
            return fallback_s
        return time.monotonic()

    def process(self, parsed: dict, now_s: Optional[float] = None) -> dict:
        sample_s = self._timestamp_s(parsed, now_s)
        distances: Dict[str, Optional[float]] = {}
        confidences: Dict[str, float] = {}

        for label in PAIR_LABELS:
            value, confidence = self.links[label].update(
                parsed["distances"].get(label), sample_s
            )
            distances[label] = value
            confidences[label] = confidence

        return {
            "seq": parsed.get("seq", "0"),
            "ms": parsed.get("ms", "0"),
            "recovered_mask": parsed.get("recovered_mask", 0),
            "distances": distances,
            "confidences": confidences,
            "rx_power_dbm": parsed.get(
                "rx_power_dbm", {label: None for label in PAIR_LABELS}
            ),
            "fp_power_dbm": parsed.get(
                "fp_power_dbm", {label: None for label in PAIR_LABELS}
            ),
            "rssi_fp_diff_db": parsed.get(
                "rssi_fp_diff_db", {label: None for label in PAIR_LABELS}
            ),
        }


def normalize_power_dbm(value: Optional[float]) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, (value + 120.0) / 100.0))


def normalize_power_difference_db(value: Optional[float]) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value / 40.0))


def build_features(
    distances: Dict[str, Optional[float]],
    max_distance: float,
    confidences: Optional[Dict[str, float]] = None,
    rx_power_dbm: Optional[Dict[str, Optional[float]]] = None,
    fp_power_dbm: Optional[Dict[str, Optional[float]]] = None,
    rssi_fp_diff_db: Optional[Dict[str, Optional[float]]] = None,
    use_signal_features: bool = True,
):
    import torch

    num_nodes = 6
    degree = [0.0] * num_nodes
    sums = [0.0] * num_nodes
    weights = [0.0] * num_nodes
    conf_sums = [0.0] * num_nodes
    dist_to_b = [max_distance] * num_nodes

    valid_pairs: List[Tuple[int, str, int, int, float, float]] = []
    for pair_idx, label in enumerate(PAIR_LABELS):
        distance = distances[label]
        confidence = 1.0 if confidences is None else confidences.get(label, 0.0)
        if distance is None or confidence <= 0.0:
            continue
        a, b = label[0], label[1]
        u, v = NODE_INDEX[a], NODE_INDEX[b]
        valid_pairs.append((pair_idx, label, u, v, distance, confidence))
        degree[u] += confidence
        degree[v] += confidence
        sums[u] += distance * confidence
        sums[v] += distance * confidence
        weights[u] += confidence
        weights[v] += confidence
        conf_sums[u] += confidence
        conf_sums[v] += confidence
        if a == "B":
            dist_to_b[v] = distance
        if b == "B":
            dist_to_b[u] = distance

    node_rows = []
    for label, idx in NODE_INDEX.items():
        is_drone = 1.0 if label in ("A", "B") else 0.0
        is_firefighter = 1.0 - is_drone
        mean_dist = sums[idx] / weights[idx] if weights[idx] > 0.0 else max_distance
        mean_conf = conf_sums[idx] / max(degree[idx], 1.0)
        node_rows.append([
            is_drone,
            is_firefighter,
            FIREFIGHTER_TAG[label],
            min(dist_to_b[idx], max_distance) / max_distance,
            min(degree[idx], 5.0) / 5.0,
            min(mean_dist, max_distance) / max_distance,
            max(0.0, min(1.0, mean_conf)),
        ])

    directed_edges = []
    directed_attrs = []
    pair_edges = []
    pair_attrs = []
    pair_indices = []

    for pair_idx, label, u, v, distance, confidence in valid_pairs:
        rssi = rx_power_dbm.get(label) if rx_power_dbm else None
        fp_power = fp_power_dbm.get(label) if fp_power_dbm else None
        power_diff = rssi_fp_diff_db.get(label) if rssi_fp_diff_db else None
        if power_diff is None and rssi is not None and fp_power is not None:
            power_diff = rssi - fp_power
        edge_attr = [
            min(distance, max_distance) / max_distance,
            1.0 / (distance + 1.0),
            1.0 if "B" in label else 0.0,
            1.0 if label[0] not in ("A", "B") and label[1] not in ("A", "B") else 0.0,
            max(0.0, min(1.0, confidence)),
            normalize_power_dbm(rssi) if use_signal_features else 0.0,
            normalize_power_dbm(fp_power) if use_signal_features else 0.0,
            normalize_power_difference_db(power_diff) if use_signal_features else 0.0,
        ]
        directed_edges.append((u, v))
        directed_edges.append((v, u))
        directed_attrs.append(edge_attr)
        directed_attrs.append(edge_attr)
        pair_edges.append((u, v))
        pair_attrs.append(edge_attr)
        pair_indices.append(pair_idx)

    x = torch.tensor(node_rows, dtype=torch.float32)
    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(directed_attrs, dtype=torch.float32)
        pair_edge_index = torch.tensor(pair_edges, dtype=torch.long)
        pair_edge_attr = torch.tensor(pair_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)
        pair_edge_index = torch.empty((0, 2), dtype=torch.long)
        pair_edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    return x, edge_index, edge_attr, pair_edge_index, pair_edge_attr, pair_indices


class EdgeGATModel:
    def __new__(cls):
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = 32
                self.node_proj = nn.Linear(NODE_FEATURE_DIM, hidden)
                self.edge_proj = nn.Linear(EDGE_FEATURE_DIM, hidden)
                self.att_src = nn.Linear(hidden, 1, bias=False)
                self.att_dst = nn.Linear(hidden, 1, bias=False)
                self.att_edge = nn.Linear(hidden, 1, bias=False)
                self.out_proj = nn.Linear(hidden * 2, hidden)
                self.edge_head = nn.Sequential(
                    nn.Linear(hidden * 2 + EDGE_FEATURE_DIM, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x, edge_index, edge_attr, pair_edge_index, pair_edge_attr):
                import torch

                h = F.relu(self.node_proj(x))
                if edge_index.numel() > 0:
                    src, dst = edge_index[0], edge_index[1]
                    eh = F.relu(self.edge_proj(edge_attr))
                    score = F.leaky_relu(
                        self.att_src(h[src]) + self.att_dst(h[dst]) + self.att_edge(eh),
                        negative_slope=0.2,
                    ).squeeze(-1)
                    alpha = torch.zeros_like(score)
                    for node in range(x.shape[0]):
                        mask = dst == node
                        if bool(mask.any()):
                            alpha[mask] = torch.softmax(score[mask], dim=0)
                    msg = h[src] * alpha.unsqueeze(-1)
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst, msg)
                    h = F.relu(self.out_proj(torch.cat([h, agg], dim=-1)))

                if pair_edge_index.numel() == 0:
                    return torch.empty((0,), dtype=torch.float32, device=x.device)

                u = pair_edge_index[:, 0]
                v = pair_edge_index[:, 1]
                logits = self.edge_head(torch.cat([h[u], h[v], pair_edge_attr], dim=-1))
                return logits.squeeze(-1)

        return Model()


class EdgeGATFilter:
    def __init__(self, model_path: str, max_distance: float):
        import torch

        self.torch = torch
        self.max_distance = max_distance
        self.model = EdgeGATModel()
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def score(
        self,
        distances: Dict[str, Optional[float]],
        confidences: Optional[Dict[str, float]] = None,
        rx_power_dbm: Optional[Dict[str, Optional[float]]] = None,
        fp_power_dbm: Optional[Dict[str, Optional[float]]] = None,
        rssi_fp_diff_db: Optional[Dict[str, Optional[float]]] = None,
    ) -> List[float]:
        torch = self.torch
        graph = build_features(
            distances, self.max_distance, confidences,
            rx_power_dbm, fp_power_dbm, rssi_fp_diff_db,
        )
        x, edge_index, edge_attr, pair_edge_index, pair_edge_attr, pair_indices = graph
        probs = [0.0] * len(PAIR_LABELS)
        with torch.no_grad():
            logits = self.model(x, edge_index, edge_attr, pair_edge_index, pair_edge_attr)
            values = torch.sigmoid(logits).cpu().tolist()
        for pair_idx, value in zip(pair_indices, values):
            probs[pair_idx] = float(value)
        return probs


def rule_scores(
    distances: Dict[str, Optional[float]],
    max_distance: float,
    confidences: Optional[Dict[str, float]] = None,
) -> List[float]:
    scores = []
    for label in PAIR_LABELS:
        distance = distances[label]
        confidence = 1.0 if confidences is None else confidences.get(label, 0.0)
        if distance is None or confidence <= 0.0:
            scores.append(0.0)
            continue
        distance_score = math.exp(-distance / max_distance)
        score = 0.70 * distance_score + 0.30 * confidence
        if "B" in label:
            score += 0.05
        scores.append(max(0.0, min(1.0, score)))
    return scores


def choose_edges(
    scores: List[float],
    distances: Dict[str, Optional[float]],
    threshold: float,
    topk: int,
    force_keep_mask: int = 0,
) -> Tuple[int, List[Optional[float]]]:
    valid_indices = [
        i for i, label in enumerate(PAIR_LABELS)
        if distances[label] is not None
    ]
    ranked = sorted(valid_indices, key=lambda idx: scores[idx], reverse=True)
    keep = {idx for idx in valid_indices if scores[idx] >= threshold}
    keep.update(ranked[:max(0, topk)])
    # Geometry recovery is deterministic and happened before GAT. Preserve
    # those valid links without changing their model/rule probability scores.
    keep.update(
        idx for idx in valid_indices if force_keep_mask & (1 << idx)
    )

    keep_mask = 0
    selected: List[Optional[float]] = []
    for idx, label in enumerate(PAIR_LABELS):
        if idx in keep:
            keep_mask |= 1 << idx
            selected.append(distances[label])
        else:
            selected.append(None)
    return keep_mask, selected


def format_distance_output(prefix: str, parsed: dict,
                           distances: Dict[str, Optional[float]],
                           confidences: Dict[str, float]) -> str:
    parts = [
        prefix,
        f"seq={parsed['seq']}",
        f"ms={parsed['ms']}",
        f"recovered_mask=0x{int(parsed.get('recovered_mask', 0)):X}",
    ]
    for label in PAIR_LABELS:
        value = distances[label]
        parts.append(f"{label}={value:.3f}" if value is not None else f"{label}=nan")
    conf_text = "|".join(f"{label}:{confidences[label]:.2f}" for label in PAIR_LABELS)
    parts.append(f"conf={conf_text}")
    powers = parsed.get("rx_power_dbm", {})
    power_text = "|".join(
        f"{label}:{powers.get(label):.2f}"
        if powers.get(label) is not None else f"{label}:nan"
        for label in PAIR_LABELS
    )
    parts.append(f"rssi_dbm={power_text}")
    fp_powers = parsed.get("fp_power_dbm", {})
    parts.append("fp_power_dbm=" + "|".join(
        f"{label}:{fp_powers.get(label):.2f}"
        if fp_powers.get(label) is not None else f"{label}:nan"
        for label in PAIR_LABELS
    ))
    differences = parsed.get("rssi_fp_diff_db", {})
    parts.append("rssi_fp_diff_db=" + "|".join(
        f"{label}:{differences.get(label):.2f}"
        if differences.get(label) is not None else f"{label}:nan"
        for label in PAIR_LABELS
    ))
    return ",".join(parts)


def format_gat_output(prefix: str, parsed: dict, keep_mask: int,
                      selected: List[Optional[float]], scores: List[float],
                      confidences: Dict[str, float]) -> str:
    parts = [
        prefix,
        f"seq={parsed['seq']}",
        f"ms={parsed['ms']}",
        f"keep_mask=0x{keep_mask:X}",
        f"recovered_mask=0x{int(parsed.get('recovered_mask', 0)):X}",
    ]
    kept_indices = [idx for idx, value in enumerate(selected) if value is not None]
    kept_indices.sort(key=lambda idx: scores[idx], reverse=True)

    for idx in kept_indices:
        parts.append(f"{PAIR_LABELS[idx]}={selected[idx]:.3f}")

    prob_text = "|".join(
        f"{PAIR_LABELS[idx]}:{scores[idx]:.2f}" for idx in kept_indices
    )
    conf_text = "|".join(
        f"{PAIR_LABELS[idx]}:{confidences[PAIR_LABELS[idx]]:.2f}"
        for idx in kept_indices
    )
    parts.append(f"prob={prob_text}")
    parts.append(f"conf={conf_text}")
    return ",".join(parts)


def iter_input_lines(args) -> Iterable[str]:
    if args.input_log:
        with open(args.input_log, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                yield line
        return

    try:
        import serial
    except ImportError:
        print("Install pyserial first: pip3 install pyserial", file=sys.stderr)
        raise

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        for raw in ser:
            yield raw.decode("utf-8", errors="ignore")


def make_filter_config(args) -> LinkFilterConfig:
    return LinkFilterConfig(
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        stale_seconds=args.stale_ms / 1000.0,
        median_window=args.median_window,
        ema_alpha=args.ema_alpha,
        max_rate_mps=args.max_rate,
        jump_soft_limit=args.jump_soft_limit,
        stable_mad_limit=args.stable_mad_limit,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--input-log")
    parser.add_argument("--model", default="gat_edge_filter.pt")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-distance", type=float, default=0.05)
    parser.add_argument("--max-distance", type=float, default=30.0)
    parser.add_argument("--stale-ms", type=float, default=1000.0)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-rate", type=float, default=3.0)
    parser.add_argument("--jump-soft-limit", type=float, default=2.0)
    parser.add_argument("--stable-mad-limit", type=float, default=0.8)
    parser.add_argument("--print-filtered", action="store_true")
    args = parser.parse_args()

    filter_pipeline = DistanceFilterPipeline(make_filter_config(args))
    gat = None
    prefix = "RULE_DIST"
    if os.path.exists(args.model):
        try:
            gat = EdgeGATFilter(args.model, args.max_distance)
            prefix = "GAT_DIST"
        except Exception as exc:
            print(f"Could not load GAT model, using rule fallback: {exc}", file=sys.stderr)
    else:
        print(f"No model file found at {args.model}; using rule fallback.", file=sys.stderr)

    for line in iter_input_lines(args):
        parsed = parse_uwb_line(line)
        if parsed is None:
            continue

        filtered = filter_pipeline.process(parsed)
        distances = filtered["distances"]
        confidences = filtered["confidences"]

        if args.print_filtered:
            print(format_distance_output("Filtered_DIST", filtered, distances, confidences), flush=True)

        if gat is None:
            scores = rule_scores(distances, args.max_distance, confidences)
        else:
            scores = gat.score(
                distances, confidences,
                filtered["rx_power_dbm"],
                filtered["fp_power_dbm"],
                filtered["rssi_fp_diff_db"],
            )

        keep_mask, selected = choose_edges(
            scores, distances, args.threshold, args.topk,
            force_keep_mask=(
                int(filtered.get("recovered_mask", 0)) |
                POSITION_DENSITY_PAIR_MASK
            ),
        )
        print(format_gat_output(prefix, filtered, keep_mask, selected, scores, confidences), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
