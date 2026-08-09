#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from gat_distance_filter import (
    DistanceFilterPipeline,
    EdgeGATModel,
    LinkFilterConfig,
    PAIR_LABELS,
    build_features,
    parse_uwb_line,
    rule_scores,
)


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "nan":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def parse_finite_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


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


def choose_topk(scores: List[float], distances: Dict[str, Optional[float]], topk: int):
    valid = [idx for idx, label in enumerate(PAIR_LABELS) if distances[label] is not None]
    return set(sorted(valid, key=lambda idx: scores[idx], reverse=True)[:topk])


def labels_from_row(row: dict, distances: Dict[str, Optional[float]],
                     confidences: Dict[str, float], args) -> List[int]:
    has_keep = any(f"keep_{label}" in row for label in PAIR_LABELS)
    has_true = any(f"true_{label}" in row for label in PAIR_LABELS)

    labels: List[int] = []
    if has_keep:
        for label in PAIR_LABELS:
            value = row.get(f"keep_{label}", "0").strip().lower()
            labels.append(1 if value in ("1", "true", "yes", "y") else 0)
        return labels

    if has_true:
        for label in PAIR_LABELS:
            measured = distances[label]
            truth = parse_float(row.get(f"true_{label}"))
            if measured is None or truth is None:
                labels.append(0)
            else:
                error_ok = abs(measured - truth) <= args.error_threshold
                range_ok = measured <= args.max_keep_distance
                confidence_ok = confidences.get(label, 0.0) >= args.min_label_confidence
                labels.append(1 if error_ok and range_ok and confidence_ok else 0)
        return labels

    if not args.pseudo_labels:
        raise ValueError("No keep_* or true_* labels found. Use --pseudo-labels for a temporary model.")

    scores = rule_scores(distances, args.max_distance, confidences)
    top = choose_topk(scores, distances, args.topk)
    for idx, label in enumerate(PAIR_LABELS):
        keep = distances[label] is not None and (scores[idx] >= args.threshold or idx in top)
        labels.append(1 if keep else 0)
    return labels


def row_to_sample(row: dict, args, pipeline: Optional[DistanceFilterPipeline] = None):
    if "raw_line" in row and row["raw_line"].strip():
        parsed = parse_uwb_line(row["raw_line"])
        if parsed is None:
            return None
        if pipeline is not None:
            filtered = pipeline.process(parsed)
            distances = filtered["distances"]
            confidences = filtered["confidences"]
            rx_power_dbm = filtered["rx_power_dbm"]
            fp_power_dbm = filtered["fp_power_dbm"]
            rssi_fp_diff_db = filtered["rssi_fp_diff_db"]
        else:
            distances = parsed["distances"]
            confidences = {
                label: 1.0 if parsed["distances"][label] is not None else 0.0
                for label in PAIR_LABELS
            }
            rx_power_dbm = parsed["rx_power_dbm"]
            fp_power_dbm = parsed["fp_power_dbm"]
            rssi_fp_diff_db = parsed["rssi_fp_diff_db"]
    else:
        distances = {label: parse_float(row.get(label)) for label in PAIR_LABELS}
        confidences = {
            label: parse_float(row.get(f"conf_{label}")) or
            (1.0 if distances[label] is not None else 0.0)
            for label in PAIR_LABELS
        }
        rx_power_dbm = {
            label: parse_finite_float(row.get(f"rssi_{label}"))
            for label in PAIR_LABELS
        }
        fp_power_dbm = {
            label: parse_finite_float(row.get(f"fp_{label}"))
            for label in PAIR_LABELS
        }
        rssi_fp_diff_db = {
            label: parse_finite_float(row.get(f"rssi_fp_diff_{label}"))
            for label in PAIR_LABELS
        }
        for label in PAIR_LABELS:
            if (rssi_fp_diff_db[label] is None and
                    rx_power_dbm[label] is not None and
                    fp_power_dbm[label] is not None):
                rssi_fp_diff_db[label] = (
                    rx_power_dbm[label] - fp_power_dbm[label]
                )

    labels = labels_from_row(row, distances, confidences, args)
    graph = build_features(
        distances, args.max_distance, confidences,
        rx_power_dbm, fp_power_dbm, rssi_fp_diff_db,
        use_signal_features=not args.disable_signal_features,
    )
    if not graph[-1]:
        return None

    pair_labels = torch.tensor([labels[idx] for idx in graph[-1]], dtype=torch.float32)
    return graph, pair_labels


def load_samples(path: str, args):
    samples = []
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                sample = row_to_sample(row, args)
                if sample is not None:
                    samples.append(sample)
    else:
        pipeline = DistanceFilterPipeline(make_filter_config(args))
        with open(path, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                line = line.strip()
                if not line.startswith("Original_DIST,"):
                    continue
                sample = row_to_sample({"raw_line": line}, args, pipeline)
                if sample is not None:
                    samples.append(sample)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV file or raw Original_DIST log file.")
    parser.add_argument("--output", default="gat_edge_filter.pt")
    parser.add_argument("--pretrained", help="Existing .pt model to fine-tune.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-distance", type=float, default=0.05)
    parser.add_argument("--max-distance", type=float, default=30.0)
    parser.add_argument("--max-keep-distance", type=float, default=20.0)
    parser.add_argument("--error-threshold", type=float, default=1.0)
    parser.add_argument("--min-label-confidence", type=float, default=0.35)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--stale-ms", type=float, default=1000.0)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-rate", type=float, default=3.0)
    parser.add_argument("--jump-soft-limit", type=float, default=2.0)
    parser.add_argument("--stable-mad-limit", type=float, default=0.8)
    parser.add_argument("--pseudo-labels", action="store_true")
    parser.add_argument(
        "--disable-signal-features", action="store_true",
        help="Zero RSSI/FPP/difference inputs for a distance-only ablation baseline.",
    )
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Number of graphs accumulated per optimizer step.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25,
                         help="Early-stop after this many epochs without val_f1 improvement.")
    parser.add_argument("--grad-clip", type=float, default=2.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = load_samples(args.input, args)
    if not samples:
        raise SystemExit("No usable training samples found.")

    random.shuffle(samples)
    split = max(1, int(len(samples) * 0.8))
    train_samples = samples[:split]
    valid_samples = samples[split:] or samples[:]

    # Class-imbalance correction: kept edges are typically a minority.
    pos = sum(float(t.sum().item()) for _, t in train_samples)
    neg = sum(float(t.numel() - t.sum().item()) for _, t in train_samples)
    pos_weight = torch.tensor([max(neg, 1.0) / max(pos, 1.0)], dtype=torch.float32)
    print(f"pos={pos:.0f},neg={neg:.0f},pos_weight={pos_weight.item():.3f}")

    model = EdgeGATModel()
    if args.pretrained:
        state = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(state)
        print(f"loaded_pretrained={os.path.abspath(args.pretrained)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state = None
    epochs_since_best = 0
    batch_size = max(1, args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_samples)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(train_samples), batch_size):
            batch = train_samples[start:start + batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_in_batch = 0
            for graph, target in batch:
                x, edge_index, edge_attr, pair_edge_index, pair_edge_attr, _ = graph
                if target.numel() == 0:
                    continue
                logits = model(x, edge_index, edge_attr, pair_edge_index, pair_edge_attr)
                loss = loss_fn(logits, target)
                batch_loss = batch_loss + loss
                valid_in_batch += 1
            if valid_in_batch == 0:
                continue
            batch_loss = batch_loss / valid_in_batch
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(batch_loss.item())
            num_batches += 1

        model.eval()
        correct = 0
        total = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        with torch.no_grad():
            for graph, target in valid_samples:
                x, edge_index, edge_attr, pair_edge_index, pair_edge_attr, _ = graph
                if target.numel() == 0:
                    continue
                pred = torch.sigmoid(model(x, edge_index, edge_attr, pair_edge_index, pair_edge_attr))
                predicted = pred >= 0.5
                actual = target >= 0.5
                correct += int((predicted == actual).sum().item())
                true_positive += int((predicted & actual).sum().item())
                false_positive += int((predicted & ~actual).sum().item())
                false_negative += int((~predicted & actual).sum().item())
                total += int(target.numel())

        acc = correct / total if total else 0.0
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step(f1)

        print(
            f"epoch={epoch},loss={avg_loss:.4f},"
            f"val_acc={acc:.3f},val_precision={precision:.3f},"
            f"val_recall={recall:.3f},val_f1={f1:.3f},"
            f"lr={optimizer.param_groups[0]['lr']:.5f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"early_stop_at_epoch={epoch},best_f1={best_f1:.3f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), args.output)
    print(f"saved={os.path.abspath(args.output)},samples={len(samples)},best_val_f1={best_f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
