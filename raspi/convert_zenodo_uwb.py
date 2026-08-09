#!/usr/bin/env python3
"""Convert Zenodo 7629141 raw UWB measurements to this project's CSV format.

The source data describes one mobile tag and up to eight anchors.  This
project's graph has six nodes, so the converter maps the tag to A and five
anchors to B..F.  Each output row is one synchronized measurement index for a
single tag position and UWB channel.  Links not represented by the source
data are left empty and are ignored by the trainer.
"""

import argparse
import csv
import itertools
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, TextIO, Tuple


PAIR_LABELS = [
    "AB", "BC", "BD", "BE", "BF", "CD", "DE", "EF",
    "CE", "CF", "DF", "AC", "AD", "AE", "AF",
]
STAR_LINKS = ["AB", "AC", "AD", "AE", "AF"]
REQUIRED = {"X_TAG", "Y_TAG", "Z_TAG", "X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR", "RANGE"}
FILE_RE = re.compile(r"^(ch[^_]+)_([^_.]+)\.csv$", re.IGNORECASE)


def finite(value: object) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_header(values: Iterable[str]) -> List[str]:
    return [value.strip().upper() for value in values]


def open_source_csv(path: Path) -> Tuple[TextIO, csv.DictReader]:
    """Open a source CSV whose version line may precede the real header."""
    fp = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
    sample = fp.read(16384)
    fp.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(fp, dialect)
    header: Optional[List[str]] = None
    for values in reader:
        normalized = normalize_header(values)
        if "RANGE" in normalized and ("TAG_ID" in normalized or "X_TAG" in normalized):
            header = normalized
            break
    if header is None:
        fp.close()
        raise ValueError(f"Could not find the measurement header in {path}")

    # The published files contain one unnamed heading. Give it a harmless name.
    seen: Dict[str, int] = {}
    unique_header = []
    for index, name in enumerate(header):
        name = name or f"UNNAMED_{index}"
        seen[name] = seen.get(name, 0) + 1
        unique_header.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return fp, csv.DictReader(fp, fieldnames=unique_header, dialect=dialect)


def true_range(row: dict) -> Optional[float]:
    coords = [finite(row.get(name)) for name in (
        "X_TAG", "Y_TAG", "Z_TAG", "X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR"
    )]
    if any(value is None for value in coords):
        return None
    xt, yt, zt, xa, ya, za = coords  # type: ignore[misc]
    return math.sqrt((xt - xa) ** 2 + (yt - ya) ** 2 + (zt - za) ** 2)


def anchor_files(position_dir: Path, requested: Optional[List[str]]) -> Dict[str, List[Path]]:
    by_channel: Dict[str, Dict[str, Path]] = {}
    for path in position_dir.glob("*.csv"):
        match = FILE_RE.match(path.name)
        if match:
            channel, anchor = match.groups()
            by_channel.setdefault(channel.lower(), {})[anchor.upper()] = path

    selected: Dict[str, List[Path]] = {}
    for channel, paths in by_channel.items():
        anchors = requested or sorted(paths, key=lambda value: (len(value), value))[:5]
        present = [paths[anchor] for anchor in anchors if anchor in paths]
        if present:
            selected[channel] = present
    return selected


def output_fields() -> List[str]:
    fields = ["source_location", "source_position", "source_channel", "sample_index"]
    for prefix in ("", "rssi_", "fp_", "rssi_fp_diff_", "true_", "keep_"):
        fields.extend(f"{prefix}{label}" for label in PAIR_LABELS)
    return fields


def convert_group(paths: List[Path], location: str, position: str, channel: str,
                  writer: csv.DictWriter, error_threshold: float,
                  keep_nlos: bool, limit: Optional[int]) -> int:
    opened: List[TextIO] = []
    readers: List[csv.DictReader] = []
    try:
        for path in paths:
            fp, reader = open_source_csv(path)
            opened.append(fp)
            readers.append(reader)

        count = 0
        for sample_index, source_rows in enumerate(itertools.zip_longest(*readers)):
            if limit is not None and count >= limit:
                break
            output = {
                "source_location": location,
                "source_position": position,
                "source_channel": channel,
                "sample_index": sample_index,
            }
            valid_links = 0
            for link, source in zip(STAR_LINKS, source_rows):
                if source is None:
                    continue
                measured = finite(source.get("RANGE"))
                truth = true_range(source)
                if measured is None or measured <= 0:
                    continue
                valid_links += 1
                output[link] = measured
                if truth is not None:
                    output[f"true_{link}"] = truth
                rss = finite(source.get("RSS"))
                rss_fp = finite(source.get("RSS_FP"))
                if rss is not None:
                    output[f"rssi_{link}"] = rss
                if rss_fp is not None:
                    output[f"fp_{link}"] = rss_fp
                if rss is not None and rss_fp is not None:
                    output[f"rssi_fp_diff_{link}"] = rss - rss_fp

                nlos_value = finite(source.get("NLOS"))
                los_ok = keep_nlos or nlos_value is None or nlos_value == 0
                error_ok = truth is not None and abs(measured - truth) <= error_threshold
                output[f"keep_{link}"] = int(los_ok and error_ok)
            if valid_links:
                writer.writerow(output)
                count += 1
        return count
    finally:
        for fp in opened:
            fp.close()


def find_position_dirs(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.csv"):
        if FILE_RE.match(path.name):
            yield path.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Extracted data_set directory, raw_data directory, or one location directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--anchors", help="Five source anchors in B,C,D,E,F order, e.g. A1,A2,A3,A4,A5")
    parser.add_argument("--error-threshold", type=float, default=1.0,
                        help="Maximum absolute range error in metres for keep=1 (default: 1.0)")
    parser.add_argument("--keep-nlos", action="store_true",
                        help="Allow accurate NLOS samples to receive keep=1")
    parser.add_argument("--limit-per-group", type=int,
                        help="Write at most N samples per position/channel (useful for a quick test)")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")
    requested = [value.strip().upper() for value in args.anchors.split(",")] if args.anchors else None
    if requested is not None and len(requested) != 5:
        parser.error("--anchors must contain exactly five comma-separated anchor IDs")

    position_dirs = sorted(set(find_position_dirs(args.input)))
    if not position_dirs:
        parser.error("No raw files like ch1_A1.csv were found below the input directory")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    groups = 0
    with args.output.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=output_fields(), extrasaction="ignore")
        writer.writeheader()
        for position_dir in position_dirs:
            channel_groups = anchor_files(position_dir, requested)
            location = next((part for part in position_dir.parts if part.lower().startswith("location")), "unknown")
            for channel, paths in sorted(channel_groups.items()):
                total += convert_group(paths, location, position_dir.name, channel, writer,
                                       args.error_threshold, args.keep_nlos,
                                       args.limit_per_group)
                groups += 1

    print(f"output={args.output.resolve()}")
    print(f"position_dirs={len(position_dirs)},groups={groups},rows={total}")
    print("mapping=tag:A,anchors:B,C,D,E,F")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
