#!/usr/bin/env python3
"""Monitor UWB serial output and save one timestamped session.

Two files are created for every run:

* ``*_terminal.txt``: every decoded line received from the serial terminal.
* ``*_gat_input.txt``: only ``Original_DIST,...`` lines, ready for GAT input.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Tuple


DEFAULT_SAVE_DIR = Path(__file__).resolve().parents[1] / "data_save"


def session_paths(save_dir: Path) -> Tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

    suffix = ""
    duplicate = 1
    while True:
        base = f"{timestamp}{suffix}"
        terminal_path = save_dir / f"{base}_terminal.txt"
        gat_input_path = save_dir / f"{base}_gat_input.txt"
        if not terminal_path.exists() and not gat_input_path.exists():
            return terminal_path, gat_input_path
        duplicate += 1
        suffix = f"_{duplicate:02d}"


def iter_file_lines(input_path: Path) -> Iterator[str]:
    with input_path.open(encoding="utf-8", errors="ignore") as source:
        for raw_line in source:
            yield raw_line.rstrip("\r\n")


def iter_serial_lines(port: str, baud: int) -> Iterator[str]:
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Install pyserial first: pip install pyserial") from exc

    with serial.Serial(port, baud, timeout=1) as device:
        while True:
            raw_line = device.readline()
            if raw_line:
                yield raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")


def save_monitor_session(lines: Iterable[str], terminal_path: Path,
                         gat_input_path: Path) -> Tuple[int, int]:
    terminal_count = 0
    gat_input_count = 0

    with terminal_path.open("x", encoding="utf-8", buffering=1) as terminal_fp, \
            gat_input_path.open("x", encoding="utf-8", buffering=1) as gat_fp:
        try:
            for line in lines:
                print(line, flush=True)
                terminal_fp.write(line + "\n")
                terminal_count += 1

                if line.startswith("Original_DIST,"):
                    gat_fp.write(line + "\n")
                    gat_input_count += 1
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

    return terminal_count, gat_input_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor UWB data and save terminal/GAT-input logs.")
    parser.add_argument("--port", default="/dev/ttyUSB0",
                        help="Serial port, for example /dev/ttyUSB0 or COM10.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--save-dir", type=Path, default=DEFAULT_SAVE_DIR,
        help=f"Output directory (default: {DEFAULT_SAVE_DIR}).")
    parser.add_argument(
        "--input-log", type=Path,
        help="Read an existing log instead of a serial port (for replay/testing).")
    args = parser.parse_args()

    terminal_path, gat_input_path = session_paths(args.save_dir)
    print(f"terminal_save={terminal_path}")
    print(f"gat_input_save={gat_input_path}")
    print("Press Ctrl+C to stop and close both files safely.")

    lines = (iter_file_lines(args.input_log) if args.input_log
             else iter_serial_lines(args.port, args.baud))

    terminal_count = 0
    gat_input_count = 0
    try:
        terminal_count, gat_input_count = save_monitor_session(
            lines, terminal_path, gat_input_path)
    finally:
        print(f"saved_terminal={terminal_path}")
        print(f"saved_gat_input={gat_input_path}")
        print(f"terminal_lines={terminal_count},gat_input_lines={gat_input_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
