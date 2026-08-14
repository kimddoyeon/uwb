#!/usr/bin/env python3
"""Live DW3000 collector on the Pi -> trained GAT -> position probability.

This is the real-time counterpart of ``uwb_probability_pipeline.py``.  The
scoring, Top-K selection, Metropolis-Hastings sampling and density rendering
are imported from that module unchanged, so a live run and a replay of the same
session produce the same picture.  What this file adds is everything the live
path needs and a log replay does not:

* The serial port is drained by its own thread.  The collector prints one
  ``Original_DIST`` row per scan (~2.5 s in the current firmware) and one frame
  costs the sampler a second or more on a Pi.  If the sampler read the port
  itself, the kernel FIFO would fill while it computes and every frame would be
  built from older and older data.  Here the reader never stops, every scan
  still reaches the rolling windows, and the sampler always starts from the
  newest scan -- lag is reported instead of silently accumulating.
* ``seq`` resets are detected.  Opening the port toggles DTR, which resets the
  ESP32, so the first live ``seq`` is usually far below the last one seen.
  Without this the age check ``seq - last_seq > max_age_scans`` would mark
  every link stale forever and no frame would ever be drawn again.
* The port is auto-detected and re-opened after an unplug, with backoff.
* The session is written to ``data_save/`` in the same two-file format
  ``data_save.py`` uses.  Only one process can hold the serial port, so this
  has to happen here or the live session is not recoverable.

Examples
--------
On the Pi with a screen attached::

    python3 raspi/realtime_uwb_gat.py --port /dev/ttyUSB0

Headless over SSH, PNG refreshed in place, one JSON line per frame::

    python3 raspi/realtime_uwb_gat.py --no-show --json-out - --fast

Numbers only, no matplotlib at all::

    python3 raspi/realtime_uwb_gat.py --no-plot

Which port is the collector on::

    python3 raspi/realtime_uwb_gat.py --list-ports
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from gat_distance_filter import parse_uwb_line
from uwb_probability_pipeline import (
    Frame,
    WindowBank,
    build_frame,
    describe_checkpoint,
    frame_json,
    load_checkpoint,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SAVE_DIR = REPO_ROOT / "data_save"

# USB-UART bridges used on the ESP32 boards this project flashes, plus
# Espressif's own native-USB VID.
KNOWN_USB_VIDS = {
    0x10C4,  # Silicon Labs CP210x
    0x1A86,  # WCH CH340 / CH9102
    0x0403,  # FTDI
    0x303A,  # Espressif native USB-CDC
}


# ============================================================
# Serial transport
# ============================================================
@dataclass
class SourceStats:
    lines: int = 0
    scans: int = 0
    dropped: int = 0
    reconnects: int = 0
    opened_s: float = 0.0
    last_line_s: float = 0.0
    last_scan_s: float = 0.0
    error: Optional[str] = None


def list_serial_ports() -> List[object]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return list(list_ports.comports())


def autodetect_port() -> Optional[str]:
    """Best guess at the collector's port, preferring known ESP32 bridges."""
    ports = list_serial_ports()
    if not ports:
        return None

    def rank(port) -> Tuple[int, str]:
        vid = getattr(port, "vid", None)
        device = str(port.device)
        if vid in KNOWN_USB_VIDS:
            return (0, device)
        if "ttyUSB" in device or "ttyACM" in device or device.startswith("COM"):
            return (1, device)
        return (2, device)

    return str(sorted(ports, key=rank)[0].device)


class SessionWriter:
    """Same two files ``data_save.py`` writes, so a live run stays replayable."""

    def __init__(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        suffix = ""
        duplicate = 1
        while True:
            base = f"{stamp}{suffix}"
            terminal = save_dir / f"{base}_terminal.txt"
            gat_input = save_dir / f"{base}_gat_input.txt"
            if not terminal.exists() and not gat_input.exists():
                break
            duplicate += 1
            suffix = f"_{duplicate:02d}"

        self.terminal_path = terminal
        self.gat_input_path = gat_input
        self._terminal = terminal.open("x", encoding="utf-8", buffering=1)
        self._gat_input = gat_input.open("x", encoding="utf-8", buffering=1)

    def write(self, line: str) -> None:
        self._terminal.write(line + "\n")
        if line.startswith("Original_DIST,"):
            self._gat_input.write(line + "\n")

    def close(self) -> None:
        for handle in (self._terminal, self._gat_input):
            try:
                handle.close()
            except Exception:
                pass


class SerialSource:
    """Reader thread: decode lines, park parsed scans in a bounded queue.

    The queue is bounded because falling behind must cost the *oldest* scans,
    never memory: a scan that is 30 s old contributes nothing to a rolling
    window the sampler is about to rebuild anyway.
    """

    def __init__(
        self,
        port: str,
        baud: int,
        raw_sink: Optional[Callable[[str], None]] = None,
        assert_reset: bool = True,
        queue_size: int = 256,
        echo: bool = False,
    ):
        self.port = port
        self.baud = baud
        self.raw_sink = raw_sink
        self.assert_reset = assert_reset
        self.echo = echo

        self.queue: "queue.Queue[dict]" = queue.Queue(maxsize=queue_size)
        self.stats = SourceStats()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="uwb-serial", daemon=True
        )

    # -- lifecycle -------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    # -- reader ----------------------------------------------------
    def _open(self):
        import serial

        if self.assert_reset:
            return serial.Serial(self.port, self.baud, timeout=1.0)

        # Keep DTR/RTS low so opening the port does not drive the ESP32's
        # auto-reset circuit and the collector keeps its scan counter.
        device = serial.Serial()
        device.port = self.port
        device.baudrate = self.baud
        device.timeout = 1.0
        try:
            device.dtr = False
            device.rts = False
        except (OSError, AttributeError):
            pass  # not every driver exposes the modem lines before open
        device.open()
        return device

    def _publish(self, parsed: dict) -> None:
        while True:
            try:
                self.queue.put_nowait(parsed)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.stats.dropped += 1
                except queue.Empty:
                    return

    def _run(self) -> None:
        # A thread that dies silently would leave the main loop waiting for
        # scans that can never arrive, so record why and let it notice.
        try:
            self._reader_loop()
        except BaseException as exc:
            self.stats.error = f"{type(exc).__name__}: {exc}"

    def _reader_loop(self) -> None:
        import serial

        backoff = 0.5
        while not self._stop.is_set():
            device = None
            try:
                device = self._open()
                self.stats.opened_s = time.monotonic()
                self.stats.error = None
                backoff = 0.5
                print(f"[SERIAL] open {self.port} @ {self.baud}", flush=True)

                while not self._stop.is_set():
                    raw = device.readline()
                    if not raw:
                        continue  # 1 s read timeout, not an error

                    line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
                    if not line:
                        continue

                    self.stats.lines += 1
                    self.stats.last_line_s = time.monotonic()
                    if self.echo:
                        print(line, flush=True)
                    if self.raw_sink is not None:
                        self.raw_sink(line)

                    parsed = parse_uwb_line(line)
                    if parsed is None:
                        continue

                    self.stats.scans += 1
                    self.stats.last_scan_s = time.monotonic()
                    self._publish(parsed)

            except serial.SerialException as exc:
                self.stats.error = str(exc)
            except OSError as exc:  # unplugged mid-read
                self.stats.error = str(exc)
            finally:
                if device is not None:
                    try:
                        device.close()
                    except Exception:
                        pass

            if self._stop.is_set():
                break

            self.stats.reconnects += 1
            print(
                f"[SERIAL] lost {self.port} ({self.stats.error}); "
                f"retry in {backoff:.1f} s",
                file=sys.stderr,
                flush=True,
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2.0, 5.0)

    # -- consumer --------------------------------------------------
    def drain(self, timeout: float) -> List[dict]:
        """Block up to ``timeout`` for one scan, then take everything queued."""
        batch: List[dict] = []
        try:
            batch.append(self.queue.get(timeout=timeout))
        except queue.Empty:
            return batch

        while True:
            try:
                batch.append(self.queue.get_nowait())
            except queue.Empty:
                return batch


# ============================================================
# Live state
# ============================================================
# ``last_completed_cycle_seq`` is a uint8_t in both uwb.cpp and uwb1.cpp, so the
# scan counter on the wire wraps 255 -> 0 every 256 scans, about every 11
# minutes at the current scan rate.  A replayed log never reaches that point,
# which is why only the live path has to care.
SEQ_MODULUS = 256


@dataclass
class LiveState:
    """Everything the sampler carries from one frame to the next."""

    window_size: int
    bank: WindowBank = field(init=False)
    #: 8-bit value last seen on the wire
    last_raw_seq: Optional[int] = None
    #: unwrapped, strictly increasing scan counter handed to the sampler
    seq: int = 0
    last_ab: Optional[float] = None
    previous_state: Optional[np.ndarray] = None
    frames: int = 0

    def __post_init__(self) -> None:
        self.bank = WindowBank(self.window_size)

    def reset(self, reason: str) -> None:
        print(f"[RESET] {reason}", flush=True)
        self.bank = WindowBank(self.window_size)
        self.last_ab = None
        self.previous_state = None

    def ingest(self, parsed: dict, max_forward_gap: int) -> Tuple[int, int]:
        """Feed one scan into the rolling windows on an unwrapped counter.

        ``WindowBank.active_links`` ages a link with ``seq - last_seq``, which
        the raw 8-bit counter breaks twice over: right after a wrap every stored
        link has a *negative* age, so a link that stopped reporting stays
        "fresh" for a whole 256-scan cycle, and a naive "seq went backwards"
        restart test would instead throw away every window once every 11
        minutes.  Accumulating the modular difference fixes both: the counter
        keeps rising across a wrap and ages stay true.

        A jump larger than ``max_forward_gap`` is the one case this cannot
        reconcile -- it is either a collector restart or a dropout longer than
        the gap -- and the windows are dropped because their contents can no
        longer be placed on the timeline.
        """
        raw = int(float(parsed.get("seq", 0))) % SEQ_MODULUS

        if self.last_raw_seq is None:
            self.seq = raw
        else:
            delta = (raw - self.last_raw_seq) % SEQ_MODULUS
            if delta > max_forward_gap:
                self.reset(
                    f"scan counter {self.last_raw_seq} -> {raw} "
                    f"(+{delta} > {max_forward_gap}: restart or long dropout)"
                )
                self.seq += 1
            else:
                # delta == 0 repeats a scan the collector already published.
                self.seq += delta

        self.last_raw_seq = raw

        # WindowBank reads the counter out of the row itself, so hand it the
        # unwrapped one without touching the caller's dict.
        self.bank.update({**parsed, "seq": self.seq})
        return self.seq, raw


def warmup_report(state: LiveState, min_samples: int) -> str:
    ready = sum(
        1 for window in state.bank.windows.values() if len(window) >= min_samples
    )
    seen = sum(1 for window in state.bank.windows.values() if len(window) > 0)
    return f"links seen={seen} ready={ready} (need >={min_samples} samples)"


def format_peaks(frame: Frame) -> str:
    sign = "+/-" if frame.mirrored else ""
    parts = []
    for person in frame.targets:
        peak = frame.peaks.get(person)
        if peak is None:
            parts.append(f"{person}=-")
        else:
            parts.append(f"{person}=({peak[0]:5.2f},{sign}{abs(peak[1]):4.2f})")
    return "  ".join(parts)


# ============================================================
# Output
# ============================================================
class Renderer:
    """Live window, atomic PNG, or nothing at all."""

    def __init__(self, args):
        self.enabled = not args.no_plot
        self.show = self.enabled and not args.no_show
        self.save_dir = Path(args.save_dir) if args.save_dir else None
        self.save_frames = args.save_frames
        self.dpi = args.dpi
        self.every = max(1, args.render_every)
        self.figure = None
        self.closed = False
        self._draw_frame = None

        if not self.enabled:
            return

        import matplotlib

        if not self.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from uwb_probability_pipeline import draw_frame

        self._plt = plt
        self._draw_frame = draw_frame
        self.figure = plt.figure("UWB live probability", figsize=(15.5, 9.0))
        if self.show:
            plt.ion()
            plt.show(block=False)

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def render(self, frame: Frame, index: int) -> None:
        if not self.enabled or index % self.every:
            return

        self._draw_frame(self.figure, frame)

        if self.save_dir:
            # Write then rename: a viewer polling density_latest.png must never
            # get a half-written file.
            staging = self.save_dir / ".density_latest.tmp.png"
            self.figure.savefig(staging, dpi=self.dpi)
            os.replace(staging, self.save_dir / "density_latest.png")
            if self.save_frames:
                self.figure.savefig(
                    self.save_dir / f"density_seq{frame.seq:05d}.png", dpi=self.dpi
                )

        if self.show:
            self._plt.pause(0.001)
            if not self._plt.fignum_exists(self.figure.number):
                self.closed = True

    def keep_alive(self) -> None:
        """Service the GUI event loop while waiting for the next scan."""
        if self.show and not self.closed:
            self._plt.pause(0.01)
            if not self._plt.fignum_exists(self.figure.number):
                self.closed = True


# ============================================================
# Arguments
# ============================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live DW3000 serial -> GAT -> position probability density",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_argument_group("serial")
    source.add_argument("--port", help="collector port; auto-detected when omitted")
    source.add_argument("--baud", type=int, default=115200)
    source.add_argument(
        "--list-ports", action="store_true", help="print serial ports and exit"
    )
    source.add_argument(
        "--no-reset",
        action="store_true",
        help="do not let opening the port reset the ESP32 (keeps its seq)",
    )
    source.add_argument(
        "--echo", action="store_true", help="print every received serial line"
    )
    source.add_argument(
        "--checkpoint", default=str(SCRIPT_DIR / "checkpoints" / "best_model.pt")
    )

    saving = parser.add_argument_group("session log")
    saving.add_argument(
        "--save-dir-log",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="where the *_terminal.txt / *_gat_input.txt pair is written",
    )
    saving.add_argument(
        "--no-save-log",
        action="store_true",
        help="do not record the session (it cannot be replayed afterwards)",
    )

    run = parser.add_argument_group("run")
    run.add_argument("--json-out", help="one JSON line per frame; '-' for stdout")
    run.add_argument("--no-show", action="store_true", help="no window, PNG only")
    run.add_argument("--no-plot", action="store_true", help="no matplotlib at all")
    run.add_argument("--save-dir", help="directory for density_latest.png")
    run.add_argument(
        "--save-frames", action="store_true", help="also keep one PNG per frame"
    )
    run.add_argument("--render-every", type=int, default=1, help="render every Nth frame")
    run.add_argument("--dpi", type=int, default=110)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument(
        "--fast",
        action="store_true",
        help="lighter sampler and grid defaults for a Raspberry Pi",
    )
    run.add_argument(
        "--stale-warn",
        type=float,
        default=12.0,
        help="warn after this many seconds without an Original_DIST row",
    )

    links = parser.add_argument_group("link windows")
    links.add_argument("--window-size", type=int, help="override checkpoint window")
    links.add_argument("--min-samples", type=int, default=3)
    links.add_argument("--max-age-scans", type=int, default=15)
    links.add_argument("--min-person-degree", type=int, default=2)
    links.add_argument(
        "--no-anchor-links",
        action="store_true",
        help="let Top-K drop links to A or B (shows raw model selection)",
    )
    links.add_argument(
        "--max-forward-gap",
        type=int,
        default=64,
        help="scan-counter jump above which the rolling windows are dropped",
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
    sampling.add_argument("--anneal", type=float, default=20.0)
    sampling.add_argument("--jump-prob", type=float, default=0.15)
    sampling.add_argument("--jump-scale", type=float, default=10.0)
    sampling.add_argument("--no-mirror", action="store_true")

    grid = parser.add_argument_group("density grid")
    grid.add_argument("--grid-bins", type=int, default=220)
    grid.add_argument("--smoothing", type=float, default=0.30, help="metres")

    args = parser.parse_args(argv)
    if args.fast:
        apply_fast_preset(parser, args)
    return args


# A Pi 4 core runs roughly 4x slower than the laptop these defaults were tuned
# on, and the collector only produces a scan every ~2.5 s. These values keep a
# frame under that budget; the posterior is coarser but not biased.
FAST_PRESET = {
    "chains": 8,
    "samples": 1200,
    "burn_in": 600,
    "grid_bins": 140,
    "dpi": 90,
}


def apply_fast_preset(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Lower the defaults the user did not set explicitly."""
    changed = []
    for dest, value in FAST_PRESET.items():
        if getattr(args, dest) == parser.get_default(dest):
            setattr(args, dest, value)
            changed.append(f"{dest}={value}")
    if changed:
        print(f"[FAST] {' '.join(changed)}")


# ============================================================
# Main
# ============================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_ports:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports found (is pyserial installed, is DW plugged in?)")
            return 1
        for port in ports:
            vid = getattr(port, "vid", None)
            marker = " <- likely collector" if vid in KNOWN_USB_VIDS else ""
            print(f"{port.device}  {port.description}{marker}")
        return 0

    try:
        import serial  # noqa: F401  (the reader thread needs it)
    except ImportError:
        print("Install pyserial first: pip3 install pyserial", file=sys.stderr)
        return 2

    port = args.port or autodetect_port()
    if not port:
        print(
            "No serial port found. Plug the collector in and check with "
            "--list-ports, or pass --port /dev/ttyUSB0.",
            file=sys.stderr,
        )
        return 2
    if not args.port:
        print(f"[SERIAL] auto-detected {port}")

    if args.no_show and not args.no_plot and not args.save_dir:
        args.save_dir = str(REPO_ROOT / "density_output")
        print(f"[OUTPUT] headless: PNG -> {args.save_dir}/density_latest.png")

    import torch

    device = torch.device("cpu")
    model, runtime, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    describe_checkpoint(runtime, checkpoint)

    state = LiveState(window_size=args.window_size or runtime.window_size)
    rng = np.random.default_rng(args.seed)
    renderer = Renderer(args)

    json_handle = None
    if args.json_out:
        json_handle = (
            sys.stdout
            if args.json_out == "-"
            else open(args.json_out, "w", encoding="utf-8")
        )

    session = None
    if not args.no_save_log:
        session = SessionWriter(args.save_dir_log)
        print(f"[SAVE] terminal={session.terminal_path}")
        print(f"[SAVE] gat_input={session.gat_input_path}")

    source = SerialSource(
        port=port,
        baud=args.baud,
        raw_sink=session.write if session else None,
        assert_reset=not args.no_reset,
        echo=args.echo,
    )
    source.start()

    print("[RUN] Ctrl+C to stop")
    started = time.monotonic()
    scans = 0
    rendered = 0
    compute_total = 0.0
    warned_stale_s = 0.0

    try:
        while True:
            batch = source.drain(timeout=0.25)

            if not batch:
                renderer.keep_alive()
                if renderer.closed:
                    break
                if not source.alive:
                    print(
                        f"[SERIAL] reader thread stopped: {source.stats.error}",
                        file=sys.stderr,
                    )
                    break
                now = time.monotonic()
                reference = source.stats.last_scan_s or source.stats.opened_s
                silent = now - reference if reference else 0.0
                if silent > args.stale_warn and now - warned_stale_s > args.stale_warn:
                    warned_stale_s = now
                    if source.stats.lines and not source.stats.scans:
                        print(
                            f"[WAIT] {source.stats.lines} serial lines but no "
                            "Original_DIST: only the B collector prints it, so "
                            "this is probably another node's port.",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[WAIT] no scan for {silent:.1f} s "
                            "(collector powered? antenna range?)",
                            file=sys.stderr,
                        )
                continue

            # Every scan feeds the rolling windows; only the newest one is
            # worth sampling, because it already contains the others' history.
            for parsed in batch:
                seq, raw_seq = state.ingest(parsed, args.max_forward_gap)
                scans += 1

            ms = float(batch[-1].get("ms_float", 0.0))
            lag = len(batch) - 1

            compute_started = time.monotonic()
            frame, state.last_ab, state.previous_state = build_frame(
                model,
                state.bank,
                seq,
                ms,
                runtime,
                device,
                args,
                rng,
                state.last_ab,
                state.previous_state,
            )
            compute_s = time.monotonic() - compute_started

            if frame is None:
                if state.frames == 0 and scans % 4 == 1:
                    print(
                        f"[WARMUP] seq={raw_seq} "
                        f"{warmup_report(state, args.min_samples)}"
                    )
                continue

            state.frames += 1
            compute_total += compute_s

            renderer.render(frame, state.frames - 1)
            rendered += 1
            if json_handle:
                json_handle.write(frame_json(frame) + "\n")
                json_handle.flush()

            scored = len(frame.selection.links) + len(frame.selection.dropped)
            # The wire counter is what the terminal log shows, the unwrapped one
            # is what the JSON carries: raw repeats every 256 scans.
            print(
                f"[FRAME] seq={raw_seq}#{seq} t={ms / 1000.0:8.1f}s "
                f"links={len(frame.selection.links)}/{scored} "
                f"K={frame.selection.k_min}+{frame.selection.alpha} "
                f"accept={frame.acceptance:.2f} "
                f"rms={frame.residual_rms:.3f}m "
                f"compute={compute_s:.2f}s lag={lag}",
                flush=True,
            )
            print(f"         {format_peaks(frame)}", flush=True)

            if lag and compute_s > 1.0:
                print(
                    f"[LAG] {lag} scan(s) arrived while sampling; "
                    "add --fast or lower --samples to keep up.",
                    file=sys.stderr,
                )

            if renderer.closed:
                break

    except KeyboardInterrupt:
        print("\n[STOP]")
    finally:
        source.stop()
        if session:
            session.close()
        if json_handle and json_handle is not sys.stdout:
            json_handle.close()

    elapsed = time.monotonic() - started
    print(
        f"[DONE] {scans} scans, {rendered} frames in {elapsed:.1f} s "
        f"(mean compute {compute_total / max(state.frames, 1):.2f} s, "
        f"{source.stats.reconnects} reconnect(s), "
        f"{source.stats.dropped} scan(s) dropped)"
    )
    if session:
        print(f"[DONE] session saved: {session.gat_input_path}")
    if state.frames == 0:
        print(
            "[DONE] no frame was produced: no scan had enough links "
            "(try --min-samples 1 --max-age-scans 40)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
