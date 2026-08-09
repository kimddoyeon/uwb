#!/usr/bin/env python3
import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--output", default="raw_dist_log.txt")
    args = parser.parse_args()

    import serial

    count = 0
    in_block = False
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        with open(args.output, "a", encoding="utf-8") as fp:
            print(f"logging RAW_DIST from {args.port} to {args.output}")
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("RAW_DIST_BEGIN"):
                    in_block = True
                    count += 1
                    print(f"{time.strftime('%H:%M:%S')} block #{count}", flush=True)
                if in_block or line.startswith("RAW_DIST_CSV,"):
                    fp.write(line + "\n")
                    fp.flush()
                if line == "RAW_DIST_END":
                    in_block = False


if __name__ == "__main__":
    raise SystemExit(main())
