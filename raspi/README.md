# Raspberry Pi distance filtering

The B node prints a `RAW_DIST` block every 50 ms. Run the filter on the
Raspberry Pi connected to B:

```bash
python3 gat_distance_filter.py --port /dev/ttyUSB0 --baud 115200
```

Defaults:

- `--ema-alpha 0.35`
- `--threshold 0.55`
- `--topk 12`
- `--stale-ms 1000`

If `GAT_DIST` follows too slowly, raise `--ema-alpha` to `0.45`.
If the distance output shakes, lower it to `0.25` or `0.30`.

## Deterministic wall-link recovery before GAT

The B collector performs wall-link recovery before serial output, filtering,
GAT, or position-density processing. The configured topology treats AB, CD,
and EF as parallel vertical pairs with B, D, and F on the same door-side
baseline. A rule runs only when all three source distances were measured in the
current scan and the missing upper link explicitly reports no response.

```text
AC = sqrt(BD^2 + (AB - CD)^2)
AE = sqrt(BF^2 + (AB - EF)^2)
CE = sqrt(DF^2 + (CD - EF)^2)
```

Each result is stored in its normal pair slot and has no fabricated RSSI or
first-path power. `Original_DIST` marks recovered links in `recovered_mask`
(AC=`0x800`, AE=`0x2000`, CE=`0x100`; bits may be combined). The Raspberry Pi
pipeline does not use this mask as a GAT feature or probability. It applies the
mask only after scoring so a deterministic recovered link cannot disappear
from `GAT_DIST`/`RULE_DIST` due to threshold or top-k selection.

To log raw ESP output:

```bash
python3 log_raw_dist.py --port /dev/ttyUSB0 --output raw_dist_log.txt
```

## Save every monitoring session

Use `data_save.py` as the serial monitor when both the complete terminal output
and clean one-line GAT input need to be retained:

```bash
python3 data_save.py --port /dev/ttyUSB0 --baud 115200
```

On Windows, use the B collector port, for example:

```powershell
python raspi/data_save.py --port COM10 --baud 115200
```

Every run creates two files under the repository's `data_save` directory with
the local start time in their names:

- `YYYY-MM-DD_HH-MM-SS_terminal.txt` contains every terminal line.
- `YYYY-MM-DD_HH-MM-SS_gat_input.txt` contains only `Original_DIST,...` rows
  and can be passed directly to `gat_distance_filter.py --input-log`.

Stop monitoring with `Ctrl+C`; both files are flushed after every line.

Until a trained `gat_edge_filter.pt` exists, `gat_distance_filter.py` uses the
same filter/confidence pipeline and a rule-based keep probability fallback,
printing it as `RULE_DIST`.

## RSSI / first-path feature ablation

New `Original_DIST` logs contain `rssi_*`, `fp_*`, and
`rssi_fp_diff_*` values for every link. Train both variants with the same input
and seed to test whether the signal features improve validation F1:

```bash
python3 train_gat_edge_filter.py --input captured_labeled.csv \
  --output gat_distance_only.pt --disable-signal-features --seed 7

python3 train_gat_edge_filter.py --input captured_labeled.csv \
  --output gat_distance_signal.pt --seed 7
```

Compare `best_val_f1` from the two runs. The signal-aware model uses eight edge
features; old five-feature checkpoints are incompatible and must be retrained.

## Convert Zenodo 7629141

Extract `data_set.zip`, then convert its raw per-anchor CSV files into the
multi-link CSV accepted by the trainer. The mobile tag is mapped to node A and
the first five anchors (or the five supplied with `--anchors`) to B through F.

```bash
python3 convert_zenodo_uwb.py --input ../dataset/data_set/raw_data \
  --output zenodo_labeled.csv --anchors A1,A2,A3,A4,A5

python3 train_gat_edge_filter.py --input zenodo_labeled.csv \
  --output gat_edge_filter.pt
```

For a quick format check before converting the full data set, add
`--limit-per-group 10`. A link gets `keep=1` only when it is LOS and its range
error, calculated from the published tag/anchor coordinates, is at most 1 m.
Change this with `--error-threshold` or allow accurate NLOS measurements with
`--keep-nlos`.
