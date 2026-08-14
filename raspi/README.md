# Raspberry Pi distance filtering

## Collected log -> probability distribution (uwb_probability_pipeline.py)

`uwb_probability_pipeline.py` is the single-process version of the whole chain:
it parses `Original_DIST`, runs the **trained** checkpoint instead of the rule
fallback, applies Top-K + alpha, and draws the per-person position probability
density. It reads either a saved log or the live serial port, so the data that
is already in `logs/` can be replayed without the hardware.

```bash
# newest logs/device-monitor-*.log, live window, one frame per scan
python raspi/uwb_probability_pipeline.py

# one specific log, final frame only, PNG without a window
python raspi/uwb_probability_pipeline.py \
  --input-log logs/device-monitor-260812-214621.log \
  --mode last --save-dir density_output --no-show

# live on the Pi
python3 raspi/uwb_probability_pipeline.py --port /dev/ttyUSB0 --baud 115200
```

Environment (torch is not needed by the other scripts in this folder):

```bash
python -m venv .venv-gat
.venv-gat/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-gat/Scripts/python -m pip install numpy matplotlib pyserial
```

On the Pi use `python3 -m venv .venv-gat` and the plain `pip install torch`.

### What it does per scan

1. Each link keeps a rolling window of the last `window_size` (8) raw
   measurements. No EMA runs first: the checkpoint was trained on the mean,
   std and slope of *raw* measurements, and pre-smoothing would collapse the
   std feature it depends on.
2. Edge features are `[mean_d / dist_scale, std_d / std_scale,
   slope / slope_scale, (mean_rss - rss_mean) / rss_std]`, all four scales read
   from the checkpoint, matching `gat/dataset.py`.
3. Links measured within `--max-age-scans` and holding at least
   `--min-samples` samples are scored; the rest are absent from the graph.
4. Top-K + alpha uses the checkpoint's `k_ratio` and `confidence_threshold`,
   over the links that actually have data. Then AB is force-kept because it
   defines the A-B baseline, every measured link to A or B is force-kept for
   the same reason `gat_distance_filter.py` force-keeps them, and every person
   keeps at least two incident links (`selectTopLinksWithCoverage.m`). Pass
   `--no-anchor-links` to see the unmodified model selection.
5. Metropolis-Hastings samples whole-node configurations that fit the selected
   distances, with A at the origin and B at `(AB, 0)`. Each sweep moves every
   person once, the burn-in is annealed, and a fraction of the proposals are
   long jumps so an under-constrained person spreads along its circle instead
   of freezing on one point.
6. Every sample is mirrored across `y = 0`: A and B both sit on that axis, so
   the reflected configuration is exactly as likely and the map has to show
   both. Peaks are therefore printed as `(x, +/-y)`.

`fit RMS` in the info panel is the diagnostic to read first. It is the median
RMS range residual of the drawn samples; when it is close to
`--posterior-sigma` the sampler found configurations that really fit, and when
it is several times larger the selected distances contradict each other and the
density below is not meaningful.

Add `--json-out -` to also print one selection JSON line per frame. That line
is accepted by the existing `gat_result_probability_map.py`.

### Checkpoint

`checkpoints/best_model.pt` is copied from the training project, and
`gat_model.py` is a copy of `gat/model.py` so the Pi does not need the training
tree. The architecture is chosen from the state-dict keys, not from
`model_config`, because both models are saved under the same file name.

The checkpoint currently in the repository holds **`EdgeOnlyMLP`**, the ablation
baseline: three dense layers over each link's own four features, with no graph
attention. It was saved at epoch 2 with `precision_at_k = 0.646` and
`reliable_if_error_le_m = 0.5`. The pipeline loads it and prints a warning.
Re-run `train.py` and drop the resulting `LightweightGAT` checkpoint in the same
place to get real GAT scores; nothing else in the pipeline changes.

### Known gaps in the current logs

In `logs/device-monitor-260812-*.log` the links BD, BE, AD, AE, EF, CF and DF
never report a measurement, so D and E are positioned only through CD, CE and
DE, relative to C. AB, AC and BC are also close to collinear (`AB + BC` exceeds
`AC` by about 0.2 m), which leaves C's position genuinely undetermined along a
long arc. The wide banana-shaped density for C is the correct posterior for that
geometry, not a sampler artefact -- confirm with `fit RMS`.


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

## Fixed C gateway collection

B remains the final collector and C is the fixed gateway. C becomes available
for the current scan after its direct pair with B completes a valid SS-TWR
exchange. RSSI and first-path power are not gateway eligibility thresholds.

If the C gateway attempt fails, B marks the pair `NO_RESPONSE`. The terminal
reports selection decisions as `GATEWAY_SELECT`, `GATEWAY_FAIL`, and
`GATEWAY_SUCCESS`. `Original_DIST` is emitted from completed-scan masks so a
gateway attempt cannot expire measurements taken near the beginning of that
scan.

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
