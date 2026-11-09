# UWB monitoring data

`raspi/data_save.py` creates two files here for every monitoring session:

- `YYYY-MM-DD_HH-MM-SS_terminal.txt`: complete serial terminal data
- `YYYY-MM-DD_HH-MM-SS_gat_input.txt`: `Original_DIST,...` lines only

The terminal file includes `GATEWAY,seq=...` and `CIR,seq=...` lines for each
scan. Match them with `Original_DIST` by `seq`; the GAT input file omits both.

Generated `.txt` files are ignored by Git.
