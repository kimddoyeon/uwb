import json
import re
from pathlib import Path

from build_clean_data import ROOT, OUT, PAIRS, parsed_scan, mask_for

items = json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
assert len(items) == 203
assert len({(r["site"],r["log"]) for r in items}) == len(items)
by_source = {}
for row in items:
    by_source.setdefault(row["source"],set()).add(row["active"])
    raw = (ROOT/"logs"/row["source"]).read_bytes()
    made = (OUT/row["site"]/"logs"/row["log"]).read_bytes()
    src_scans = [parsed_scan(x) for x in raw.splitlines() if x.startswith(b"Original_DIST,")]
    dst_scans = [parsed_scan(x) for x in made.splitlines() if x.startswith(b"Original_DIST,")]
    assert len(src_scans) == len(dst_scans) == row["scans"], row["log"]
    if row["kind"] == "source":
        assert made == raw, row["log"]
        continue
    active = set(row["active"])
    allowed = mask_for(active)
    assert len(active) == row["node_count"] and {"A","B"} <= active
    assert not (set(row["removed"]) & active)
    assert not (set(row["isolated"]) & active)
    assert made.count(b"SIM_GATEWAY,seq=") == len(src_scans), row["log"]
    assert made.count(b"SOURCE_GATEWAY,seq=") == len(src_scans), row["log"]
    expected_gw = (row["gateway"] or "").encode("ascii")
    assert all(x.endswith(b"node="+expected_gw) for x in made.splitlines() if x.startswith(b"SIM_GATEWAY,")), row["log"]
    for src,dst in zip(src_scans,dst_scans):
        assert src["seq"] == dst["seq"], row["log"]
        for key,val in src.items():
            match = re.fullmatch(r"(?:rssi_|fp_|rssi_fp_diff_)?([A-F]{2})",key)
            if match and not set(match[1]) <= active:
                assert dst[key] == "nan", (row["log"],key)
            elif key.endswith("_mask"):
                assert int(dst[key],16) == int(val,16) & allowed, (row["log"],key)
            else:
                assert dst[key] == val, (row["log"],key)
    for line in made.splitlines():
        match = re.match(rb"\|\s*([A-F]{2})\s*\|",line)
        if match and not set(match[1].decode()) <= active:
            assert b"INACTIVE" in line and b"nan" in line, (row["log"],line)
assert all(len(s) == len([x for x in items if x["source"] == source]) for source,s in by_source.items())
print(f"Validated {len(items)} logs from {len(by_source)} sources; source values preserved, subset masks and gateways checked")
