from __future__ import annotations

import itertools
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "augmented_clean"
PAIRS = "AB BC BD BE BF CD DE EF CE CF DF AC AD AE AF".split()
NODES = set("ABCDEF")

SITES = {
    "2공1층": {
        "pdf": "2공1층.pdf",
        "roots": ["150150", "151114", "152114", "152552", "153043", "154538", "155207", "155751", "160216", "161139", "170816", "171323", "171911", "172452"],
        "date": "260906", "first_page": 2, "moving_from_page": 7,
        "coords": {
            2: {"A": [830,462], "B": [910,461], "C": [984,319], "D": [983,463], "E": [984,568], "F": [986,770]},
            3: {"A": [830,462], "B": [910,461], "C": [985,357], "D": [898,365], "E": [985,566], "F": [885,566]},
            4: {"A": [830,462], "B": [910,461], "C": [985,426], "D": [946,385], "E": [985,534], "F": [945,568]},
            5: {"A": [830,462], "B": [910,461], "C": [980,461], "D": [979,383], "E": [980,571], "F": [884,571]},
            6: {"A": [830,462], "B": [910,461], "C": [980,461], "D": [980,384], "E": [980,570], "F": [886,386]},
            7: {"A": [830,462], "B": [910,461], "C": [930,585], "D": [985,383], "E": [985,771], "F": [985,514]},
            8: {"A": [830,462], "B": [910,461], "C": [944,185], "D": [985,383], "E": [985,774], "F": [985,514]},
            9: {"A": [830,462], "B": [910,461], "C": [940,480], "D": [985,383], "E": [985,775], "F": [985,514]},
            10: {"A": [830,462], "B": [910,461], "C": [962,185], "D": [985,383], "E": [985,775], "F": [985,514]},
            11: {"A": [830,462], "B": [910,461], "C": [970,616], "D": [985,383], "E": [953,200], "F": [985,514]},
            12: {"A": [830,462], "B": [895,461], "C": [930,190], "D": [974,190], "E": [924,482], "F": [924,456]},
            13: {"A": [825,462], "B": [890,461], "C": [940,184], "D": [965,184], "E": [975,773], "F": [994,773]},
            14: {"A": [825,462], "B": [890,461], "C": [935,190], "D": [960,190], "E": [985,773], "F": [998,773]},
            15: {"A": [825,462], "B": [890,461], "C": [950,616], "D": [968,616], "E": [975,773], "F": [995,773]},
        },
        "image_size": [1820, 851],
    },
    "5강1층": {
        "pdf": "5강 1층.pdf", "roots": ["173042", "171615"], "date": "260904", "pages": [2,5],
        "coords": {
            2: {"A": [294,160], "B": [280,132], "C": [280,143], "D": [298,140], "E": [210,121], "F": [210,129]},
            5: {"A": [297,140], "B": [278,138], "C": [244,134], "D": [278,123], "E": [244,148], "F": [295,164]},
        },
        "image_size": [364, 253],
    },
    "2공4층": {
        "pdf": "제2공4층 (2).pdf", "roots": ["150158", "153447", "150924", "151805"], "date": "260824", "pages": [47,48,49,50],
        "coords": {
            47: {"A": [427,75], "B": [378,77], "C": [376,138], "D": [178,76], "E": [178,111], "F": [178,140]},
            48: {"A": [29,54], "B": [29,76], "C": [113,75], "D": [174,75], "E": [117,116], "F": [178,115]},
            49: {"A": [375,34], "B": [375,78], "C": [421,80], "D": [176,75], "E": [175,134], "F": [377,138]},
            50: {"A": [376,35], "B": [376,79], "C": [425,80], "D": [178,76], "E": [395,270], "F": [438,268]},
        },
        "image_size": [683, 347],
    },
}

def parsed_scan(line: bytes):
    result = {}
    for token in line.rstrip(b"\r\n").split(b",")[1:]:
        if b"=" in token:
            key, value = token.split(b"=",1)
            result[key.decode("ascii")] = value.decode("ascii", "replace")
    return result

def scan_summary(raw: bytes):
    scans = [parsed_scan(line) for line in raw.splitlines() if line.startswith(b"Original_DIST,")]
    if not scans: raise ValueError("No Original_DIST rows")
    vals = {p: [] for p in PAIRS}
    for scan in scans:
        mask = int(scan.get("fresh_mask", "0"), 16)
        for i, p in enumerate(PAIRS):
            try: d = float(scan.get(p, "nan"))
            except ValueError: continue
            if mask & (1 << i) and math.isfinite(d) and d > 0:
                vals[p].append(d)
    graph = {n:set() for n in NODES}
    for pair, numbers in vals.items():
        if numbers:
            a,b = pair
            graph[a].add(b); graph[b].add(a)
    return scans, vals, graph

def component(graph, removed):
    seen = {"B"}
    todo = ["B"]
    while todo:
        node = todo.pop()
        for nxt in graph[node] - removed - seen:
            seen.add(nxt); todo.append(nxt)
    return seen

def gateway(graph, active, coords, moving):
    if moving: return None
    candidates = (graph["B"] & active) - {"B"}
    if not candidates: return None
    counts = {n: len(graph[n] & (active-{n})) for n in candidates}
    max_count = max(counts.values())
    tied = [n for n in candidates if counts[n] == max_count]
    if len(tied) == 1: return tied[0]
    if not coords: raise ValueError(f"No coordinates for static tie: {tied}")
    return min(tied, key=lambda n:(sum(math.dist(coords[n],coords[o]) for o in active if o != n), n))

def mask_for(active):
    return sum(1 << i for i,p in enumerate(PAIRS) if set(p) <= active)

PAIR_FIELD = re.compile(rb"^(?:rssi_|fp_|rssi_fp_diff_)?([A-F]{2})$")
MASK_FIELDS = {b"attempted_mask", b"fresh_mask", b"recovered_mask", b"last_measured_mask", b"last_no_response_mask", b"last_recovered_mask"}
TABLE = re.compile(rb"^(\|\s*)([A-F]{2})(\s*\|)(.*)$")

def transform(raw, active, source_name, removed, isolated, sim_gateway):
    allowed_mask = mask_for(active)
    header = (
        "# SYNTHETIC_NODE_SUBSET: original measured values retained for active pairs\r\n"
        f"# SOURCE_LOG: {source_name}\r\n"
        f"# PHYSICALLY_REMOVED: {''.join(sorted(removed))}\r\n"
        f"# DISCONNECTED_AFTER_REMOVAL: {''.join(sorted(isolated)) or '-'}\r\n"
        f"# ACTIVE_NODES: {''.join(sorted(active))}\r\n"
        f"# NODE_COUNT: {len(active)}\r\n"
        f"# SIM_GATEWAY: {sim_gateway or ''}\r\n"
        "# SOURCE_GATEWAY rows preserve original acquisition routes; SIM_GATEWAY is a virtual assignment.\r\n\r\n"
    ).encode("ascii")
    out = bytearray(header)
    for line in raw.splitlines(keepends=True):
        ending = b"\r\n" if line.endswith(b"\r\n") else (b"\n" if line.endswith(b"\n") else b"")
        body = line[:-len(ending)] if ending else line
        if body.startswith(b"Original_DIST,"):
            pieces = body.split(b",")
            for i in range(1,len(pieces)):
                if b"=" not in pieces[i]: continue
                key,value = pieces[i].split(b"=",1)
                match = PAIR_FIELD.fullmatch(key)
                if match and not set(match.group(1).decode("ascii")) <= active:
                    pieces[i] = key+b"=nan"
                elif key in MASK_FIELDS:
                    try: old = int(value,16)
                    except ValueError: old = 0
                    pieces[i] = key + b"=0x" + format(old & allowed_mask, "X").encode("ascii")
            body = b",".join(pieces)
        elif body.startswith(b"GATEWAY,"):
            fields = body.split(b",")
            # The source line records actual route information for retained pairs only.
            kept = [b"SOURCE_GATEWAY", fields[1]]
            for field in fields[2:]:
                if b"=" in field and set(field.split(b"=",1)[0].decode("ascii")) <= active:
                    kept.append(field)
            body = b",".join(kept)
            out.extend(body + ending)
            seq = parsed_scan(b"X,"+fields[1]).get("seq", "")
            out.extend(f"SIM_GATEWAY,seq={seq},node={sim_gateway or ''}".encode("ascii") + ending)
            continue
        else:
            match = TABLE.match(body)
            if match and match.group(2).decode("ascii") in PAIRS and not set(match.group(2).decode("ascii")) <= active:
                fields = body.split(b"|")
                if len(fields) >= 6:
                    fields[2:6] = [b" nan         ", b" nan       ", b" INACTIVE ", b" nan "]
                    body = b"|".join(fields)
        out.extend(body + ending)
    return bytes(out)

def main():
    OUT.mkdir(exist_ok=True)
    manifest = []
    site_counts = {}
    for site, cfg in SITES.items():
        site_dir = OUT/site
        (site_dir/"logs").mkdir(parents=True,exist_ok=True)
        (site_dir/"structures").mkdir(exist_ok=True)
        counts = {4:0,5:0,6:0}
        for idx, clock in enumerate(cfg["roots"]):
            page = cfg.get("pages", list(range(cfg.get("first_page",0), cfg.get("first_page",0)+len(cfg["roots"]))))[idx]
            source_name = f'device-monitor-{cfg["date"]}-{clock}.log'
            raw = (ROOT/"logs"/source_name).read_bytes()
            scans, vals, graph = scan_summary(raw)
            if component(graph,set()) != NODES: raise ValueError(f"Disconnected source: {source_name}")
            moving = page >= cfg.get("moving_from_page", 10**9)
            combos = [set(c) for k in (1,2) for c in itertools.combinations("CDEF",k)]
            variants = []
            found = set()
            for removed in combos:
                active = component(graph,removed)
                if len(active) not in (4,5) or "A" not in active: continue
                key = tuple(sorted(active))
                if key in found: continue
                found.add(key)
                variants.append((removed, active))
            cases = [(set(),NODES)] + variants
            for removed, active in cases:
                isolated = NODES - removed - active
                gw = gateway(graph,active,cfg["coords"].get(page),moving) if removed else None
                kind = "source" if not removed else "synthetic"
                suffix = f"{len(active)}node" + (f"-remove{''.join(sorted(removed))}" if removed else "")
                name = f'made-device-monitor-{cfg["date"]}-{clock}-{suffix}.log'
                pdf_name = name[:-4]+".pdf"
                data = raw if not removed else transform(raw,active,source_name,removed,isolated,gw)
                (site_dir/"logs"/name).write_bytes(data)
                counts[len(active)] += 1
                rec = {"site":site,"source":source_name,"source_pdf":cfg["pdf"],"page":page,
                    "kind":kind,"log":name,"structure_pdf":pdf_name,"removed":"".join(sorted(removed)),
                    "isolated":"".join(sorted(isolated)),"active":"".join(sorted(active)),"node_count":len(active),
                    "gateway":gw,"moving":moving,"scans":len(scans),"coords":cfg["coords"].get(page),
                    "image_size":cfg["image_size"],"pairs":{p:{"count":len(vals[p]) if set(p)<=active else 0,
                    "median":round(statistics.median(vals[p]),3) if vals[p] and set(p)<=active else None,
                    "mean":round(statistics.mean(vals[p]),3) if vals[p] and set(p)<=active else None,
                    "true_distance":None,"mae":None} for p in PAIRS}}
                manifest.append(rec)
        site_counts[site]=counts
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"total":len(manifest),"sites":site_counts},ensure_ascii=False))

if __name__ == "__main__": main()
