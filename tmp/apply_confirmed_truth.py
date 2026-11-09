import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_clean_data import ROOT, scan_summary

path=ROOT/'tmp'/'enriched_manifest.json'
records=json.loads(path.read_text(encoding='utf8'))
cache={}
confirmed={
    # User checked these two dimensions against the drawings on 2026-10-27.
    ('device-monitor-260906-150150.log','BC'):(7.7,'사용자 도면 계산: 약 7.7m'),
    ('device-monitor-260904-173042.log','AB'):(9.0,'사용자 도면 계산: 9m'),
}
count={'2공1층 AB':0,'2공1층 BC':0,'5강1층 AB':0}
for row in records:
    source=row['source']
    if source not in cache:
        _,vals,_=scan_summary((ROOT/'logs'/source).read_bytes())
        cache[source]=vals
    truth={}
    if row['site']=='2공1층': truth['AB']=(4.7,'2공1층.pdf 1쪽 AB 4.7m')
    for pair in ('AB','BC'):
        if (source,pair) in confirmed: truth[pair]=confirmed[source,pair]
    for pair,(distance,provenance) in truth.items():
        vals=cache[source][pair]
        item=row['pairs'][pair]
        if set(pair)<=set(row['active']):
            item['true_distance']=distance
            item['true_source']=provenance
            if vals and item['count']:
                item['mae']=round(statistics.mean(abs(v-distance) for v in vals),3)
                count[f"{row['site']} {pair}"]+=1
            else:
                item['mae']=None
path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
print('Applied confirmed distances:',count)
