import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_clean_data import ROOT, PAIRS, scan_summary

old=ROOT/'augmented_clean'
new=ROOT/'augmented_clean_v2'
new.mkdir(exist_ok=True)
rows=json.loads((ROOT/'tmp'/'enriched_manifest.json').read_text(encoding='utf8'))
pages=json.loads((ROOT/'tmp'/'floor4_front_pages.json').read_text(encoding='utf8'))
environment=json.loads((ROOT/'tmp'/'floor4_front_environment.json').read_text(encoding='utf8'))
duplicates={
    ('device-monitor-260824-150158.log','CE'):('device-monitor-260823-184517.log',13),
    ('device-monitor-260824-150924.log','F'):('device-monitor-260823-194156.log',23),
    ('device-monitor-260824-151805.log','DF'):('device-monitor-260823-174822.log',8),
}
kept=[]
for row in rows:
    if (row['source'],row['removed']) in duplicates: continue
    site=row['site']
    for sub in ('logs','structures'):(new/site/sub).mkdir(parents=True,exist_ok=True)
    shutil.copy2(old/site/'logs'/row['log'],new/site/'logs'/row['log'])
    shutil.copy2(old/site/'structures'/row['structure_pdf'],new/site/'structures'/row['structure_pdf'])
    kept.append(row)

front=[]
for page_text,root_id in pages.items():
    page=int(page_text)
    active='ABCD' if page<=18 else 'ABCDE'
    node_count=len(active)
    source=f'device-monitor-{root_id}.log'
    raw=(ROOT/'logs'/source).read_bytes()
    scans,vals,graph=scan_summary(raw)
    name=f'device-monitor-{root_id}-{node_count}node-original.log'
    pdf_name=name[:-4]+'.pdf'
    (new/'2공4층'/'logs'/name).write_bytes(raw)
    pair_data={}
    for p in PAIRS:
        measurements=vals[p] if set(p)<=set(active) else []
        pair_data[p]={
            'count':len(measurements),
            'median':round(statistics.median(measurements),3) if measurements else None,
            'mean':round(statistics.mean(measurements),3) if measurements else None,
            'true_distance':None,'mae':None,
            'environment':environment[root_id]['env'].get(p) if set(p)<=set(active) else None,
        }
    front.append({
        'site':'2공4층','source':source,'source_pdf':'제2공4층 (2).pdf','page':page,
        'kind':'field_original','log':name,'structure_pdf':pdf_name,'removed':'','isolated':'',
        'active':active,'node_count':node_count,'gateway':None,'moving':False,
        'scans':len(scans),'coords':None,'image_size':[683,347],'pairs':pair_data,
    })

ordered=[*sorted((r for r in kept if r['site']=='2공1층'),key=lambda r:(r['page'],r['node_count']==6,r['log'])),
         *sorted((r for r in kept if r['site']=='5강1층'),key=lambda r:(r['page'],r['node_count']==6,r['log'])),
         *sorted(front,key=lambda r:r['page']),
         *sorted((r for r in kept if r['site']=='2공4층'),key=lambda r:(r['page'],r['node_count']==6,r['log']))]
(new/'manifest.json').write_text(json.dumps(ordered,ensure_ascii=False,indent=2),encoding='utf8')
(new/'duplicates_excluded.json').write_text(json.dumps([
    {'synthetic_source':k[0],'removed':k[1],'measured_source':v[0],'measured_page':v[1]}
    for k,v in duplicates.items()],ensure_ascii=False,indent=2),encoding='utf8')
print('Revised records:',len(ordered),'front originals:',len(front),'excluded duplicates:',len(duplicates))
