import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'augmented_clean'
records=json.loads((out/'manifest.json').read_text(encoding='utf8'))
annotations=json.loads((root/'tmp'/'environment_source.json').read_text(encoding='utf8'))

for row in records:
    root_id=row['source'].removeprefix('device-monitor-').removesuffix('.log')
    for pair,item in row['pairs'].items():
        if not set(pair) <= set(row['active']):
            item['environment']=None
            continue
        source=annotations.get(root_id,{}).get(pair)
        item['environment']={
            'los_nlos':source['los_nlos'],
            'los_note':source['los_note'],
            'wall':source['wall'],
            'door':source['door'],
            'corner':source['corner'],
            'provenance':'기존 엑셀 현장 기록',
        } if source is not None else None

(root/'tmp'/'enriched_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
print('Prepared',len(records),'records with',sum(bool(p.get('environment')) for r in records for p in r['pairs'].values()),'retained-pair annotations')
