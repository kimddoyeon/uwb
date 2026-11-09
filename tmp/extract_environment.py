import json
import re
from pathlib import Path
from openpyxl import load_workbook

root=Path(__file__).resolve().parents[1]
source=root/'data (1)_정리완료_5강까지_거리전체.xlsx'
rows=list(load_workbook(source,read_only=True,data_only=True).active.values)
pairs='AB BC BD BE BF CD DE EF CE CF DF AC AD AE AF'.split()
bases={
    '260824-150158':410,'260824-153447':424,'260824-150924':438,'260824-151805':452,
    '260906-150150':558,'260906-151114':603,'260906-152114':648,'260906-152552':693,'260906-153043':738,
    '260904-173042':1957,'260904-171615':2122,
}
result={}
for root_id,b in bases.items():
    entry={}
    for j,p in enumerate(pairs):
        raw_los=rows[b+8][j+1]
        if raw_los is None:
            entry[p]=None
            continue
        text=str(raw_los).strip()
        match=re.match(r'^(nlos|los)',text,re.I)
        if not match: raise ValueError((root_id,p,text))
        door=rows[b+10][j+1]
        if door not in ('none','open','close'): raise ValueError((root_id,p,door))
        wall=rows[b+9][j+1]
        corner=rows[b+11][j+1]
        if wall not in (0,1) or corner not in (0,1): raise ValueError((root_id,p,wall,corner))
        entry[p]={
            'los_nlos':match.group(1).lower(),
            'los_note':text[len(match.group(1)):].strip('() '),
            'wall':wall,'door':door,'corner':corner,
            'old_distance_candidate':rows[b+7][j+1],
        }
    result[root_id]=entry
(root/'tmp'/'environment_source.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('Extracted',len(result),'static source layouts,',sum(x is not None for r in result.values() for x in r.values()),'pair annotations')
