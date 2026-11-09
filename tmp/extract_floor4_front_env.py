import json
import re
from pathlib import Path
from openpyxl import load_workbook

root=Path(__file__).resolve().parents[1]
pages=json.loads((root/'tmp'/'floor4_front_pages.json').read_text(encoding='utf8'))
rows=list(load_workbook(root/'data (1)_2공1층_완료.xlsx',read_only=True,data_only=True).active.values)
pairs='AB BC BD BE BF CD DE EF CE CF DF AC AD AE AF'.split()
records={}
for page_text,root_id in pages.items():
    page=int(page_text)
    hits=[i for i,r in enumerate(rows,1) if len(r)>1 and r[0]=='로그' and root_id[-6:] in str(r[1])]
    if len(hits)!=1: raise ValueError((root_id,hits))
    b=hits[0]
    env={}
    for j,p in enumerate(pairs):
        raw=rows[b+8][j+1]
        if raw is None: continue
        match=re.match(r'^(nlos|los)',str(raw),re.I)
        if not match: raise ValueError((root_id,p,raw))
        door=rows[b+10][j+1]
        door={'opne':'open','closed':'close'}.get(door,door)
        if door not in ('none','open','close'): raise ValueError((root_id,p,door))
        env[p]={
            'los_nlos':match.group(1).lower(),
            'los_note':str(raw)[len(match.group(1)):].strip('() '),
            'wall':rows[b+9][j+1],
            'door':door,
            'corner':rows[b+11][j+1],
            'provenance':'기존 엑셀 현장 기록',
        }
    records[root_id]={'page':page,'env':env}

# Explicit annotations on the photographs: open doors on page 31;
# X over the particular door on pages 32 and 44 means closed.
manual={
    # Page 6: A is at the unmarked room-405 entrance; the other entrance has X.
    '260822-143413':{'AC':'open','AD':'open','BC':'close','BD':'close'},
    '260825-110208':{'DE':'open'},
    # Page 32: C/D share room 402, A/B/E share room 405. Both entrances have X.
    '260825-110856':{'AC':'close','AD':'close','BC':'close','BD':'close',
                     'CE':'close','DE':'close'},
    # Page 34: C and A face each other across the X-marked room-405 entrance.
    '260825-112040':{'AC':'close'},
    # Page 44: C/E share room 402, whose entrance has an X.
    '260825-124316':{'AC':'close','AE':'close','BC':'close','BE':'close',
                     'CD':'close','DE':'close'},
}
for root_id, entries in manual.items():
    for pair,door in entries.items():
        old=records[root_id]['env'].get(pair,{})
        records[root_id]['env'][pair]={
            **old,
            'door':door,
            'los_nlos':'nlos' if door=='close' else old.get('los_nlos'),
            'los_note':old.get('los_note') or ('도면의 문 X 표시' if door=='close' else '도면의 문 앞 노드, X 없음'),
            'provenance':f"제2공4층 (2).pdf {records[root_id]['page']}쪽 문 표시",
        }
(root/'tmp'/'floor4_front_environment.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
print('Prepared',len(records),'front layouts,',sum(len(v['env']) for v in records.values()),'pair annotations')
