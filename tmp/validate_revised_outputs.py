import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from openpyxl import load_workbook

root=Path(__file__).resolve().parents[1]
out=root/'augmented_clean_v2'
rows=json.loads((out/'manifest.json').read_text(encoding='utf8'))
assert len(rows)==242
assert len({(r['site'],r['log']) for r in rows})==242
assert Counter(r['kind'] for r in rows)=={'synthetic':180,'field_original':42,'source':20}
sites={'2공1층':154,'5강1층':18,'2공4층':70}
names={'2공1층':'2공 1층','5강1층':'5강 1층','2공4층':'2공 4층'}
pairs='AB BC BD BE BF CD DE EF CE CF DF AC AD AE AF'.split()
mae_count=0
for site,expected in sites.items():
    subset=[r for r in rows if r['site']==site]
    assert len(subset)==expected
    wb=load_workbook(out/site/f'{names[site]}_노드별_데이터.xlsx',read_only=True,data_only=True)
    index=list(wb['로그 목록'].iter_rows(min_row=8,values_only=True))
    detail=list(wb['쌍별 거리와 오차'].iter_rows(min_row=8,values_only=True))
    assert len(index)==expected and len(detail)==expected*15
    for i,r in enumerate(subset):
        a=index[i]
        assert a[0]==i+1 and a[3]==f"logs/{r['log']}" and a[4]==f"structures/{r['structure_pdf']}"
        log=out/site/'logs'/r['log']
        pdf=out/site/'structures'/r['structure_pdf']
        assert log.exists() and pdf.exists()
        assert pdf.read_bytes().startswith(b'%PDF-')
        if r['kind']!='synthetic':
            assert log.read_bytes()==(root/'logs'/r['source']).read_bytes()
        for j,p in enumerate(pairs):
            b=detail[i*15+j]
            item=r['pairs'][p]
            assert b[0]==i+1 and b[1]==p and b[10]==item['count']
            assert b[13]==item['true_distance'] and b[14]==item['mae']
            assert b[6] in ('none','open','close',None)
            if r['kind']=='field_original' and not set(p)<=set(r['active']):
                assert b[5]=='미배치'
            mae_count+=b[14] is not None
    with ZipFile(out/f'{site}_로그_엑셀_구조도.zip') as z:
        assert z.testzip() is None
        assert len(z.namelist())==expected*2+2
    print(site,Counter(r['node_count'] for r in subset),'rows',len(detail))
for d in json.loads((out/'duplicates_excluded.json').read_text(encoding='utf8')):
    assert not any(r['source']==d['synthetic_source'] and r['removed']==d['removed'] for r in rows)
assert mae_count==161
print('Validated 242 logs/PDFs, 3 workbooks, 3 archives, 3 excluded duplicates, 161 mean-error cells')
