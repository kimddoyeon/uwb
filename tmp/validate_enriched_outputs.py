import json
from pathlib import Path
from zipfile import ZipFile
from openpyxl import load_workbook

root=Path(__file__).resolve().parents[1]
out=root/'augmented_clean'
items=json.loads((root/'tmp'/'enriched_manifest.json').read_text(encoding='utf8'))
names={'2공1층':'2공 1층','5강1층':'5강 1층','2공4층':'2공 4층'}
pairs='AB BC BD BE BF CD DE EF CE CF DF AC AD AE AF'.split()
total_env=0; total_error=0
for site,title in names.items():
    recs=[x for x in items if x['site']==site]
    book=out/site/f'{title}_노드별_데이터_수정.xlsx'
    wb=load_workbook(book,read_only=True,data_only=True)
    ix=list(wb['로그 목록'].iter_rows(min_row=8,values_only=True))
    dt=list(wb['쌍별 거리와 오차'].iter_rows(min_row=8,values_only=True))
    assert len(ix)==len(recs) and len(dt)==15*len(recs)
    for i,r in enumerate(recs):
        assert ix[i][3]==f"logs/{r['log']}"
        assert ix[i][4]==f"structures/{r['structure_pdf']}"
        assert ix[i][6]==(r['removed'] or None)
        assert ix[i][7]==(r['isolated'] or None)
        for j,p in enumerate(pairs):
            row=dt[i*15+j]
            item=r['pairs'][p]
            env=item['environment']
            assert row[0]==i+1 and row[1]==p
            assert row[3]==('꺼짐' if p[0] in r['removed'] else '켜짐')
            assert row[4]==('꺼짐' if p[1] in r['removed'] else '켜짐')
            assert row[6]==(env['door'] if env else None)
            assert row[7]==(env['corner'] if env else None)
            assert row[8]==(env['los_nlos'] if env else None)
            assert row[9]==(env['wall'] if env else None)
            assert row[10]==item['count']
            assert row[13]==item['true_distance'] and row[14]==item['mae']
            total_env+=env is not None
            total_error+=item['mae'] is not None
    z=out/f'{site}_로그_엑셀_구조도_수정.zip'
    with ZipFile(z) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(recs)*2+2
        assert any(n.endswith(f'{title}_노드별_데이터_수정.xlsx') for n in archive.namelist())
print('Validated',len(items),'logs:',total_env,'environment rows and',total_error,'calculated error rows')
