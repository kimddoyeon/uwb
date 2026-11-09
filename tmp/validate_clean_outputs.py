import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook

root=Path(__file__).resolve().parents[1]/'augmented_clean'
rows=json.loads((root/'manifest.json').read_text(encoding='utf8'))
pair_order='AB BC BD BE BF CD DE EF CE CF DF AC AD AE AF'.split()
titles={'2공1층':'2공 1층','5강1층':'5강 1층','2공4층':'2공 4층'}
for site,title in titles.items():
    sub=[r for r in rows if r['site']==site]
    book=root/site/f'{title}_노드별_데이터.xlsx'
    wb=load_workbook(book,read_only=True,data_only=True)
    index=list(wb['로그 목록'].iter_rows(min_row=8,values_only=True))
    detail=list(wb['쌍별 거리와 오차'].iter_rows(min_row=8,values_only=True))
    assert len(index)==len(sub) and len(detail)==len(sub)*15,site
    for i,r in enumerate(sub):
        x=index[i]
        assert x[0]==i+1 and x[3]==f"logs/{r['log']}" and x[4]==f"structures/{r['structure_pdf']}"
        assert x[8]==r['active'] and x[9]==r['node_count'] and x[10]==r['scans']
        log=root/site/x[3]
        pdf=root/site/x[4]
        assert log.exists() and pdf.exists()
        data=pdf.read_bytes()
        assert data.startswith(b'%PDF-') and b'%%EOF' in data[-100:]
        for j,p in enumerate(pair_order):
            y=detail[i*15+j]
            assert y[0]==i+1 and y[5]==p and y[7]==r['pairs'][p]['count']
            assert y[8]==r['pairs'][p]['mean'] and y[9]==r['pairs'][p]['median']
            assert y[10] is None and y[11] is None
    z=root/f'{site}_로그_엑셀_구조도.zip'
    with ZipFile(z) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(sub)*2+2
    print(site,Counter(r['node_count'] for r in sub),len(detail),'pair rows',round(z.stat().st_size/1048576,1),'MB zip')
print('All 203 log/PDF pairs, 3 Excel files, and 3 archives validated')
