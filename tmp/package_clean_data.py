from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1] / 'augmented_clean_v2'
for site in ('2공1층','5강1층','2공4층'):
    target = root / f'{site}_로그_엑셀_구조도.zip'
    files = sorted(p for p in (root/site).rglob('*') if p.is_file() and p.suffix in ('.log','.pdf','.xlsx'))
    with ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        z.write(root/'README.md','README.md')
        for p in files: z.write(p,p.relative_to(root).as_posix())
    print(site,len(files),round(target.stat().st_size/1048576,1))
