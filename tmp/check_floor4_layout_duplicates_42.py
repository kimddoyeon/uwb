import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

root=Path(__file__).resolve().parents[1]
imgdir=root/'tmp'/'pdfs'
records=json.loads((root/'augmented_clean'/'manifest.json').read_text(encoding='utf8'))
variants=[r for r in records if r['site']=='2공4층' and r['kind']=='synthetic']

def red_marks(page):
    rgb=np.asarray(Image.open(imgdir/f'pdf4-hi-{page}.png').convert('RGB'))[:590,:900,:].astype(np.int16)
    mask=(rgb[:,:,0]>90)&(rgb[:,:,0]>rgb[:,:,1]*1.45)&(rgb[:,:,0]>rgb[:,:,2]*1.45)
    lbl,n=ndimage.label(mask,np.ones((3,3),dtype=np.uint8))
    areas=np.bincount(lbl.ravel())
    pts=[]
    for i in range(1,n+1):
        area=areas[i]
        if not 15<=area<=500: continue
        ys,xs=np.where(lbl==i)
        if np.ptp(xs)>65 or np.ptp(ys)>65: continue
        pts.append((float(xs.mean()),float(ys.mean())))
    return pts

cache={p:red_marks(p) for p in [*range(3,19),*range(20,46)]}
matches=[]
for row in variants:
    targets=[tuple(2*v for v in row['coords'][n]) for n in row['active']]
    pages=range(3,19) if row['node_count']==4 else range(20,46)
    for page in pages:
        marks=cache[page]
        mins=[min(math.dist(t,m) for m in marks) for t in targets]
        if max(mins)<=42:
            matches.append((row['source'],row['removed'],row['active'],page,[round(x,1) for x in mins]))
print('Candidates with all node-position marks within 42px:',len(matches))
for m in matches:print(*m)
