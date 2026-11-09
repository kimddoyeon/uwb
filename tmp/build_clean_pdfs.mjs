import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PDFDocument, StandardFonts, rgb } from 'pdf-lib';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'augmented_clean');
const records = JSON.parse(await fs.readFile(path.join(out, 'manifest.json'),'utf8'));
const colors = {
  active: rgb(0.05,0.58,0.34), removed: rgb(0.87,0.18,0.16),
  isolated: rgb(0.91,0.47,0.07), gateway: rgb(0.98,0.68,0.04),
  ink: rgb(0.07,0.13,0.22), gray: rgb(0.39,0.45,0.51), white: rgb(1,1,1),
};
const imagePrefix = { '2공1층':'pdf1-hi-', '5강1층':'pdf2-hi-', '2공4층':'pdf4-hi-' };
const cache = new Map();
for (const row of records) {
  const key = `${row.site}-${row.page}`;
  let imageBytes = cache.get(key);
  if (!imageBytes) {
    imageBytes = await fs.readFile(path.join(root,'tmp','pdfs',`${imagePrefix[row.site]}${row.page}.jpg`));
    cache.set(key,imageBytes);
  }
  const doc = await PDFDocument.create();
  const photo = await doc.embedJpg(imageBytes);
  const font = await doc.embedFont(StandardFonts.Helvetica);
  const bold = await doc.embedFont(StandardFonts.HelveticaBold);
  const maxW = 1050, maxH = 590;
  const factor = Math.min(maxW/photo.width, maxH/photo.height);
  const iw = photo.width*factor, ih = photo.height*factor;
  const W = 1100, H = ih+175, ix=(W-iw)/2, iy=94;
  const page = doc.addPage([W,H]);
  page.drawRectangle({x:0,y:0,width:W,height:H,color:colors.white});
  page.drawText(`NODE LAYOUT  |  ${row.site === '2공1층' ? 'Bldg 2 - Floor 1' : row.site === '2공4층' ? 'Bldg 2 - Floor 4' : 'Lecture Bldg 5 - Floor 1'}`,
    {x:26,y:H-33,size:19,font:bold,color:colors.ink});
  page.drawText(`${row.source.replace('device-monitor-','').replace('.log','')}  |  ${row.node_count} nodes  |  ${row.kind === 'source' ? 'original acquisition' : 'derived node subset'}`,
    {x:27,y:H-53,size:11,font,color:colors.gray});
  page.drawText(`Source layout PDF / page ${row.page}`,
    {x:W-333,y:H-34,size:10,font,color:colors.gray});
  page.drawImage(photo,{x:ix,y:iy,width:iw,height:ih});
  page.drawRectangle({x:ix,y:iy,width:iw,height:ih,borderWidth:1,borderColor:rgb(0.79,0.83,0.86)});
  for (const node of 'ABCDEF') {
    const pos = row.coords?.[node];
    if (!pos) throw new Error(`Missing photo coordinate ${key} ${node}`);
    const x=ix + pos[0]/row.image_size[0]*iw;
    const y=iy + (1-pos[1]/row.image_size[1])*ih;
    const status = row.removed.includes(node) ? 'removed' : row.isolated.includes(node) ? 'isolated' : 'active';
    const c=colors[status];
    const radius = row.site==='5강1층' ? 10 : 8;
    page.drawCircle({x,y,size:radius,borderColor:c,borderWidth:2.4});
    if (status==='removed') {
      page.drawLine({start:{x:x-radius*.65,y:y-radius*.65},end:{x:x+radius*.65,y:y+radius*.65},thickness:2.3,color:c});
      page.drawLine({start:{x:x-radius*.65,y:y+radius*.65},end:{x:x+radius*.65,y:y-radius*.65},thickness:2.3,color:c});
    } else if(status==='isolated') {
      page.drawLine({start:{x:x-radius*.75,y:y-radius*.75},end:{x:x+radius*.75,y:y+radius*.75},thickness:2.3,color:c});
    }
    if (row.gateway===node) page.drawCircle({x,y,size:radius+4,borderColor:colors.gateway,borderWidth:3});
  }
  page.drawText(`ACTIVE: ${row.active.split('').join('  ')}     REMOVED: ${row.removed || '-'}     DISCONNECTED: ${row.isolated || '-'}`,
    {x:28,y:68,size:12,font:bold,color:colors.ink});
  const gwText = row.kind==='source' ? 'original log: no single simulated gateway' :
    row.moving ? 'SIM GATEWAY:           (moving layout; left blank)' :
    `SIM GATEWAY: ${row.gateway || ''}`;
  page.drawText(gwText,{x:28,y:48,size:11,font,color:colors.ink});
  page.drawText('Green = retained     Red X = physically removed     Orange slash = disconnected     Gold ring = simulated gateway',
    {x:28,y:27,size:10,font,color:colors.gray});
  if(row.moving) page.drawText('Moving node markers indicate labels on the source path, not a fixed position.',
    {x:W-446,y:48,size:9,font,color:colors.gray});
  const target = path.join(out,row.site,'structures',row.structure_pdf);
  await fs.writeFile(target,await doc.save());
}
console.log(`Created ${records.length} one-page PDFs from ${cache.size} source layout photos`);
