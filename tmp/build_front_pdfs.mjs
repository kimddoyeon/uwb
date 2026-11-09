import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PDFDocument, StandardFonts, rgb } from 'pdf-lib';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'augmented_clean_v2');
const rows=JSON.parse(await fs.readFile(path.join(out,'manifest.json'),'utf8')).filter(x=>x.kind==='field_original');
for(const row of rows){
  const doc=await PDFDocument.create();
  const image=await doc.embedJpg(await fs.readFile(path.join(root,'tmp','pdfs',`pdf4-hi-${row.page}.jpg`)));
  const font=await doc.embedFont(StandardFonts.Helvetica);
  const bold=await doc.embedFont(StandardFonts.HelveticaBold);
  const iw=1050,ih=iw*image.height/image.width,W=1100,H=ih+165,ix=25,iy=86;
  const page=doc.addPage([W,H]);
  page.drawRectangle({x:0,y:0,width:W,height:H,color:rgb(1,1,1)});
  page.drawText('FIELD MEASUREMENT  |  Bldg 2 - Floor 4',{x:26,y:H-32,size:19,font:bold,color:rgb(.07,.13,.22)});
  page.drawText(`${row.source.replace('device-monitor-','').replace('.log','')}  |  ${row.node_count} nodes  |  source layout PDF page ${row.page}`,
    {x:27,y:H-53,size:11,font,color:rgb(.34,.4,.47)});
  page.drawImage(image,{x:ix,y:iy,width:iw,height:ih});
  page.drawRectangle({x:ix,y:iy,width:iw,height:ih,borderWidth:1,borderColor:rgb(.79,.83,.86)});
  page.drawText(`ACTIVE NODES: ${row.active.split('').join('  ')}`,{x:28,y:52,size:12,font:bold,color:rgb(.07,.13,.22)});
  page.drawText('Original field recording. Node positions are marked in red on the source photograph.',
    {x:28,y:30,size:10,font,color:rgb(.34,.4,.47)});
  await fs.writeFile(path.join(out,'2공4층','structures',row.structure_pdf),await doc.save());
}
console.log(`Created ${rows.length} PDFs for original 4- and 5-node floor layouts`);
