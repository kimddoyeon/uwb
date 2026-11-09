import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
for(const [site,title] of [['2공1층','2공 1층'],['5강1층','5강 1층'],['2공4층','2공 4층']]) {
  const file=path.join(root,'augmented_clean',site,`${title}_노드별_데이터_수정.xlsx`);
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  const view=await wb.render({sheetName:'쌍별 거리와 오차',range:'A1:O14',scale:1,format:'png'});
  await fs.writeFile(path.join(root,'tmp',`environment-preview-${site}.png`),new Uint8Array(await view.arrayBuffer()));
}
console.log('Rendered three pair-detail previews');
