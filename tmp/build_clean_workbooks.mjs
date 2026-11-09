import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root,'augmented_clean_v2');
const all = JSON.parse(await fs.readFile(path.join(out,'manifest.json'),'utf8'));
const PAIRS = 'AB BC BD BE BF CD DE EF CE CF DF AC AD AE AF'.split(' ');
const SITE_TITLE = {'2공1층':'2공 1층','5강1층':'5강 1층','2공4층':'2공 4층'};
const navy='#172B4D', teal='#006E75', muted='#526171', light='#EAF4F5';
const isOriginal=r=>r.kind==='source'||r.kind==='field_original';
const kindLabel=r=>isOriginal(r)?'실측 원본':'증강';

function styleTitle(sheet, endCol, title, note) {
  sheet.showGridLines=false;
  sheet.mergeCells(`A1:${endCol}1`);
  sheet.getRange('A1').values=[[title]];
  sheet.getRange(`A1:${endCol}1`).format={fill:navy,font:{name:'Malgun Gothic',size:16,bold:true,color:'#FFFFFF'},rowHeight:34};
  sheet.mergeCells(`A2:${endCol}2`);
  sheet.getRange('A2').values=[[note]];
  sheet.getRange(`A2:${endCol}2`).format={fill:light,font:{name:'Malgun Gothic',size:10,color:muted},rowHeight:25};
}
function header(sheet, range) {
  sheet.getRange(range).format={fill:teal,font:{name:'Malgun Gothic',size:10,bold:true,color:'#FFFFFF'},rowHeight:30,
    verticalAlignment:'center',wrapText:true};
}
const summaries=[];
for(const site of Object.keys(SITE_TITLE)) {
  const records=all.filter(r=>r.site===site);
  const wb=Workbook.create();
  const index=wb.worksheets.add('로그 목록');
  const detail=wb.worksheets.add('쌍별 거리와 오차');
  index.tabColor=navy; detail.tabColor=teal;
  styleTitle(index,'L',`${SITE_TITLE[site]} | 노드 구성별 로그`,
    '실측 원본과 중복되지 않는 증강 조합을 모았습니다. 환경은 확인된 현장 기록을 반영하고, 실거리·오차는 확정된 쌍만 표시합니다.');
  for(const [range,value] of [
    ['A4:B4',`실측 원본 ${records.filter(isOriginal).length}개`],
    ['D4:E4',`증강 ${records.filter(r=>r.kind==='synthetic').length}개`],
    ['H4:I4',`전체 ${records.length}개`]]) {
    index.mergeCells(range);
    const cell=range.split(':')[0];
    index.getRange(cell).values=[[value]];
    index.getRange(range).format={fill:light,font:{name:'Malgun Gothic',size:11,bold:true,color:navy},rowHeight:28};
  }
  const ih=['번호','구분','원본 로그','로그 파일','구조도 PDF','원본 도면 쪽','전원 꺼진/제거 노드','연결 단절 노드','유효 노드','노드 수','스캔 수','이동 배치'];
  index.getRange('A7:L7').values=[ih]; header(index,'A7:L7');
  const rows=records.map((r,i)=>[i+1,kindLabel(r),r.source,`logs/${r.log}`,`structures/${r.structure_pdf}`,r.page,
    r.removed||null,r.isolated||null,r.active,r.node_count,r.scans,r.moving?'예':'아니요']);
  index.getRange(`A8:L${7+rows.length}`).values=rows;
  index.getRange(`A8:L${7+rows.length}`).format.font={name:'Malgun Gothic',size:9,color:navy};
  index.getRange(`A8:L${7+rows.length}`).format.rowHeight=21;
  index.getRange(`A8:L${7+rows.length}`).format.borders={insideHorizontal:{style:'thin',color:'#E0E7ED'}};
  for(const [col,width] of Object.entries({A:7,B:9,C:37,D:62,E:62,F:15,G:14,H:18,I:15,J:10,K:11,L:12}))
    index.getRange(`${col}:${col}`).format.columnWidth=width;
  index.freezePanes.freezeRows(7);

  styleTitle(detail,'S',`${SITE_TITLE[site]} | 15개 노드 쌍별 구조·거리·오차`,
    '한 로그마다 15쌍을 각각 기록합니다. 평균 절대오차 = 유효 측정마다 |측정거리 - 실거리|의 평균입니다.');
  detail.mergeCells('A3:S3');
  detail.getRange('A3').values=[['전원 꺼짐과 연결 단절은 다릅니다. 환경·실거리·오차의 빈칸은 미확정 값이며 0으로 해석하지 마세요.']];
  detail.getRange('A3:S3').format={font:{name:'Malgun Gothic',size:10,color:'#A44816'},rowHeight:22};
  const dh=['로그 번호','노드 쌍','노드 수','첫 노드 전원','둘째 노드 전원','쌍 상태','문 상태','코너','LOS/NLOS','벽','유효 측정 수','측정 평균 (m)','측정 중앙값 (m)','실제 거리 (m)','평균 절대오차 (m)','원본 도면 쪽','환경 비고','환경 출처','거리 출처'];
  detail.getRange('A7:S7').values=[dh]; header(detail,'A7:S7');
  const drows=[];
  for(let i=0;i<records.length;i++) {
    const r=records[i];
    for(const p of PAIRS) {
      const x=r.pairs[p];
      const off=[...p].some(n=>r.removed.includes(n));
      const isolated=[...p].some(n=>r.isolated.includes(n));
      const absent=r.kind==='field_original'&&[...p].some(n=>!r.active.includes(n));
      const state=absent?'미배치':off?'전원 꺼짐':isolated?'연결 단절':x.count?'측정 있음':'측정 없음';
      const env=x.environment;
      const power=n=>r.kind==='field_original'&&!r.active.includes(n)?'미배치':r.removed.includes(n)?'꺼짐':'켜짐';
      drows.push([i+1,p,r.node_count,power(p[0]),power(p[1]),state,
        env?.door??null,env?.corner??null,env?.los_nlos??null,env?.wall??null,
        x.count,x.mean,x.median,x.true_distance,x.mae,r.page,env?.los_note||null,env?.provenance??null,x.true_source??null]);
    }
  }
  detail.getRange(`A8:S${7+drows.length}`).values=drows;
  detail.getRange(`A8:S${7+drows.length}`).format.font={name:'Malgun Gothic',size:9,color:navy};
  detail.getRange(`L8:O${7+drows.length}`).setNumberFormat('0.000');
  for(const [col,width] of Object.entries({A:12,B:10,C:10,D:15,E:15,F:15,G:14,H:10,I:14,J:10,K:16,L:18,M:20,N:19,O:24,P:16,Q:30,R:23,S:25}))
    detail.getRange(`${col}:${col}`).format.columnWidth=width;
  detail.freezePanes.freezeRows(7);
  wb.recalculate();
  const target=path.join(out,site,`${SITE_TITLE[site]}_노드별_데이터.xlsx`);
  const file=await SpreadsheetFile.exportXlsx(wb);
  await file.save(target);
  const preview=await wb.render({sheetName:'로그 목록',range:'A1:L15',scale:1,format:'png'});
  await fs.writeFile(path.join(root,'tmp',`preview-${site}.png`),new Uint8Array(await preview.arrayBuffer()));
  summaries.push({site,logs:rows.length,pairRows:drows.length,target});
}
console.log(JSON.stringify(summaries));
