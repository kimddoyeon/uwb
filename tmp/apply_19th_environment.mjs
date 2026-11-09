import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const projectDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const madeDir = path.join(projectDir, "made_log");
const source = path.join(madeDir, "data (1)_정리완료_5강까지_거리전체_19일_4-6노드추가.xlsx");
const output = path.join(madeDir, "data (1)_정리완료_5강까지_거리전체_19일_4-6노드추가_환경반영.xlsx");
const pairLabels = ["AB", "BC", "BD", "BE", "BF", "CD", "DE", "EF", "CE", "CF", "DF", "AC", "AD", "AE", "AF"];
const nodeLabels = ["A", "B", "C", "D", "E", "F"];

const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const sheet = wb.worksheets.items[0];
const used = sheet.getUsedRange();
const values = used.values;
let updated = 0;

for (let r = 0; r < values.length; r++) {
  const marker = String(values[r][0] ?? "");
  const name = String(values[r][1] ?? "");
  if (marker.length !== 2 || !name.includes("260919")) continue;

  const activeRow = values[r + 3] ?? [];
  const headerRow = values[r + 6] ?? [];
  const activeNodes = new Set(nodeLabels.filter((node, index) => activeRow[index + 1] === 1));
  const pairs = headerRow.slice(1, 16).map((value, index) => value || pairLabels[index]);
  const metadata = [
    [...(values[r + 9] ?? [])],
    [...(values[r + 10] ?? [])],
    [...(values[r + 11] ?? [])],
    [...(values[r + 12] ?? [])],
  ];
  for (let c = 0; c < pairs.length; c++) {
    const pair = String(pairs[c]);
    const allowed = pair.length === 2 && activeNodes.has(pair[0]) && activeNodes.has(pair[1]);
    metadata[0][c + 1] = allowed ? "nlos" : null;
    metadata[1][c + 1] = allowed ? 0 : null;
    metadata[2][c + 1] = allowed ? "none" : null;
    metadata[3][c + 1] = allowed ? 0 : null;
  }
  sheet.getRange(`A${r + 10}:P${r + 13}`).values = metadata;
  updated++;
}

wb.recalculate();
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
const preview = await wb.render({ sheetName: sheet.name, range: "A2377:P2422", scale: 1, format: "png" });
await fs.writeFile(path.join(projectDir, "tmp", "19th_environment_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const exported = await SpreadsheetFile.exportXlsx(wb);
await exported.save(output);
console.log(JSON.stringify({ updatedBlocks: updated, output, errorScan: errors.ndjson }, null, 2));
