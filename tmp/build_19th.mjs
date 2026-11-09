import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const projectDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceWorkbook = path.join(projectDir, "data (1)_정리완료_5강까지_거리전체.xlsx");
const outputDir = path.join(projectDir, "made_log");
const outputWorkbook = path.join(outputDir, "data (1)_정리완료_5강까지_거리전체_19일추가.xlsx");
const pairLabels = ["AB", "BC", "BD", "BE", "BF", "CD", "DE", "EF", "CE", "CF", "DF", "AC", "AD", "AE", "AF"];

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function parseLog(text) {
  const rows = [];
  for (const rawLine of text.split(/\r?\n/)) {
    if (!rawLine.startsWith("Original_DIST,")) continue;
    const fields = {};
    for (const token of rawLine.split(",").slice(1)) {
      const eq = token.indexOf("=");
      if (eq >= 0) fields[token.slice(0, eq)] = token.slice(eq + 1);
    }
    const freshMask = Number.parseInt(fields.fresh_mask ?? "0", 16) || 0;
    const values = {};
    pairLabels.forEach((label, index) => {
      const value = Number.parseFloat(fields[label] ?? "NaN");
      if ((freshMask & (1 << index)) !== 0 && Number.isFinite(value) && value > 0) {
        values[label] = value;
      }
    });
    rows.push(values);
  }
  const distances = pairLabels.map((label) => {
    const values = rows.map((row) => row[label]).filter((value) => Number.isFinite(value));
    return values.length ? Number(median(values).toFixed(3)) : null;
  });
  return { rows: rows.length, distances };
}

await fs.mkdir(outputDir, { recursive: true });
const sourceLogDir = path.join(projectDir, "logs");
const sourceLogs = (await fs.readdir(sourceLogDir))
  .filter((name) => /^device-monitor-260919-\d{6}\.log$/i.test(name))
  .sort();
if (sourceLogs.length !== 6) throw new Error(`Expected 6 logs for 260919, found ${sourceLogs.length}`);

const summaries = [];
for (const sourceName of sourceLogs) {
  const sourcePath = path.join(sourceLogDir, sourceName);
  const text = await fs.readFile(sourcePath, "utf8");
  const parsed = parseLog(text);
  const madeName = sourceName.replace(/^device-monitor-/, "made-device-monitor-").replace(/\.log$/i, "-6node.log");
  await fs.writeFile(path.join(outputDir, madeName), text, "utf8");
  summaries.push({ sourceName, madeName, ...parsed });
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourceWorkbook));
const sheet = workbook.worksheets.items[0];
const used = sheet.getUsedRange();
const existingValues = used.values;
let lastNonEmptyRow = -1;
for (let r = 0; r < existingValues.length; r++) {
  if (existingValues[r].some((value) => value !== null && value !== "")) lastNonEmptyRow = r;
}
let startRow = lastNonEmptyRow + 3; // preserve the two blank rows between blocks
const blockHeight = 15;

for (const summary of summaries) {
  const start = startRow;
  const end = start + blockHeight - 1;
  const sourceStyleStart = end - blockHeight + 1;
  const templateStart = Math.max(0, start - blockHeight);
  sheet.getRange(`A${start + 1}:P${end + 1}`).copyFrom(sheet.getRange(`A${templateStart + 1}:P${templateStart + blockHeight}`), "all");
  const rows = Array.from({ length: blockHeight }, () => Array(16).fill(null));
  rows[0][0] = "로그";
  rows[0][1] = summary.madeName;
  rows[1][0] = "노드 상태";
  rows[2] = ["node", "A", "B", "C", "D", "E", "F", null, null, null, null, null, null, null, null, null];
  rows[3] = ["active", 1, 1, 1, 1, 1, 1, null, null, null, null, null, null, null, null, null];
  rows[5][0] = "환경";
  rows[6] = [null, ...pairLabels];
  rows[7] = ["pair", ...summary.distances.map((value) => value === null ? null : 1)];
  rows[8] = ["distance", ...summary.distances];
  rows[9][0] = "los_nlos";
  rows[10][0] = "wall";
  rows[11][0] = "door";
  rows[12][0] = "corner";
  sheet.getRange(`A${start + 1}:P${end + 1}`).values = rows;
  startRow = end + 1;
}

workbook.recalculate();
const checkRange = `A${lastNonEmptyRow + 4}:P${startRow}`;
const check = await workbook.inspect({ kind: "table", sheetId: sheet.name, range: checkRange, include: "values,formulas", tableMaxRows: 100, tableMaxCols: 16, tableMaxCellChars: 120 });
console.log("CHECK_RANGE", checkRange);
console.log(check.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log("ERROR_SCAN", errors.ndjson);

const preview = await workbook.render({ sheetName: sheet.name, range: checkRange, scale: 1, format: "png" });
await fs.writeFile(path.join(projectDir, "tmp", "19th_append_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputWorkbook);
console.log(JSON.stringify({ outputWorkbook, outputDir, sourceLogs, summaries, appendedStartRow: lastNonEmptyRow + 4, appendedEndRow: startRow }, null, 2));
