import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const projectDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const madeDir = path.join(projectDir, "made_log");
const sourceWorkbook = path.join(madeDir, "data (1)_정리완료_5강까지_거리전체_19일추가.xlsx");
const outputWorkbook = path.join(madeDir, "data (1)_정리완료_5강까지_거리전체_19일_4-6노드추가.xlsx");
const sourceLogDir = path.join(projectDir, "logs");
const pairLabels = ["AB", "BC", "BD", "BE", "BF", "CD", "DE", "EF", "CE", "CF", "DF", "AC", "AD", "AE", "AF"];
const nodeLabels = ["A", "B", "C", "D", "E", "F"];

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function parseSummary(text) {
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
      if ((freshMask & (1 << index)) !== 0 && Number.isFinite(value) && value > 0) values[label] = value;
    });
    rows.push(values);
  }
  return {
    rows: rows.length,
    distances: Object.fromEntries(pairLabels.map((label) => {
      const values = rows.map((row) => row[label]).filter((value) => Number.isFinite(value));
      return [label, values.length ? Number(median(values).toFixed(3)) : null];
    })),
  };
}

function maskFor(activeNodes) {
  let mask = 0;
  pairLabels.forEach((label, index) => {
    if (activeNodes.has(label[0]) && activeNodes.has(label[1])) mask |= 1 << index;
  });
  return mask;
}

function transformLog(text, activeNodes) {
  const allowedMask = maskFor(activeNodes);
  const fullMask = (1 << pairLabels.length) - 1;
  const pairIsAllowed = (label) => activeNodes.has(label[0]) && activeNodes.has(label[1]);
  return text.split(/\r?\n/).map((line) => {
    if (line.startsWith("Original_DIST,")) {
      const tokens = line.split(",");
      return tokens.map((token, index) => {
        if (index === 0 || !token.includes("=")) return token;
        const [key, value] = token.split(/=(.*)/s);
        const pairKey = key.match(/^(?:rssi_|fp_|rssi_fp_diff_)?([A-F]{2})$/)?.[1];
        if (pairKey && !pairIsAllowed(pairKey)) return `${key}=nan`;
        if (key === "attempted_mask") return `${key}=0x${allowedMask.toString(16).toUpperCase()}`;
        if (key === "fresh_mask" || key === "last_measured_mask") {
          const original = Number.parseInt(value, 16) || 0;
          return `${key}=0x${(original & allowedMask).toString(16).toUpperCase()}`;
        }
        if (key === "last_no_response_mask") {
          const original = Number.parseInt(value, 16) || 0;
          return `${key}=0x${((original | (fullMask ^ allowedMask)) & fullMask).toString(16).toUpperCase()}`;
        }
        if (key === "recovered_mask" || key === "last_recovered_mask") {
          const original = Number.parseInt(value, 16) || 0;
          return `${key}=0x${(original & allowedMask).toString(16).toUpperCase()}`;
        }
        return token;
      }).join(",");
    }
    if (line.startsWith("GATEWAY,")) {
      return line.split(",").filter((token, index) => {
        if (index < 2 || !token.includes("=")) return true;
        const key = token.split("=", 1)[0];
        return pairIsAllowed(key);
      }).join(",");
    }
    const tableMatch = line.match(/^(\|\s*)([A-F]{2})(\s*\|\s*)([^|]+)(\s*\|\s*)([^|]+)(\s*\|\s*)([^|]+)(\s*\|)/);
    if (tableMatch && !pairIsAllowed(tableMatch[2])) {
      return `${tableMatch[1]}${tableMatch[2]}${tableMatch[3]}nan${tableMatch[5]}nan${tableMatch[7]}TIMEOUT${tableMatch[9]}nan${line.slice(tableMatch[0].length)}`;
    }
    return line;
  }).join("\n");
}

const sourceLogs = (await fs.readdir(sourceLogDir)).filter((name) => /^device-monitor-260919-\d{6}\.log$/i.test(name)).sort();
if (sourceLogs.length !== 6) throw new Error(`Expected 6 logs for 260919, found ${sourceLogs.length}`);

const summaries = [];
for (const sourceName of sourceLogs) {
  const text = await fs.readFile(path.join(sourceLogDir, sourceName), "utf8");
  summaries.push({ sourceName, text, summary: parseSummary(text) });
}

const fiveRemovals = ["C", "D", "E", "F"];
const fourRemovals = ["CD", "CE", "CF", "DE", "DF", "EF"];
const variants = [];
for (const item of summaries) {
  for (const removed of fiveRemovals) variants.push({ ...item, nodeCount: 5, removed, activeNodes: new Set(nodeLabels.filter((node) => node !== removed)) });
  for (const removed of fourRemovals) variants.push({ ...item, nodeCount: 4, removed, activeNodes: new Set(nodeLabels.filter((node) => !removed.includes(node))) });
}

for (const variant of variants) {
  const base = variant.sourceName.replace(/^device-monitor-/, "made-device-monitor-").replace(/\.log$/i, "");
  const name = `${base}-${variant.nodeCount}node-remove${variant.removed}.log`;
  await fs.writeFile(path.join(madeDir, name), transformLog(variant.text, variant.activeNodes), "utf8");
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourceWorkbook));
const sheet = workbook.worksheets.items[0];
const used = sheet.getUsedRange();
const existingValues = used.values;
let lastNonEmptyRow = -1;
for (let r = 0; r < existingValues.length; r++) if (existingValues[r].some((value) => value !== null && value !== "")) lastNonEmptyRow = r;
let startRow = lastNonEmptyRow + 3;
const blockHeight = 15;
const templateStart = Math.max(0, startRow - blockHeight);

for (const variant of variants) {
  const start = startRow;
  const end = start + blockHeight - 1;
  const base = variant.sourceName.replace(/^device-monitor-/, "made-device-monitor-").replace(/\.log$/i, "");
  const name = `${base}-${variant.nodeCount}node-remove${variant.removed}.log`;
  sheet.getRange(`A${start + 1}:P${end + 1}`).copyFrom(sheet.getRange(`A${templateStart + 1}:P${templateStart + blockHeight}`), "all");
  const rows = Array.from({ length: blockHeight }, () => Array(16).fill(null));
  rows[0][0] = "로그";
  rows[0][1] = name;
  rows[1][0] = "노드 상태";
  rows[2] = ["node", ...nodeLabels, ...Array(9).fill(null)];
  rows[3] = ["active", ...nodeLabels.map((node) => variant.activeNodes.has(node) ? 1 : 0), ...Array(9).fill(null)];
  rows[5][0] = "환경";
  rows[6] = [null, ...pairLabels];
  rows[7] = ["pair", ...pairLabels.map((label) => variant.activeNodes.has(label[0]) && variant.activeNodes.has(label[1]) ? 1 : null)];
  rows[8] = ["distance", ...pairLabels.map((label) => variant.activeNodes.has(label[0]) && variant.activeNodes.has(label[1]) ? variant.summary.distances[label] : null)];
  rows[9][0] = "los_nlos";
  rows[10][0] = "wall";
  rows[11][0] = "door";
  rows[12][0] = "corner";
  sheet.getRange(`A${start + 1}:P${end + 1}`).values = rows;
  startRow = end + 1;
}

workbook.recalculate();
const checkRows = sheet.getRange(`A${lastNonEmptyRow + 4}:P${startRow}`).values;
const newLogCount = checkRows.filter((row) => String(row[0] ?? "").length === 2 && row[1] !== null && row[1] !== "").length;
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(JSON.stringify({ sourceLogs: sourceLogs.length, variants: variants.length, fiveNode: sourceLogs.length * fiveRemovals.length, fourNode: sourceLogs.length * fourRemovals.length, workbookNewBlocks: newLogCount, outputWorkbook, errorScan: errors.ndjson }, null, 2));

const firstPreview = await workbook.render({ sheetName: sheet.name, range: `A${lastNonEmptyRow + 4}:P${lastNonEmptyRow + 49}`, scale: 1, format: "png" });
await fs.writeFile(path.join(projectDir, "tmp", "19th_variants_preview.png"), new Uint8Array(await firstPreview.arrayBuffer()));
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputWorkbook);
