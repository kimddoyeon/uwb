import fs from 'node:fs/promises';
import { getDocument } from 'file:///C:/Users/dltpd/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdfjs-dist/legacy/build/pdf.mjs';
import canvasModule from 'file:///C:/Users/dltpd/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas/index.js';

const { createCanvas } = canvasModule;
const [pdfPath, startText, endText, prefix, scaleText] = process.argv.slice(2);
const pdf = await getDocument({ data: new Uint8Array(await fs.readFile(pdfPath)), useSystemFonts: true }).promise;
for (let n = Number(startText); n <= Number(endText); n++) {
  const page = await pdf.getPage(n);
  const viewport = page.getViewport({ scale: Number(scaleText || 3) });
  const canvas = createCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height));
  await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;
  const pngPath = `${prefix}${n}.png`;
  await fs.writeFile(pngPath, canvas.toBuffer('image/png'));
  console.log(`${n} ${canvas.width}x${canvas.height}`);
}
