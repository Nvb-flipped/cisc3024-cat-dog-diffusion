import fs from 'node:fs/promises';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtimeNode = process.argv[2];
const output = path.resolve(root, process.argv[3] ?? 'private_build/presentation_development');
const python = process.argv[4] ?? path.join(root, '.venvs', 'training', 'Scripts', 'python.exe');
const require = createRequire(path.join(runtimeNode, 'package.json'));
const {Presentation, PresentationFile} = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const resultsPath = 'results/experiments/capacity_32_seed17_v002/comparison.json';
const reference = JSON.parse(await fs.readFile(path.join(root, resultsPath), 'utf8'));
const grid = path.join(output, 'tiles');
const deck = Presentation.create({slideSize: {width: 1280, height: 720}});
const color = {paper: '#FAF9F6', ink: '#162A2D', muted: '#516263', accent: '#187269', light: '#E4EEEA'};

function text(slide, name, value, x, y, width, height, options = {}) {
  const shape = slide.shapes.add({name, geometry: 'textbox',
    position: {left: x, top: y, width, height}, fill: 'none',
    line: {fill: 'none', width: 0}});
  shape.text = value;
  shape.text.style = {typeface: 'Arial', color: color.ink, fontSize: 28,
    wrap: 'square', autoFit: 'none', insets: 0, ...options};
}

function slide(title, notes) {
  const page = deck.slides.add();
  page.background.fill = color.paper;
  if (title) text(page, 'title', title, 72, 58, 1140, 70, {fontSize: 49, bold: true});
  page.speakerNotes.text = notes;
  return page;
}

function point(page, name, heading, body, x, y, width) {
  text(page, `${name}-heading`, heading, x, y, width, 43, {fontSize: 30, bold: true});
  text(page, `${name}-body`, body, x, y + 51, width, 95, {fontSize: 26, color: color.muted});
}

async function picture(page, filename, alt, left, top, side) {
  const blob = await fs.readFile(filename);
  page.images.add({blob, contentType: 'image/png', alt, fit: 'contain',
    position: {left, top, width: side, height: side}});
}

async function tiles(page, species, first, count, x, y, side, stride) {
  for (let index = 0; index < count; index++) {
    const name = `${species}_${String(first + index).padStart(2, '0')}`;
    await picture(page, `${grid}/${name}.png`, `Generated ${species} ${first + index}, official seed 0`,
      x + index * stride, y, side);
  }
}

await fs.mkdir(grid, {recursive: true});
const prepareTiles = spawnSync(python, ['-c', `
import sys
from pathlib import Path
from PIL import Image
original, destination = sys.argv[1:]
with Image.open(original) as source:
    source = source.convert('RGB')
    if source.size != (546, 274):
        raise ValueError('Unexpected official seed-0 grid dimensions')
    for species, top in [('cat', 2), ('dog', 138)]:
        for index in range(10):
            left = 2 + 34 * index
            tile = source.crop((left, top, left + 32, top + 32))
            tile.resize((192, 192), Image.Resampling.NEAREST).save(Path(destination) / f'{species}_{index:02d}.png')
`, path.join(root, resultsPath, '..', 'seed_0.png'), grid], {encoding: 'utf8'});
if (prepareTiles.status !== 0) {
  throw new Error(prepareTiles.stderr || prepareTiles.error?.message || 'Could not prepare exact tiles');
}

const title = slide(null, '20 s. State the target and why the memorization cap matters. The pictured tiles '
  + 'are the first two official seed-0 outputs per class from the v002 development reference. '
  + `Source: ${resultsPath}.`);
text(title, 'course', 'CISC3024  /  Development results', 72, 115, 620, 34,
  {fontSize: 25, color: color.accent, bold: true});
text(title, 'title', 'Generating cat and dog heads', 72, 180, 630, 176,
  {fontSize: 64, bold: true});
text(title, 'task', 'Class-conditional diffusion on 32 × 32 photographs', 72, 393, 610, 95,
  {fontSize: 32, color: color.muted});
await tiles(title, 'cat', 0, 2, 765, 143, 196, 204);
await tiles(title, 'dog', 0, 2, 765, 347, 196, 204);

const data = slide('From source photos to exact crops', '55 s. Explain the student crop and duplicate '
  + 'checks and the side-length correction. v002 is the model reference. v003 adds eight personally '
  + 'approved crops and preserves the validation set. The final personal audit is pending. '
  + 'Sources: manifests/v002.provenance.json, manifests/v003.provenance.json, '
  + 'results/curation/v003_cache_parity.json.');
text(data, 'count', '1,197', 74, 160, 478, 138, {fontSize: 122, color: color.accent, bold: true});
text(data, 'count-caption', 'Exact crops used by the v002 model', 76, 320, 472, 75, {fontSize: 33, bold: true});
text(data, 'counts', '583 cats · 614 dogs\n63 separate pool holdout crops', 76, 417, 480, 86,
  {fontSize: 26, color: color.muted});
point(data, 'geometry', '60–80% head width in the crop',
  'One centered, roughly upright head with ears. Source head width at least 32 px.', 633, 174, 550);
point(data, 'review', 'Personal crop and duplicate review',
  'The reviewer confirmed the species, exact geometry and one kept crop per family.', 633, 357, 550);
text(data, 'expansion', 'v003 has 1,205 training crops and the same holdout.', 76, 611, 1080, 42,
  {fontSize: 25, color: color.muted});

const method = slide('Class-conditional pixel diffusion', '45 s. Describe the forward noise process, '
  + 'noise prediction and null conditioning. Explain how CFG combines conditional and null estimates '
  + 'and how deterministic DDIM returns an image. Primary references: '
  + 'https://arxiv.org/abs/2006.11239, https://arxiv.org/abs/2207.12598, '
  + 'https://arxiv.org/abs/2010.02502.');
point(method, 'train', 'Training', 'Add Gaussian noise to an exact crop and learn to predict it. '
  + 'The network sees the class or a null label.', 74, 179, 515);
point(method, 'sample', 'Sampling', 'Start from seeded noise. CFG combines both predictions, '
  + 'then DDIM removes noise in 50 deterministic steps.', 648, 179, 540);
text(method, 'equation', 'guided noise = null + 2.5 × (conditional − null)', 74, 402, 1104, 72,
  {fontSize: 39, bold: true, color: color.accent});
text(method, 'capacity', '5.08M parameters · 30,000 updates · EMA weights', 74, 552, 1120, 56,
  {fontSize: 30, color: color.muted});

const demo = slide('Live generation', '45 s. Preload the evaluated export using python -m src.release.demo. '
  + 'Enter 0 twice and compare the image hash. Enter 1 and show the change. '
  + 'This slide shows recorded development reference outputs from the official seed-0 grid. '
  + 'Keep the final graded export and rehearsal artifacts paired before presenting.');
text(demo, 'cats', 'Cat · label 0', 74, 163, 490, 49, {fontSize: 30, bold: true});
text(demo, 'dogs', 'Dog · label 1', 658, 163, 490, 49, {fontSize: 30, bold: true});
await tiles(demo, 'cat', 2, 4, 74, 224, 124, 132);
await tiles(demo, 'dog', 2, 4, 658, 224, 124, 132);
await tiles(demo, 'cat', 6, 4, 74, 356, 124, 132);
await tiles(demo, 'dog', 6, 4, 658, 356, 124, 132);
text(demo, 'demo-command', 'Repeat seed 0, then switch to seed 1.', 74, 545, 1130, 48,
  {fontSize: 31, color: color.accent, bold: true});
text(demo, 'demo-caption', 'Shown here are official seed-0 development outputs at 32 × 32.', 74, 612, 1130, 38,
  {fontSize: 23, color: color.muted});

const pct = (value) => `${(100 * value).toFixed(2)}%`;
const performance = slide('Official development evaluation', '45 s. Each seed generates 1,000 crops '
  + 'per class. Explain that more training can lower FID while increasing copying. '
  + 'The 50k baseline had FID 52.90 and 57.80% memorization. '
  + `Sources: ${resultsPath}, results/experiments/baseline_v001_seed17/seed_0.json.`);
const values = [['Seed', 'Accuracy', 'FID', 'Recall', 'Memorization'],
  ...reference.evidence.map(({seed, metrics}) => [String(seed), pct(metrics.accuracy),
    metrics.fid.toFixed(2), pct(metrics.recall), pct(metrics.memorization_rate)]),
  ['Excellent', '≥90%', '≤50', '≥40%', '≤5% cap']];
const table = performance.tables.add({rows: values.length, columns: values[0].length,
  left: 74, top: 160, width: 1128, height: 300, values});
for (let row = 0; row < values.length; row++) {
  for (let col = 0; col < values[0].length; col++) {
    const cell = table.getCell(row, col);
    cell.fill = row === 0 ? color.ink : row === 4 ? color.light : '#FFFFFF';
    cell.text.style = {typeface: 'Arial', fontSize: 27, bold: row === 0 || row === 4,
      color: row === 0 ? '#FFFFFF' : color.ink, alignment: 'center'};
  }
}
text(performance, 'score', '11/15 on each of three generation seeds', 75, 498, 1130, 56,
  {fontSize: 34, bold: true, color: color.accent});
text(performance, 'comparison', 'The 50k full-width baseline flagged 57.80% of its outputs as copies.',
  75, 575, 1130, 65, {fontSize: 25, color: color.muted});

const verification = slide('Verification and remaining checks', '30 s. Describe the fixed-batch diagnostic '
  + 'and seed-isolation checks. The personal data check, model check and results reproduction '
  + 'belong to the student. Insert the authentic independent observation and resulting rule after '
  + 'recording them in AGENT_LOG.md. Sources: results/diagnostics/overfit/result.json, '
  + 'results/experiments/capacity_32_seed17_v002/offline_cpu_reference.json, AGENT_LOG.md.');
point(verification, 'overfit', 'Fixed two-crop diagnostic',
  'Probe noise MSE fell from 1.15542 to 0.01722 in 2,000 updates.', 74, 176, 1080);
point(verification, 'interface', 'Offline CPU and CUDA checks',
  'Fixed seeds repeat. Changing other labels leaves each output intact. Caller RNG is preserved.',
  74, 330, 1080);
point(verification, 'student', 'Personal audit and independent checks',
  'Pending final review and documentation.', 74, 484, 1080);

await fs.mkdir(output, {recursive: true});
for (const [index, page] of deck.slides.items.entries()) {
  const png = await deck.export({slide: page, format: 'png', scale: 1});
  await fs.writeFile(path.join(output, `slide-${index + 1}.png`), new Uint8Array(await png.arrayBuffer()));
}
await (await PresentationFile.exportPptx(deck)).save(path.join(output, 'development.pptx'));
