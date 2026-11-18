const token = document.querySelector('meta[name="review-token"]').content;
const field = id => document.getElementById(id);
const geometryKeys = ['x0', 'y0', 'size', 'angle'];
let items = [], position = 0, page = 0, total = 0;
let previewHash = '', currentBlob = '', measurement = [], widthBasis = 'student_entry';
let renderVersion = 0, loadVersion = 0, saving = false, loading = false;

function busy(value) {
  loading = value;
  document.querySelectorAll('[data-verdict], #previous, #next, #preview, #unreviewed, #proposedWidth, #geometryPanel input, #proposalFields input, #label')
    .forEach(control => { control.disabled = value || saving; });
  field('mode').disabled = saving;
  field('page').disabled = saving;
}

function message(text, error = false) {
  field('message').textContent = text;
  field('message').className = error ? 'error' : '';
}

function geometry() {
  return Object.fromEntries(geometryKeys.map(key => [key, Number(field(key).value)]));
}

function updateOccupancy() {
  const width = Number(field('head_width').value), side = Number(field('size').value);
  field('occupancy').value = width > 0 && side > 0 ? (width / side).toFixed(3) : '';
}

function isRejected() { return field('mode').value === 'rejected'; }

function updateProgress() {
  const offset = field('mode').value === 'full' ? page * 200 : 0;
  field('progress').textContent = `${offset + position + 1} / ${total} · ${items.filter(x => x.prior).length} saved on this page`;
}

function drawRuler() {
  const image = field('source');
  const svg = field('ruler');
  svg.replaceChildren();
  if (!image.naturalWidth || !items.length) return;
  svg.setAttribute('viewBox', `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  function element(name, attributes) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    svg.append(node);
  }
  if (!isRejected()) {
    const g = geometry();
    element('rect', { x: g.x0, y: g.y0, width: g.size, height: g.size,
      fill: 'none', stroke: '#d57a20', 'stroke-width': 2,
      transform: `rotate(${g.angle} ${g.x0 + g.size/2} ${g.y0 + g.size/2})` });
  }
  for (const point of measurement) {
    element('circle', { cx: point.x, cy: point.y, r: 3, fill: '#145d92' });
  }
  if (measurement.length === 2) {
    element('line', { x1: measurement[0].x, y1: measurement[0].y,
      x2: measurement[1].x, y2: measurement[1].y, stroke: '#145d92', 'stroke-width': 2 });
  }
}

async function preview(version = renderVersion) {
  previewHash = '';
  const params = new URLSearchParams({ pool_id: items[position].pool_id, ...geometry() });
  const response = await fetch('/crop?' + params);
  if (version !== renderVersion) return false;
  if (!response.ok) { message((await response.json()).error, true); return false; }
  const blob = await response.blob();
  if (version !== renderVersion) return false;
  if (currentBlob) URL.revokeObjectURL(currentBlob);
  currentBlob = URL.createObjectURL(blob);
  previewHash = response.headers.get('X-Crop-SHA256');
  field('crop').src = field('native').src = currentBlob;
  field('crop').style.visibility = field('native').style.visibility = 'visible';
  drawRuler();
  return true;
}

async function show() {
  const version = ++renderVersion;
  if (!items.length) { busy(true); message('No proposals available in this review set yet.', true); return; }
  busy(true);
  const item = items[position], prior = item.prior;
  const draft = item.draft;
  previewHash = '';
  measurement = prior?.head_measurement || [];
  widthBasis = prior?.head_width_basis || (draft?.head_width ? 'agent_draft_confirmed' : 'student_entry');
  field('source').style.visibility = 'hidden';
  field('ruler').replaceChildren();
  field('crop').style.visibility = field('native').style.visibility = 'hidden';
  const sourceLoaded = new Promise((resolve, reject) => {
    field('source').onload = () => resolve();
    field('source').onerror = () => reject(new Error('Source could not be loaded. Try this review set again.'));
  });
  // Attach the failure handler immediately, even while awaiting the crop response below.
  sourceLoaded.catch(() => {});
  field('source').src = '/source?pool_id=' + encodeURIComponent(item.pool_id);
  if (isRejected()) {
    field('identity').textContent = `${item.pool_id} · rejected-source diagnostic`;
  } else {
    const kind = item.proposal_origin === 'agent_recovery_draft' ? 'recovery draft' : 'detector proposal';
    field('identity').textContent = `${item.pool_id} · ${kind} ${item.candidate_id} · suggested ${Number(item.label) === 0 ? 'cat' : 'dog'}`;
  }
  updateProgress();
  for (const id of ['cropArea', 'geometryPanel', 'proposalFields', 'proposedActions']) {
    field(id).classList.toggle('hidden', isRejected());
  }
  for (const id of ['rejectedNote', 'rejectedActions']) field(id).classList.toggle('hidden', !isRejected());
  field('reviewTitle').textContent = isRejected() ? 'Check for a missed head' : 'Check the complete specification';
  field('reason').value = prior?.reason ?? draft?.reason ?? '';
  field('reasonCategory').value = prior ? '' : draft?.reason_category ?? '';
  field('draftReview').classList.toggle('hidden', !draft);
  field('draftConfirmed').checked = false;
  field('draftNote').textContent = draft ? draft.note : '';
  field('reviewer').value = field('reviewer').value || prior?.reviewer || '';
  if (!isRejected()) {
    for (const key of geometryKeys) field(key).value = prior?.[key] ?? item[key];
    field('label').value = prior?.label ?? draft?.label ?? item.label;
    field('head_width').value = (prior ? prior.head_width : draft?.head_width) || '';
    updateOccupancy();
    field('proposedWidth').classList.toggle('hidden', !item.head_width);
    field('angle_reason').value = prior?.angle_reason ?? '';
    document.querySelectorAll('[data-criterion]').forEach(input => {
      input.checked = (prior ?? draft)?.criteria?.[input.dataset.criterion] || false;
    });
    const hasHint = item.screening_hint !== null && item.screening_hint !== undefined;
    field('screeningHint').classList.toggle('hidden', !hasHint);
    field('screeningHint').textContent = hasHint ?
      `Screening suggestion: ${Number(item.screening_hint) === 0 ? 'cat' : 'dog'}. Confirm using the photograph.` : '';
    if (!await preview(version)) return;
  }
  await sourceLoaded;
  if (version !== renderVersion) return;
  field('source').style.visibility = 'visible';
  field('crop').style.visibility = field('native').style.visibility = 'visible';
  drawRuler();
  busy(false);
  message(prior ? 'Saved verdict: ' + prior.verdict : 'Awaiting your judgment');
}

async function load(candidateId = null) {
  const version = ++loadVersion;
  ++renderVersion;
  busy(true);
  const mode = field('mode').value;
  const shortlist = mode === 'shortlist';
  field('pageControl').classList.toggle('hidden', mode !== 'full' && !shortlist);
  field('shortlistHelp').classList.toggle('hidden', !shortlist);
  const response = await fetch(`/items?mode=${field('mode').value}&page=${page}`);
  const result = await response.json();
  if (version !== loadVersion) return;
  if (!response.ok) throw new Error(result.error || 'Review set could not be loaded');
  items = result.items; total = result.total;
  const linkedPosition = candidateId ? items.findIndex(item => item.candidate_id === candidateId) : 0;
  position = Math.max(0, linkedPosition);
  page = result.page;
  field('page').value = page + 1;
  field('page').max = Math.max(1, shortlist ? result.batch_count : Math.ceil(total / 200));
  if (shortlist) {
    const batch = result.batch_id;
    const suffix = batch && batch !== 'v001' && /^v\d{3}$/.test(batch) ? `_${batch}` : '';
    field('shortlistFindings').href = `http://127.0.0.1:8769/priority_shortlist${suffix}_findings.html`;
  }
  await show();
  if (version === loadVersion && linkedPosition < 0) message('The linked crop is unavailable in this review batch.', true);
}

async function next() {
  if (position + 1 < items.length) { position++; return show(); }
  if (field('mode').value === 'full' && (page + 1) * 200 < total) { page++; return load(); }
  message('End of this set. Saved decisions remain available when you return.');
}

async function save(verdict) {
  if (!items.length || saving || loading) return;
  const selected = items[position];
  if (selected.draft && !field('draftConfirmed').checked) {
    message('Confirm that you personally checked this draft before saving.', true);
    return;
  }
  saving = true;
  busy(false);
  const data = { candidate_id: selected.candidate_id, reviewer: field('reviewer').value,
    verdict, reason: field('reason').value, ...geometry(), label: Number(field('label').value),
    head_width: Number(field('head_width').value), draft_confirmed: field('draftConfirmed').checked,
    head_width_basis: widthBasis, head_measurement: measurement.length === 2 ? measurement : null,
    angle_reason: field('angle_reason').value, preview_sha256: previewHash,
    criteria: Object.fromEntries([...document.querySelectorAll('[data-criterion]')]
      .map(input => [input.dataset.criterion, input.checked])) };
  try {
    const response = await fetch('/decision', { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Review-Token': token }, body: JSON.stringify(data) });
    const result = await response.json();
    if (!response.ok) { message(result.error, true); return; }
    selected.prior = result;
    if (items[position] !== selected) return;
    updateProgress();
    message('Saved ' + verdict);
    saving = false;
    if (field('advance').checked) await next();
  } catch (error) {
    message(error.message, true);
  } finally {
    saving = false;
    busy(loading);
  }
}

field('source').onclick = event => {
  if (isRejected() || loading || saving) return;
  const image = field('source'), bounds = image.getBoundingClientRect();
  if (measurement.length === 2) measurement = [];
  measurement.push({ x: (event.clientX - bounds.left) * image.naturalWidth / bounds.width,
    y: (event.clientY - bounds.top) * image.naturalHeight / bounds.height });
  if (measurement.length === 2) {
    field('head_width').value = Math.abs(measurement[1].x - measurement[0].x).toFixed(1);
    updateOccupancy();
    widthBasis = 'source_clicks';
    message('Measured source width: ' + field('head_width').value + ' px');
  }
  drawRuler();
};
field('head_width').oninput = () => { widthBasis = 'student_entry'; measurement = []; updateOccupancy(); drawRuler(); };
field('proposedWidth').onclick = () => {
  field('head_width').value = Number(items[position].head_width).toFixed(1);
  updateOccupancy();
  widthBasis = 'proposal_confirmed'; measurement = []; drawRuler();
  message('Proposed width copied. Confirm that it spans the actual head.');
};
field('reasonCategory').onchange = () => { field('reason').value = field('reasonCategory').value; };
field('mode').onchange = () => { page = 0; load().catch(error => message(error.message, true)); };
field('page').onchange = () => { page = Math.max(0, Number(field('page').value) - 1); load(); };
field('preview').onclick = async () => {
  busy(true);
  try { await preview(); } catch (error) { message(error.message, true); }
  finally { busy(false); }
};
field('previous').onclick = () => { position = Math.max(0, position - 1); show(); };
field('next').onclick = next;
field('unreviewed').onclick = () => {
  const index = items.findIndex(item => !item.prior || item.prior.verdict === 'uncertain');
  if (index < 0) message('Every item on this page has a saved decision.');
  else { position = index; show(); }
};
for (const key of geometryKeys) field(key).oninput = () => {
  previewHash = ''; ++renderVersion;
  field('crop').style.visibility = field('native').style.visibility = 'hidden';
  measurement = []; widthBasis = 'student_entry';
  field('head_width').value = field('occupancy').value = '';
  field('draftConfirmed').checked = false;
  const draft = items[position]?.draft;
  if (draft?.reason && field('reason').value === draft.reason) {
    field('reason').value = field('reasonCategory').value = '';
  }
  if (!draft) document.querySelectorAll('[data-criterion]').forEach(input => { input.checked = false; });
  drawRuler();
  message(draft ? 'Geometry changed. Preview the new crop and recheck the retained checklist before saving.' :
    'Geometry changed. Preview and check the corrected crop before saving.');
};
field('label').onchange = () => {
  if (!items[position]?.draft) document.querySelector('[data-criterion="species_correct"]').checked = false;
  field('draftConfirmed').checked = false;
  message('Species changed. Recheck the species criterion using the photograph.');
};
document.querySelectorAll('[data-verdict]').forEach(button => button.onclick = () => save(button.dataset.verdict));
function openReviewLink() {
  const query = new URLSearchParams(globalThis.location?.search || '');
  if (query.get('mode') !== 'shortlist') return load();
  field('mode').value = 'shortlist';
  const batch = Number(query.get('batch'));
  if (Number.isSafeInteger(batch) && batch > 0) page = batch - 1;
  return load(query.get('candidate'));
}
openReviewLink().catch(error => message(error.message, true));
