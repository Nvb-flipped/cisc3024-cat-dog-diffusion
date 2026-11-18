// Synthetic DOM fixtures: no browser, pool, server, or real review log is used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

function fixture(search = '') {
  const nodes = new Map();
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {value: '', checked: false, disabled: false,
      dataset: {}, style: {}, classList: {toggle() {}}, replaceChildren() {}});
    return nodes.get(id);
  }
  node('mode').value = 'calibration';
  const criteria = ['species_correct', 'real_photo'].map(name => ({checked: true, dataset: {criterion: name}}));
  const buttons = ['good', 'bad'].map(verdict => ({dataset: {verdict}}));
  const pending = [];
  const context = vm.createContext({URLSearchParams, location: {search},
    URL: {createObjectURL() { throw new Error('Stale preview must not create a blob URL'); }, revokeObjectURL() {}},
    document: {getElementById: node,
      querySelector(selector) { return selector.startsWith('meta') ? {content: 'synthetic-token'} : criteria[0]; },
      querySelectorAll(selector) {
        if (selector === '[data-criterion]') return criteria;
        if (selector === '[data-verdict]') return buttons;
        return [...nodes.values(), ...criteria, ...buttons];
      }},
    fetch(url, options) { return new Promise(resolve => pending.push({url, options, resolve})); }});
  const source = fs.readFileSync(path.join(__dirname, '../src/curation/review.js'), 'utf8')
    .replace(/\nopenReviewLink\(\)\.catch\(error => message\(error\.message, true\)\);\s*$/, '');
  vm.runInContext(source, context);
  vm.runInContext("items = [{candidate_id:'synthetic-a', pool_id:'fixture'}]; loading = false;", context);
  return {context, pending, node, criteria, run(code) { return vm.runInContext(code, context); }};
}

function response(data) { return {ok: true, async json() { return data; }}; }

test('a shortlist link opens the requested saved crop and batch', async () => {
  const f = fixture('?mode=shortlist&batch=2&candidate=synthetic-b');
  f.run('show = async () => {}');
  const load = f.run('openReviewLink()');
  assert.equal(f.pending[0].url, '/items?mode=shortlist&page=1');
  f.pending[0].resolve(response({items: [{candidate_id: 'synthetic-a'}, {candidate_id: 'synthetic-b'}],
    total: 2, page: 1, batch_count: 2, batch_id: 'v002'}));
  await load;
  assert.equal(f.run('items[position].candidate_id'), 'synthetic-b');
  assert.equal(f.node('page').value, 2);
});

test('two overlapping save calls send only one decision', async () => {
  const f = fixture();
  const first = f.run("save('good')");
  await f.run("save('good')");
  assert.equal(f.pending.length, 1);
  assert.equal(f.pending[0].url, '/decision');
  f.pending[0].resolve(response({candidate_id: 'synthetic-a', verdict: 'good'}));
  await first;
  assert.equal(f.run('items[0].prior.candidate_id'), 'synthetic-a');
  assert.equal(f.run('saving'), false);
});

test('late save response does not mark a newly displayed crop as reviewed', async () => {
  const f = fixture();
  const saving = f.run("save('good')");
  f.run("items = [{candidate_id:'synthetic-b',pool_id:'other'}]; position=0;");
  f.pending[0].resolve(response({candidate_id: 'synthetic-a', verdict: 'good'}));
  await saving;
  assert.equal(f.run('items[0].prior'), undefined);
});

test('late preview for an older crop never replaces the current preview', async () => {
  const f = fixture();
  const preview = f.run('preview()');
  f.run('renderVersion++');
  f.pending[0].resolve({ok: true});
  assert.equal(await preview, false);
  assert.equal(f.run('previewHash'), '');
  assert.equal(f.node('crop').src, undefined);
});

test('corrected geometry requires new measurements and specification checks', () => {
  const f = fixture();
  f.node('head_width').value = '42';
  f.node('occupancy').value = '.7';
  f.run("previewHash = 'old-preview';");
  f.node('size').oninput();
  assert.equal(f.run('previewHash'), '');
  assert.equal(f.node('head_width').value, '');
  assert.equal(f.node('occupancy').value, '');
  assert(f.criteria.every(item => !item.checked));
  assert.equal(f.node('crop').style.visibility, 'hidden');
});

test('typed source width updates occupancy from current side', () => {
  const f = fixture();
  f.node('size').value = '192';
  f.node('head_width').value = '128';
  f.node('head_width').oninput();
  assert.equal(f.node('occupancy').value, '0.667');
  f.node('head_width').value = '';
  f.node('head_width').oninput();
  assert.equal(f.node('occupancy').value, '');
});

test('editing a drafted crop keeps checklist choices and resets final confirmation', () => {
  const f = fixture();
  f.run('items[0].draft = {head_width: 44}; previewHash = "draft-preview";');
  f.criteria[1].checked = false;
  f.node('draftConfirmed').checked = true;
  f.node('head_width').value = '44';
  f.node('size').oninput();
  assert.deepEqual(f.criteria.map(criterion => criterion.checked), [true, false]);
  assert.equal(f.node('draftConfirmed').checked, false);
  assert.equal(f.run('previewHash'), '');
  assert.equal(f.node('head_width').value, '');
  f.node('draftConfirmed').checked = true;
  f.node('label').onchange();
  assert.deepEqual(f.criteria.map(criterion => criterion.checked), [true, false]);
  assert.equal(f.node('draftConfirmed').checked, false);
});

test('saving an agent draft requires an explicit student check', async () => {
  const f = fixture();
  f.run('items[0].draft = {head_width: 44};');
  await f.run("save('good')");
  assert.equal(f.pending.length, 0);
  assert.match(f.node('message').textContent, /personally checked/);
  f.node('draftConfirmed').checked = true;
  const saving = f.run("save('good')");
  assert.equal(f.pending.length, 1);
  assert.equal(JSON.parse(f.pending[0].options.body).draft_confirmed, true);
  f.pending[0].resolve(response({candidate_id: 'synthetic-a', verdict: 'good'}));
  await saving;
});
