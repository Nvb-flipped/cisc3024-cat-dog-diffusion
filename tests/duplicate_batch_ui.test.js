// Local DOM simulation; no browser or personal review endpoint is used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');
const path = require('node:path');

function setup() {
  function node(tag='div') {
    return {tag, value:'', children:[], disabled:false, hidden:false,
      append(...nodes) { this.children.push(...nodes); },
      replaceChildren(...nodes) { this.children=nodes; }};
  }
  const nodes = new Map();
  const field = id => { if(!nodes.has(id))nodes.set(id,node());return nodes.get(id); };
  const rows = Array.from({length:16},(_,i)=>({candidate_id:`fixture-${i}`,pool_id:`fixture-source-${i}`,
    batch:50,label:i%2,x0:0,y0:0,size:64,angle:0,review_status:'pending'}));
  const catalog={batch_count:50,batches:Array.from({length:50},(_,i)=>({batch:i+1,items:rows,
    review_counts:{pending:16,good:0,bad:0,uncertain:0,changed:0}})),
    matches:{'fixture-0':rows.slice(1,4).map(r=>({candidate_id:r.candidate_id}))},
    comparison_items:Object.fromEntries(rows.map(r=>[r.candidate_id,r]))};
  const requests=[];
  const context=vm.createContext({URLSearchParams,location:{search:'?batch=50'},
    document:{getElementById:field,createElement:node,querySelector:()=>({content:'fixture-token'}),querySelectorAll:()=>[]},
    fetch:async(url,options)=>{requests.push({url,options});assert.equal(options,undefined);return {ok:true,json:async()=>catalog};}});
  const html=fs.readFileSync(path.join(__dirname,'../src/curation/duplicate_review.html'),'utf8');
  const script=html.split('<script>')[1].split('</script>')[0].replace('load();loadDraftBatches();','');
  vm.runInContext(script,context);
  return {field,requests,run:code=>vm.runInContext(code,context)};
}

test('all fifty prepared batches show at most eight cards and comparisons never save decisions',async()=>{
  const f=setup();await f.run('loadDraftBatches()');
  assert.equal(f.field('draftBatch').children.length,50);
  assert.equal(f.field('draftBatch').value,'50');
  assert.equal(f.field('draftCards').children.length,8);
  assert.equal(f.field('draft-batches').open,true);
  f.field('draftCards').children[0].children.find(n=>n.tag==='button').onclick();
  assert.equal(f.field('draftCards').children.length,4);
  assert.equal(f.field('draftBack').hidden,false);
  f.field('draftBack').onclick();f.field('draftNext').onclick();
  assert.equal(f.field('draftCards').children.length,8);
  assert.equal(f.field('draftNext').disabled,true);
  assert.deepEqual(f.requests,[{url:'/batches',options:undefined}]);
});
