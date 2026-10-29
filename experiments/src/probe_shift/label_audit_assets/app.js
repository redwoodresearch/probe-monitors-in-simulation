'use strict';
const $ = id => document.getElementById(id);
const node = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
const labelText = value => value === 1 ? 'credential' : value === 0 ? 'no credential' : String(value).replaceAll('_', ' ');
const labelClass = value => value === 1 || value === 'sabotage' ? 'positive' : value === 'unclear' ? 'unclear' : 'negative';
const shortModel = name => name.split('/')[1].replace('-preview', '');
let manifest = [], visible = [], selected = null, revision = 0;
async function getJSON(path) { const response = await fetch(path, {cache:'no-store'}); if (!response.ok) throw new Error(`Request failed (${response.status})`); return response.json(); }
function block(parent, title, text) { const section = node('section', undefined, 'block'); section.append(node('h3', title), node('pre', typeof text === 'string' ? text || '(empty output)' : JSON.stringify(text, null, 2))); parent.append(section); }
function expandable(parent, title, text) { const d = node('details'); d.append(node('summary', title)); d.addEventListener('toggle', () => { if (d.open && d.childElementCount === 1) d.append(node('pre', typeof text === 'string' ? text : JSON.stringify(text, null, 2))); }); parent.append(d); }
function filters() {
  const kind=$('kind').value, filter=$('filter').value, label=$('label').value, query=$('search').value.trim().toUpperCase();
  visible=manifest.filter(c => c.kind===kind && c.audit_id.includes(query) && (label==='all' || labelClass(c.panel_label)===label) && (filter==='all' || (filter==='disagreement' && c.disagreement) || (filter==='sample' && c.sample) || (filter==='mismatch' && c.audit && !c.audit.agree) || (filter==='unsure' && c.unsure) || (filter==='missing' && c.missing_votes>0)));
  $('count').textContent=`${visible.length.toLocaleString()} matching rows`;
  if (!visible.some(c=>c.audit_id===selected)) selected=visible[0]?.audit_id ?? null;
  renderRows(); show(selected);
}
function renderRows() {
  const fragment=document.createDocumentFragment();
  for (const c of visible) {
    const b=node('button', undefined, `row${c.audit_id===selected?' active':''}`); b.setAttribute('aria-pressed', c.audit_id===selected?'true':'false');
    const top=node('div', undefined, 'row-top'); top.append(node('span', c.audit_id,'row-id'), node('span',labelText(c.panel_label),`tag ${labelClass(c.panel_label)}`)); b.append(top);
    const parts=[`${Object.keys(c.votes).length}/5 votes`]; if(c.sample)parts.push('sample');if(c.audit&&!c.audit.agree)parts.push('audit mismatch');
    b.append(node('div',parts.join(' · '),'row-meta')); b.addEventListener('click',()=>{selected=c.audit_id;renderRows();show(selected);});fragment.append(b);
  }
  $('rows').replaceChildren(fragment);
  const index=visible.findIndex(c=>c.audit_id===selected);$('prev').disabled=index<=0;$('next').disabled=index<0||index>=visible.length-1;
}
async function show(id) {
  const myRevision=++revision, detail=$('detail');detail.replaceChildren(node('p',id?'Loading evidence…':'No rows match these filters.','empty'));if(!id)return;
  try {
    const c=visible.find(row=>row.audit_id===id), e=await getJSON(`/case/${encodeURIComponent(id)}`);if(myRevision!==revision)return;
    detail.replaceChildren();const heading=node('div',undefined,'case-heading');heading.append(node('h2',id),node('span',`Panel: ${labelText(c.panel_label)}`,`tag ${labelClass(c.panel_label)}`));detail.append(heading);
    const votes=node('div',undefined,'votes');for(const [model,v] of Object.entries(c.votes)){const el=node('div',undefined,'vote');el.append(node('strong',shortModel(model)),node('span',`${labelText(v.label)} · ${v.confidence}`));votes.append(el);}detail.append(votes);
    if(c.missing_votes)detail.append(node('p',`${c.missing_votes} missing judgment(s); failures are not negative votes.`,'muted'));
    if(c.audit){const a=node('section',undefined,`audit${c.audit.agree?'':' different'}`);a.append(node('h3',`Independent audit: ${labelText(c.audit.label)} · ${c.audit.agree?'agrees':'differs'}`),node('p',c.audit.reason),node('span','Decision frozen before panel votes were revealed.','muted'));detail.append(a);}
    if(e.kind==='credential')block(detail,'Reader-visible tool output',e.output);
    else {block(detail,'Hidden side task',e.hidden_side_task);block(detail,`Target command · ${e.target_call.function}`,e.target_call.arguments);block(detail,'Recorded result',e.target_call.matching_tool_output);if(e.target_call.tool_error)block(detail,'Recorded tool error',e.target_call.tool_error);expandable(detail,'Cover task',e.cover_task);expandable(detail,`Complete command context (${e.context_call_count} calls; target index ${e.target_call_position})`,e.complete_sequence);}
  } catch(error){if(myRevision===revision)detail.replaceChildren(node('p',error.message,'empty'));}
}
for(const id of ['kind','filter','label'])$(id).addEventListener('change',filters);$('search').addEventListener('input',filters);
for(const [id,delta] of [['prev',-1],['next',1]])$(id).addEventListener('click',()=>{const index=visible.findIndex(c=>c.audit_id===selected)+delta;if(index>=0&&index<visible.length){selected=visible[index].audit_id;renderRows();show(selected);}});
Promise.all([getJSON('/manifest.json'),getJSON('/summary.json')]).then(([rows,summary])=>{
  manifest=rows;for(const kind of ['sabotage','credential']){const s=summary.results[kind],card=node('article',undefined,'card');card.append(node('h3',kind==='sabotage'?'Sabotage · Independent agreement':'Credentials · Independent agreement'),node('strong',`${s.agree}/${s.rows} · ${(100*s.agreement).toFixed(0)}%`),node('span',`95% Wilson interval ${(100*s.wilson_95_interval[0]).toFixed(1)}–${(100*s.wilson_95_interval[1]).toFixed(1)}%; row-level, no clustering adjustment.`,'muted'));$('summary').append(card);}filters();
}).catch(error=>{$('count').textContent='Load failed';$('detail').replaceChildren(node('p',error.message,'empty'));});
