import {expect,it,vi} from 'vitest'
import {JSDOM} from 'jsdom'
import fs from 'node:fs'
const template=fs.readFileSync('../backend/scripts/factual_review.html','utf8')
function page(){
 const rows=['en','hi','hinglish'].map(language=>({group_id:'G',question_id:'G_'+language,language,domain:'contract_law',question:'Question?',reference_answer:'Reference',factual:{act_id:'ACT',answer_type:'provision',source_version:'v1',split:'test',canonical_answer:{act_id:'ACT',provision_type:'section',values:['12']},supporting_quotes:[{quote:'Source quote'}]}}))
 const pack={run_fingerprint:'test',groups:[{rows,source:{document_metadata:{title:'Act'},document_id:'D',chunk_id:'C',page_start:1,page_end:1,text:'Source quote'}}]}
 const downloads=[]
 const dom=new JSDOM(template.replace('__PACK__',JSON.stringify(pack)),{runScripts:'dangerously',beforeParse(w){w.structuredClone=structuredClone;w.URL.createObjectURL=b=>{downloads.push(b);return 'blob:test'};w.URL.revokeObjectURL=()=>{};w.HTMLAnchorElement.prototype.click=()=>{};}})
 return {dom,downloads,d:dom.window.document,w:dom.window}
}
it('approving then editing a translation clears group approval',()=>{
 const {dom,d,w}=page();const approval=d.querySelector('article input[type=checkbox]');approval.checked=true;approval.dispatchEvent(new w.Event('change'));
 expect(d.getElementById('status').textContent).toContain('1/1 groups approved');
 const question=d.querySelector('.language textarea');question.value='Edited question';question.dispatchEvent(new w.Event('input'));
 expect(approval.checked).toBe(false);expect(d.getElementById('status').textContent).toContain('0/1');dom.window.close()
})
it('export requires named reviewer and explicit group approval',()=>{
 const {dom,d,w,downloads}=page();d.getElementById('export').click();expect(d.getElementById('message').textContent).toContain('reviewer name');
 const reviewer=d.getElementById('reviewer');reviewer.value='Reviewer';reviewer.dispatchEvent(new w.Event('input'));
 d.getElementById('export').click();expect(downloads.length).toBe(0);
 const check=d.querySelector('article input[type=checkbox]');check.checked=true;check.dispatchEvent(new w.Event('change'));d.getElementById('export').click();
 expect(downloads.length).toBe(2);expect(d.getElementById('message').textContent).toContain('Exported 3 rows');dom.window.close()
})
