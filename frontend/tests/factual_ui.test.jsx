import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react'
import FactualResults from '../src/components/FactualResults'
import {api} from '../src/lib/api'
vi.mock('../src/lib/api',()=>({api:{get:vi.fn(),post:vi.fn()},errorMessage:e=>e.message,downloadExperimentCsv:vi.fn(),downloadExperimentJson:vi.fn()}))
vi.mock('recharts',()=>({ResponsiveContainer:({children})=><div>{children}</div>,BarChart:()=>null,Bar:()=>null,CartesianGrid:()=>null,Legend:()=>null,Tooltip:()=>null,XAxis:()=>null,YAxis:()=>null}))
afterEach(()=>{cleanup();vi.clearAllMocks()})
const sys={factual_accuracy:.5,correct_count:1,total_questions:2,retrieval:null,median_seconds:2,p95_seconds:3,failure_rate:0,invalid_output_rate:.5,retry_count:0}
const aggregate={valid_for_reporting:true,label_semantics:['source_chunk'],systems:{llm_only:sys,rag_no_rerank:{...sys,retrieval:{chunk_hit_at_1:.5,chunk_hit_at_3:1,chunk_hit_at_5:1,chunk_mrr:.75}},rag_rerank:{...sys,retrieval:{chunk_hit_at_1:1,chunk_hit_at_3:1,chunk_hit_at_5:1,chunk_mrr:1}}},by_language:{},by_domain:{},paired_comparisons:[]}
it('shows automated factual metrics and an explicit LLM-only retrieval N/A',()=>{
 render(<FactualResults exp={{experiment_id:'E',name:'Factual test',status:'completed',aggregate}} items={[]} refresh={()=>{}}/>);
 expect(screen.getByText(/dedicated factual answer/)).toBeTruthy();expect(screen.getByText(/Source-chunk labels measure/)).toBeTruthy();
 const first=document.querySelector('tbody tr');expect(first.textContent).toContain('LLM-only');expect(first.textContent).toContain('50.0%');expect(first.textContent).toContain('N/A');
 expect(screen.queryByText('Evaluate answers')).toBeNull()
})
it('resumes interrupted requests using the existing experiment id',async()=>{
 api.post.mockResolvedValue({data:{}});const refresh=vi.fn();
 render(<FactualResults exp={{experiment_id:'E',name:'Interrupted',status:'partial'}} items={[]} refresh={refresh}/>);
 fireEvent.click(screen.getByText('Resume interrupted requests'));
 await waitFor(()=>expect(api.post).toHaveBeenCalledWith('/experiments/E/run'));expect(refresh).toHaveBeenCalled()
})
it('surfaces invalid factual output and retrieval diagnosis without manual scoring',()=>{
 const item={question_id:'Q',question:'Question?',language:'hi',reference_answer:'Reference',factual:{canonical_answer:{values:['12']}},scores:{rag_rerank:{valid_output:false,reason:'Invalid factual response'}},rag_rerank:{answer:'wrong format'},retrieval_by_condition:{rag_rerank:{diagnosis:'gold_in_prompt'}}};
 render(<FactualResults exp={{name:'Inspect'}} items={[item]} refresh={()=>{}}/>);
 fireEvent.change(screen.getByLabelText('Question'),{target:{value:'Q'}});
 expect(screen.getByText(/Diagnosis: gold_in_prompt/)).toBeTruthy();expect(screen.getByText(/Invalid factual response/)).toBeTruthy()
})
