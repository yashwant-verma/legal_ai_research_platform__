import React from 'react'
import {afterEach,expect,it,vi} from 'vitest'
import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react'
import {MemoryRouter} from 'react-router-dom'
import KnowledgeBase from '../src/pages/KnowledgeBase'
import Experiments from '../src/pages/Experiments'
import {api} from '../src/lib/api'
vi.mock('../src/lib/api',()=>({api:{get:vi.fn(),post:vi.fn(),put:vi.fn()},errorMessage:e=>e.message,downloadExperimentCsv:vi.fn(),downloadExperimentJson:vi.fn()}))
afterEach(()=>{cleanup();vi.clearAllMocks()})
it('upload stays busy while the backend job is processing',async()=>{
 api.get.mockImplementation(async path=>({data:path==='/kb/stats'?{}:path==='/kb/ingest/jobs'?[{job_id:'J',filename:'Constitution.pdf',status:'processing'}]:[]}))
 render(<KnowledgeBase/>);
 await screen.findByText('Constitution.pdf');
 expect(screen.getByRole('button',{name:'Ingesting…'}).disabled).toBe(true)
 expect(screen.getByText(/processing/)).toBeTruthy()
})
it('completed upload displays result and makes another upload available',async()=>{
 api.get.mockImplementation(async path=>({data:path==='/kb/stats'?{}:path==='/kb/ingest/jobs'?[{job_id:'J',filename:'Constitution.pdf',status:'completed',result:{status:'indexed'}}]:[]}))
 render(<KnowledgeBase/>);await screen.findByText(/indexed/);
 expect(screen.getByRole('button',{name:'Ingest and index'}).disabled).toBe(false)
})
it('review submits numeric metrics to the selected question and system',async()=>{
 const exp={experiment_id:'E',name:'Pilot',status:'completed',mode:'compare',completed_questions:1,total_questions:1,created_at:new Date().toISOString()};
 const item={question_id:'Q_EN',question:'Legal test question?',language:'en',domain:'law',reference_answer:'Reference',rag:{answer:'Generated',evidence:[]}};
 api.get.mockImplementation(async path=>({data:path==='/experiments'?[exp]:path.endsWith('/items')?[item]:path==='/datasets'?[]:exp}));api.put.mockResolvedValue({data:{}})
 render(<MemoryRouter><Experiments/></MemoryRouter>);
 await screen.findByText('Pilot');
 const eye=document.querySelector('tbody button');fireEvent.click(eye);
 await screen.findByText('Evaluate answers');
 fireEvent.change(screen.getByLabelText('Question'),{target:{value:'Q_EN'}});
 fireEvent.change(screen.getByLabelText('Evaluator name'),{target:{value:'Reviewer'}});
 fireEvent.change(screen.getByLabelText('correctness'),{target:{value:'1'}});
 fireEvent.click(screen.getByRole('button',{name:'Save evaluation'}));
 await waitFor(()=>expect(api.put).toHaveBeenCalledWith('/experiments/E/items/Q_EN/evaluation/rag',expect.objectContaining({evaluator:'Reviewer',correctness:1,groundedness:null})))
})
