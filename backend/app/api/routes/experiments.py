from fastapi import APIRouter, BackgroundTasks, HTTPException, Response
from app.schemas.experiment import ExperimentCreate, ManualEvaluationIn
from app.services.experiments import export_bundle, aggregate_experiment, create_experiment, export_csv, get_experiment, get_items, list_experiments, run_experiment, save_manual_evaluation
router=APIRouter(prefix='/experiments',tags=['experiments'])
@router.get('')
def all_experiments(): return list_experiments()
@router.post('')
def create(payload:ExperimentCreate):
    try: return create_experiment(payload)
    except KeyError: raise HTTPException(404,'Dataset not found')
    except ValueError as e: raise HTTPException(422,str(e))
    except Exception as e: raise HTTPException(503,f'Could not freeze experiment: {e}')
@router.post('/{eid}/run',status_code=202)
def run(eid:str,bg:BackgroundTasks):
    try: exp=get_experiment(eid)
    except KeyError: raise HTTPException(404,'Experiment not found')
    if exp.get('evaluation_kind')=='factual':
        from app.services.factual_experiments import is_active
        if is_active(eid): raise HTTPException(409,'Experiment already running in this worker')
    elif exp['status']=='running': raise HTTPException(409,'Experiment already running')
    if exp['status'] in {'completed','blocked'}: raise HTTPException(409,'Create a new experiment; this run is completed or blocked.')
    bg.add_task(run_experiment,eid); return {'experiment_id':eid,'status':'queued'}
@router.get('/{eid}')
def get(eid:str):
    try: return get_experiment(eid)
    except KeyError: raise HTTPException(404,'Experiment not found')
@router.get('/{eid}/items')
def items(eid:str): return get_items(eid)
@router.get('/{eid}/results')
def results(eid:str): return aggregate_experiment(eid)
@router.put('/{eid}/items/{qid}/evaluation/{system}')
def evaluate(eid:str,qid:str,system:str,payload:ManualEvaluationIn):
    try: return save_manual_evaluation(eid,qid,system,payload)
    except ValueError as e: raise HTTPException(422,str(e))
@router.get('/{eid}/export.csv')
def export(eid:str): return Response(export_csv(eid),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{eid}.csv"'})

@router.get('/{eid}/export.json')
def full_export(eid:str):
    import json
    try: body=export_bundle(eid)
    except KeyError: raise HTTPException(404,'Experiment not found')
    return Response(json.dumps(body,ensure_ascii=False,default=str),media_type='application/json',headers={'Content-Disposition':f'attachment; filename="{eid}.json"'})

@router.get('/{eid}/evaluations')
def evaluations(eid:str):
    from app.db.mongo import db
    return list(db.evaluations.find({'experiment_id':eid},{'_id':0}))

@router.get('/{eid}/summary.csv')
def summary_csv(eid:str):
    from app.services.factual_experiments import export_csv as factual_csv
    try:
        exp=get_experiment(eid)
        if exp.get('evaluation_kind')!='factual': raise HTTPException(422,'Summary export requires a factual comparison.')
        return Response(factual_csv(eid,summary=True),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{eid}_summary.csv"'})
    except KeyError: raise HTTPException(404,'Experiment not found')
