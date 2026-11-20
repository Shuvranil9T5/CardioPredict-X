from functools import lru_cache
import json, os, io
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine,text
from ml.dataset import ROOT,FEATURES,NUM,CAT
app=FastAPI(title='CardioPredict-X',version='1.0')
engine=create_engine('sqlite:///'+str(ROOT/'artifacts/history.db'))
class Patient(BaseModel):
    age:float|None=Field(None,ge=1,le=120)
    trestbps:float|None=Field(None,gt=0,le=300)
    chol:float|None=Field(None,gt=0,le=1000)
    thalch:float|None=Field(None,gt=0,le=300)
    oldpeak:float|None=Field(None,ge=-10,le=20)
    ca:float|None=Field(None,ge=0,le=4)
    sex:str|None=None; cp:str|None=None; fbs:str|None=None; restecg:str|None=None; exang:str|None=None; slope:str|None=None; thal:str|None=None
class Request(BaseModel):
    patient:Patient
    threshold:float=Field(.5,gt=0,lt=1)
    save:bool=False
class Simulation(BaseModel):
    original:Patient; modified:Patient
@lru_cache
def bundle():
    path=ROOT/'artifacts/model.joblib'
    if not path.exists(): raise HTTPException(503,'Train the model first: python -m ml.train')
    return joblib.load(path)
def frame(p):
    d=p.model_dump(); return pd.DataFrame([{k:(str(d[k]).lower() if k in CAT and d[k] is not None else d[k] if d[k] is not None else np.nan) for k in FEATURES}])
def predict(r):
    b=bundle(); x=frame(r.patient); p=float(b['model'].predict_proba(x)[0,1]); samples=[m.predict_proba(x)[0,1] for m in b['ensemble']]; lo,hi=np.quantile(samples,[.05,.95]); warnings=[]
    for c in NUM:
        if pd.notna(x[c][0]) and not b['ranges'][c][0]<=x[c][0]<=b['ranges'][c][1]: warnings.append(c+' outside training range')
    for c in CAT:
        if pd.notna(x[c][0]) and x[c][0] not in b['categories'][c]: warnings.append(c+' unseen category')
    missing=int(x.isna().sum(axis=1).iloc[0]); abstain=bool(warnings or missing>3 or lo<=r.threshold<=hi); result={'score':p,'interval':[float(lo),float(hi)],'classification':'abstain' if abstain else ('present' if p>=r.threshold else 'absent'),'warnings':warnings,'missing':missing,'disclaimer':'Research classification only; not diagnosis or future risk. Ensemble spread is not a calibrated confidence interval.'}
    if r.save and os.getenv('ENABLE_HISTORY')=='1':
        with engine.begin() as conn:
            conn.execute(text('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, result TEXT)')); conn.execute(text('INSERT INTO history(result) VALUES (:r)'),{'r':json.dumps(result)})
    return result
@app.get('/health')
def health(): return {'status':'ok','model_available':(ROOT/'artifacts/model.joblib').exists()}
@app.post('/api/v1/predict')
def route_predict(r:Request): return predict(r)
@app.post('/api/v1/explain')
def explain(r:Request):
    import shap
    b=bundle(); bg=b['background'].copy(); x=frame(r.patient); combined=pd.concat([bg,x],ignore_index=True); encoded=combined.copy(); mapping={}
    for c in CAT:
        cats=list(dict.fromkeys(combined[c].dropna().astype(str))); mapping[c]=cats; encoded[c]=combined[c].map({v:i for i,v in enumerate(cats)}).fillna(-1)
    def f(arr):
        z=pd.DataFrame(arr,columns=FEATURES)
        for c in CAT: z[c]=z[c].map(lambda v:mapping[c][int(v)] if pd.notna(v) and 0<=int(v)<len(mapping[c]) else np.nan)
        return b['model'].predict_proba(z)[:,1]
    ex=shap.KernelExplainer(f,encoded.iloc[:-1].astype(float)); values=ex.shap_values(encoded.iloc[-1:].astype(float),nsamples=128); return {'base':float(ex.expected_value),'contributions':dict(zip(FEATURES,np.asarray(values).reshape(-1).tolist()))}
@app.post('/api/v1/simulate')
def simulate(r:Simulation):
    a=predict(Request(patient=r.original)); b=predict(Request(patient=r.modified)); return {'original':a,'modified':b,'difference_pp':100*(b['score']-a['score'])}
@app.get('/api/v1/metrics')
def metrics():
    p=ROOT/'artifacts/metrics.json'
    if not p.exists(): raise HTTPException(503,'Train first')
    return json.loads(p.read_text())
@app.get('/api/v1/models')
def models(): return metrics()['leaderboard']
@app.get('/api/v1/dataset/statistics')
def stats():
    p=ROOT/'artifacts/dataset.csv'
    if not p.exists(): raise HTTPException(503,'Train first')
    d=pd.read_csv(p); return {'records':len(d),'missing':d.isna().sum().to_dict(),'centers':d.dataset.value_counts().to_dict()}
@app.get('/api/v1/history')
def history():
    with engine.begin() as c:
        c.execute(text('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, result TEXT)')); return [dict(x) for x in c.execute(text('SELECT * FROM history ORDER BY id DESC')).mappings()]
@app.delete('/api/v1/history')
def clear():
    history()
    with engine.begin() as c: c.execute(text('DELETE FROM history'))
    return {'deleted':True}
@app.post('/api/v1/report')
def report(r:Request):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    result=predict(r.model_copy(update={'save':False})); buf=io.BytesIO(); s=getSampleStyleSheet(); SimpleDocTemplate(buf).build([Paragraph('CardioPredict-X | Research report',s['Title']),Spacer(1,20)]+[Paragraph(f'{k}: {v}',s['BodyText']) for k,v in result.items()]); buf.seek(0); return StreamingResponse(buf,media_type='application/pdf')
