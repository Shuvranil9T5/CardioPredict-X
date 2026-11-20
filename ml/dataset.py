from pathlib import Path
import pandas as pd
import numpy as np
NUM=['age','trestbps','chol','thalch','oldpeak','ca']
CAT=['sex','cp','fbs','restecg','exang','slope','thal']
FEATURES=NUM+CAT
ROOT=Path(__file__).resolve().parents[1]
def load(path=None):
    if path is None:
        local=ROOT/'data/raw/heart_disease_uci.csv'
        if local.exists(): path=local
        else:
            import kagglehub
            folder=Path(kagglehub.dataset_download('redwankarimsony/heart-disease-data'))
            path=next(folder.rglob('heart_disease_uci.csv'))
    d=pd.read_csv(path).replace('?',np.nan)
    required=set(FEATURES+['num','dataset'])
    if not required <= set(d): raise ValueError(f'Missing columns: {required-set(d)}')
    d=d.drop(columns=['id'],errors='ignore').drop_duplicates().copy()
    for c in NUM: d[c]=pd.to_numeric(d[c],errors='coerce')
    for c in ['trestbps','chol','thalch','age']: d.loc[d[c]<=0,c]=np.nan
    for c in CAT: d[c]=d[c].map(lambda x: str(x).lower() if pd.notna(x) else np.nan)
    n=pd.to_numeric(d.num,errors='coerce'); d=d[n.isin(range(5))].copy(); d['target']=(n.loc[d.index]>0).astype(int)
    return d
