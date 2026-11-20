import argparse, json
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import clone
from sklearn.metrics import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from ml.dataset import *
def pipeline(model):
    return Pipeline([('prep',ColumnTransformer([('num',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)])),('model',model)])
def candidates():
    m={'Logistic regression':LogisticRegression(max_iter=2000),'Random forest':RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=1),'XGBoost':XGBClassifier(n_estimators=120,max_depth=3,n_jobs=1,random_state=42),'LightGBM':LGBMClassifier(n_estimators=120,verbosity=-1,n_jobs=1,random_state=42),'SVM':SVC(probability=True,random_state=42),'Gradient boosting':GradientBoostingClassifier(random_state=42)}
    m['Stacking']=StackingClassifier(estimators=[('lr',clone(m['Logistic regression'])),('rf',clone(m['Random forest'])),('gb',clone(m['Gradient boosting']))],cv=5,n_jobs=1)
    return m
def metrics(y,p):
    a=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,a,labels=[0,1]).ravel(); both=len(set(y))==2
    return dict(accuracy=accuracy_score(y,a),precision=precision_score(y,a,zero_division=0),sensitivity=recall_score(y,a,zero_division=0),specificity=float(tn/max(1,tn+fp)),f1=f1_score(y,a,zero_division=0),roc_auc=roc_auc_score(y,p) if both else None,pr_auc=average_precision_score(y,p) if both else None,mcc=matthews_corrcoef(y,a),balanced_accuracy=balanced_accuracy_score(y,a),brier=brier_score_loss(y,p),n=len(y),confusion=[[int(tn),int(fp)],[int(fn),int(tp)]])
def train(path=None,trials=10):
    d=load(path); X=d[FEATURES]; y=d.target; A,B,a,b=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); cv=StratifiedKFold(5,shuffle=True,random_state=42)
    models=candidates(); board={}; curves={}
    for name,m in models.items():
        scores=cross_val_score(pipeline(m),A,a,cv=cv,scoring='roc_auc',error_score='raise'); board[name]={'cv_auc':float(scores.mean()),'cv_std':float(scores.std())}
    best=max(board,key=lambda k:board[k]['cv_auc']); base=pipeline(models[best]);
    if trials:
        import optuna
        def objective(t):
            # Tune regularization of a logistic calibration-independent candidate.
            p=pipeline(LogisticRegression(C=t.suggest_float('C',.001,100,log=True),max_iter=2000)); return cross_val_score(p,A,a,cv=cv,scoring='roc_auc').mean()
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42)); study.optimize(objective,n_trials=trials)
        if study.best_value>board[best]['cv_auc']:
            best='Tuned logistic regression'; base=pipeline(LogisticRegression(**study.best_params,max_iter=2000)); board[best]={'cv_auc':study.best_value}
    for name,m in models.items():
        fitted=pipeline(m).fit(A,a); p=fitted.predict_proba(B)[:,1]; board[name].update(metrics(b,p)); f,t,_=roc_curve(b,p); pr,re,_=precision_recall_curve(b,p); curves[name]={'fpr':f.tolist(),'tpr':t.tolist(),'precision':pr.tolist(),'recall':re.tolist()}
    final=CalibratedClassifierCV(base,cv=cv,method='sigmoid').fit(A,a); p=final.predict_proba(B)[:,1]; rng=np.random.default_rng(42); ensemble=[]; aucs=[]
    for _ in range(20):
        ix=rng.integers(len(A),size=len(A)); ensemble.append(clone(base).fit(A.iloc[ix],a.iloc[ix]))
    for _ in range(500):
        ix=rng.integers(len(b),size=len(b)); yy=b.iloc[ix];
        if yy.nunique()==2: aucs.append(roc_auc_score(yy,p[ix]))
    external={}
    for center in d.dataset.unique():
        mask=d.dataset==center
        if y[~mask].nunique()==2: external[str(center)]=metrics(y[mask],clone(base).fit(X[~mask],y[~mask]).predict_proba(X[mask])[:,1])
    groups={}; hold=d.loc[B.index]
    for col,vals in [('sex',hold.sex),('age',pd.cut(hold.age,[0,40,60,120])) ,('center',hold.dataset)]:
        for v in vals.dropna().unique():
            mask=(vals==v).to_numpy(); groups[f'{col}: {v}']=metrics(b.iloc[mask],p[mask])
    frac,mean=calibration_curve(b,p,n_bins=8); out={'best':best,'leaderboard':board,'curves':curves,'test':metrics(b,p),'auc_ci':np.quantile(aucs,[.025,.975]).tolist(),'external':external,'groups':groups,'calibration':{'observed':frac.tolist(),'predicted':mean.tolist()},'records':len(d)}
    dest=ROOT/'artifacts'; dest.mkdir(exist_ok=True); joblib.dump({'model':final,'ensemble':ensemble,'background':A.sample(min(25,len(A)),random_state=42),'ranges':{c:[float(A[c].min()),float(A[c].max())] for c in NUM},'categories':{c:A[c].dropna().unique().tolist() for c in CAT}},dest/'model.joblib'); (dest/'metrics.json').write_text(json.dumps(out,indent=2)); d.to_csv(dest/'dataset.csv',index=False); print(json.dumps(out['test'],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--csv'); p.add_argument('--trials',type=int,default=10); args=p.parse_args(); train(args.csv,args.trials)
