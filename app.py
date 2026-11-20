import os
from html import escape
from pathlib import Path
import pandas as pd
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(page_title='CardioPredict-X',layout='wide',page_icon=None)
theme = st.sidebar.selectbox('Appearance', ['Midnight', 'Daylight'], key='appearance')
cssroot = Path(__file__).resolve().parent / 'assets/css'
css = '\n'.join((cssroot / name).read_text() for name in ['tokens.css','layout.css','components.css','animations.css','responsive.css'])
if theme == 'Daylight': css += (cssroot / 'light.css').read_text()
st.markdown('<style>'+css+'</style>', unsafe_allow_html=True)
API=os.getenv('API_URL','http://localhost:8000'); prefix='/api/v1/'
def call(path,payload=None,method=None):
    try:
        r=requests.request(method or ('POST' if payload is not None else 'GET'),API+path,json=payload,timeout=120); r.raise_for_status(); return r.content if path.endswith('/report') else r.json()
    except requests.RequestException as e: st.error(f'API unavailable: {e}'); return None
pages=['Overview','AI Prediction','Explainability','What-if Simulator','Model Benchmark','Dataset Intelligence','Reliability & Fairness','Reports','About']
st.sidebar.markdown('<div class="brand"><div class="brand-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M3 12h4l2-5 4 10 2-5h6"/><path d="M12 21S2 15 2 8a5 5 0 0 1 10-1 5 5 0 0 1 10 1c0 7-10 13-10 13Z"/></svg></div><div><div class="brand-name">CardioPredict <b>X</b></div><div class="brand-sub">Intelligence platform</div></div></div>',unsafe_allow_html=True)
st.sidebar.caption('WORKSPACE')
page=st.sidebar.radio('Navigation',pages,label_visibility='collapsed',key='navigation')
health=call('/health')
ready=bool(health and health.get('model_available'))
st.sidebar.markdown(f'<div class="status {"ready" if ready else ""}"><span class="dot"></span>{"Model ready" if ready else "Model unavailable"}</div>',unsafe_allow_html=True)
st.sidebar.caption('Educational research prototype')
st.markdown('<div class="page-kicker">Workspace / '+page+'</div>',unsafe_allow_html=True)
import plotly.io as pio
pio.templates['cardio']=go.layout.Template(layout=dict(paper_bgcolor='#ffffff' if theme=='Daylight' else '#111c2e',plot_bgcolor='#ffffff' if theme=='Daylight' else '#111c2e',font=dict(color='#17263b' if theme=='Daylight' else '#dce7f2',family='Inter, Arial'),colorway=['#22d3ee','#818cf8','#f3bb72','#f47b91','#72b9ef'],xaxis=dict(gridcolor='#dce4ef' if theme=='Daylight' else '#263649'),yaxis=dict(gridcolor='#dce4ef' if theme=='Daylight' else '#263649')))
pio.templates.default='cardio' 
def form(key, defaults=None):
    defaults = defaults or {}
    p = {}
    numeric = {'age':('Age · years',1.,120.,55.), 'trestbps':('Resting blood pressure · mmHg',1.,300.,130.), 'chol':('Serum cholesterol · mg/dL',1.,1000.,240.), 'thalch':('Peak heart rate · bpm',1.,300.,150.), 'oldpeak':('ST depression',-10.,20.,1.), 'ca':('Major vessels',0.,4.,0.)}
    categorical = {'sex':('Recorded sex',['male','female']), 'cp':('Chest pain type',['typical angina','atypical angina','non-anginal','asymptomatic']), 'fbs':('Fasting blood sugar > 120 mg/dL',['false','true']), 'restecg':('Resting ECG',['normal','st-t abnormality','lv hypertrophy']), 'exang':('Exercise-induced angina',['false','true']), 'slope':('ST segment slope',['upsloping','flat','downsloping']), 'thal':('Thallium test',['normal','fixed defect','reversable defect'])}
    for i,(title,fields) in enumerate([('Patient demographics',['age','sex']),('Cardiovascular measurements',['trestbps','chol','thalch']),('Clinical symptoms',['cp','exang','fbs']),('Diagnostic parameters',['restecg','oldpeak','slope','ca','thal'])],1):
        with st.container(border=True):
            st.markdown(f'<div class="section-label">0{i} / {title}</div>',unsafe_allow_html=True)
            cols=st.columns(2)
            for j,k in enumerate(fields):
                with cols[j%2]:
                    if k in numeric:
                        label,lo,hi,val=numeric[k]
                        v=defaults.get(k,val)
                        p[k]=st.number_input(label,min_value=lo,max_value=hi,value=None if v is None else float(v),key=key+k,help='Leave blank if the measurement is unavailable. Example values are prefilled for demonstration.')
                    else:
                        label,opts=categorical[k]; opts=[None]+opts
                        v=defaults.get(k)
                        p[k]=st.selectbox(label,opts,index=opts.index(v) if v in opts else 0,key=key+k,format_func=lambda v: 'Unknown' if v is None else v.capitalize())
    count=sum(v is not None for v in p.values()); st.progress(count/13,text=f'{count} of 13 fields populated')
    return p

def result(r):
    if not r: return
    st.subheader('Assessment outcome')
    score=float(r['score']); lo,hi=r['interval']; classification=r['classification']
    a,b=st.columns([1,1])
    with a:
        fig=go.Figure(go.Indicator(mode='gauge+number',value=score*100,number={'suffix':'%'},title={'text':'Disease classification score'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#22D3EE'},'bgcolor':'rgba(0,0,0,0)'})); fig.update_layout(height=320,margin=dict(l=30,r=30,t=65,b=25)); st.plotly_chart(fig,use_container_width=True)
    with b:
        with st.container(border=True):
            st.metric('Model decision',classification.capitalize()); st.metric('Ensemble spread',f'{lo:.1%} – {hi:.1%}'); st.caption('Spread is not a clinical confidence interval.')
            if classification=='abstain': st.warning('Model abstained: insufficient reliability for a classification.')
            for warning in r.get('warnings',[]): st.warning(warning)
            st.caption(f"Missing inputs: {r.get('missing','Unknown')}")
    st.info(r['disclaimer'])
if page=='Overview':
    st.markdown('''<div class="hero"><small>EXPLAINABLE CARDIOVASCULAR AI</small><h1>Intelligence Behind Every Heartbeat.</h1><p>Explainable cardiovascular classification powered by advanced machine learning.</p><svg viewBox="0 0 600 80"><path class="ecg" d="M0 45 H100 L115 35 L130 45 H160 L175 60 L190 5 L205 75 L220 45 H300 L315 35 L330 45 H360 L375 60 L390 5 L405 75 L420 45 H600"/></svg></div>''',unsafe_allow_html=True)
    st.button('Launch AI prediction', on_click=lambda: st.session_state.update(navigation='AI Prediction'))
    if ready:
        m=call(prefix+'metrics')
        if m:
            c=st.columns(4); c[0].metric('Clinical records',m['records']); c[1].metric('Selected model',m['best']); c[2].metric('Test ROC-AUC',round(m['test']['roc_auc'],3)); c[3].metric('Input features',13)
    path=Path(__file__).resolve().parent/'artifacts/dataset.csv'
    if path.exists():
        d=pd.read_csv(path)
        if 'dataset' in d:
            st.subheader('Clinical data coverage')
            counts=d['dataset'].value_counts().rename_axis('Medical center').reset_index(name='Records')
            st.plotly_chart(px.bar(counts,x='Records',y='Medical center',orientation='h',text_auto=True),use_container_width=True)
    if not ready: st.info('Train models to unlock live analytics.')
elif page=='AI Prediction':
    st.title(page); p=form('predict'); threshold=st.slider('Classification threshold',.05,.95,.5); save=st.checkbox('Save result locally (server opt-in required)'); payload={'patient':p,'threshold':threshold,'save':save}
    if st.button('Run AI classification', disabled=not ready, type='primary'):
        with st.spinner('Evaluating model and reliability…'):
            st.session_state['last']=payload; st.session_state['result']=call(prefix+'predict',payload)
            st.session_state.pop('pdf',None)
    result(st.session_state.get('result'))
    if st.session_state.get('result') and st.button('Prepare PDF'):
        pdf=call(prefix+'report',st.session_state['last']);
        if pdf: st.session_state['pdf']=pdf
    if st.session_state.get('pdf'): st.download_button('Download report',st.session_state['pdf'],'cardiopredict.pdf','application/pdf')
elif page=='Explainability':
    st.title('Explainable AI laboratory'); p=form('explain',st.session_state.get('last',{}).get('patient'))
    if st.button('Compute local SHAP',disabled=not ready):
        r=call(prefix+'explain',{'patient':p});
        if r:
            d=pd.DataFrame({'Feature':list(r['contributions']),'SHAP':list(r['contributions'].values())}); st.plotly_chart(px.bar(d,x='SHAP',y='Feature',orientation='h',color='SHAP',color_continuous_scale='RdBu')); st.caption('Associations, not medical causation.')
elif page=='What-if Simulator':
    st.title(page); st.subheader('Original inputs'); a=form('a'); st.subheader('Modified inputs'); b=form('b',a)
    if st.button('Compare scenarios',disabled=not ready):
        r=call(prefix+'simulate',{'original':a,'modified':b});
        if r: st.metric('Change in score',f"{r['difference_pp']:.2f} percentage points"); st.plotly_chart(px.bar(x=['Original','Modified'],y=[r['original']['score'],r['modified']['score']])); st.caption('Not a causal treatment effect.')
elif page=='Model Benchmark':
    st.title(page); m=call(prefix+'metrics');
    if m:
        d=pd.DataFrame(m['leaderboard']).T; st.dataframe(d); st.plotly_chart(px.bar(d,y='cv_auc')); fig=go.Figure(); pr=go.Figure()
        for n,c in m['curves'].items(): fig.add_scatter(x=c['fpr'],y=c['tpr'],name=n); pr.add_scatter(x=c['recall'],y=c['precision'],name=n)
        st.plotly_chart(fig); st.plotly_chart(pr); st.plotly_chart(px.imshow(m['test']['confusion'],text_auto=True)); st.plotly_chart(px.line(x=m['calibration']['predicted'],y=m['calibration']['observed'],markers=True))
elif page=='Dataset Intelligence':
    st.title(page); path=Path(__file__).resolve().parent/'artifacts/dataset.csv'
    if path.exists():
        d=pd.read_csv(path); st.dataframe(d.head(30)); st.plotly_chart(px.histogram(d,x='age',color='dataset')); st.plotly_chart(px.bar(d.isna().sum())); st.plotly_chart(px.imshow(d.select_dtypes('number').corr())); st.plotly_chart(px.histogram(d,x='target',color='dataset'))
    else: st.info('Train first.')
elif page=='Reliability & Fairness':
    st.title(page); m=call(prefix+'metrics');
    if m: st.subheader('Held-out center evaluation'); st.dataframe(pd.DataFrame(m['external']).T); st.subheader('Test subgroup evaluation'); st.dataframe(pd.DataFrame(m['groups']).T); st.write('Bootstrap ROC-AUC interval:',m['auc_ci']); st.warning('Small cohorts (especially n < 30) are unreliable. Center results are conditional on model selection using pooled data.')
elif page=='Reports':
    st.title(page); h=call(prefix+'history');
    if h is not None:
        d=pd.DataFrame(h); st.dataframe(d); st.download_button('Export CSV',d.to_csv(index=False),'history.csv');
        if st.checkbox('Confirm delete all history') and st.button('Delete history'): call(prefix+'history',method='DELETE'); st.rerun()
else: st.title('About CardioPredict-X'); st.markdown('Kaggle / UCI Heart Disease • scikit-learn • FastAPI • Streamlit. Binary classification of historical disease labels. Not a medical device, diagnosis, or prospective risk estimator.')
st.caption('Research use only. Do not use for clinical decisions.')
