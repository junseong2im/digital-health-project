import numpy as np
import pandas as pd
import pytest
import torch
from metabolic.data import targets, features, FEATURES, BITS
from metabolic.decisions import Patient, Question, render_decisions
from metabolic.network import ParallelDecisionNet
from metabolic.train import choose_threshold

def test_target_boundaries_and_missing():
    d=pd.DataFrame({'HE_glu':[100,99,99],'HE_sbp':[129,130,129], 'HE_dbp':[84,84,85],
        'HE_TG':[149,150,149],'HE_HDL_st2':[40,49,50],'sex':[1,2,2]})
    assert targets(d).tolist() == [1,14,2]
    d.loc[0,'HE_glu']=np.nan
    with pytest.raises(ValueError): targets(d)

def test_coherent_parallel_primitives():
    rng=np.random.default_rng(42)
    for _ in range(50):
        p=rng.dirichlet(np.ones(16))
        r=render_decisions(p,.4)
        a=r['answers']
        assert a['risk']['noul'] == pytest.approx(1-p[0])
        assert a['burden']['score'] == pytest.approx(p @ BITS.sum(axis=1))
        assert sum(a['burden']['probabilities'].values()) == pytest.approx(1)
        assert all(v <= a['risk']['noul']+1e-7 for v in r['component_probabilities'].values())
        assert a['screening']['choice'] == ('screen_positive' if 1-p[0]>=.4 else 'screen_negative')
    q={'burden':{'type':'score','concept':'abnormality_count','minimum':0,'maximum':100}}
    p=np.zeros(16);p[15]=1
    assert render_decisions(p,.5,q)['answers']['burden']['score']==100

def test_schema_rejects_invalid_values_and_unsupported_tasks():
    with pytest.raises(ValueError): Patient(age=40,sex=1,HE_BMI=25)
    with pytest.raises(ValueError): Patient(age=30,sex=1,HE_BMI=float('nan'))
    with pytest.raises(ValueError): Patient(age=30,sex=1,HE_BMI=25,HE_glu=105)
    with pytest.raises(ValueError): Question(type='choice',concept='any_metabolic_abnormality',labels=['a','b','c'])
    with pytest.raises(ValueError): Question(type='score',concept='any_metabolic_abnormality')
    with pytest.raises(ValueError): render_decisions(np.zeros(16),.5)
    with pytest.raises(ValueError): Patient(age=30,sex=1)

def test_threshold_selected_without_test_and_sensitivity_met():
    y=np.array([0,1,1,1,1]);p=np.array([.3,.2,.4,.6,.8]);w=np.ones(5)
    assert choose_threshold(y,p,w,.75)==.4

def test_one_forward_batch():
    model=ParallelDecisionNet(12).eval()
    calls=[]
    hook=model.register_forward_hook(lambda *a:calls.append(1))
    p=torch.softmax(model(torch.ones(3,12)),dim=1).detach().numpy()
    hook.remove()
    assert len(calls)==1 and p.shape==(3,16)
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-6)

def test_preprocessing_excludes_target_and_unknown_codes():
    from metabolic.train import preprocessing
    d=pd.DataFrame({'age':[20.,30.,35.], 'HE_BMI':[20.,30.,np.nan], 'HE_wc':[70.,90.,80.],
        'HE_ht':[170.,180.,175.],'sex':[1,2,1],'incm':[1,2,9],'edu':[3,4,9],
        'sm_presnt':[0,1,9],'dr_month':[0,1,9],'pa_aerobic':[1,0,9], 'HE_glu':[90,200,120]})
    x=features(d)
    assert list(x)==FEATURES and 'HE_glu' not in x
    assert np.isnan(x.iloc[2].incm)
    pre=preprocessing().fit(x.iloc[:2])
    altered=d.copy(); altered['HE_glu']=9999
    np.testing.assert_allclose(pre.transform(features(d)),pre.transform(features(altered)))
    assert pre.named_transformers_['numeric']['impute'].statistics_[1]==25

def test_cohort_distinguishes_undiagnosed_and_untreated():
    from metabolic.data import cohort
    row={'age':30,'sex':1,'DI1_dg':0,'DI2_dg':0,'DE1_dg':0,'DI1_2':8,'DI2_2':8,
         'DE1_31':8,'DE1_32':8,'HE_dprg':np.nan,'HE_fst':12,'HE_glu':100,
         'HE_sbp':120,'HE_dbp':70,'HE_TG':110,'HE_HDL_st2':50,'wt_itvex':1,'psu':'A','kstrata':1}
    d=pd.DataFrame([row,row,row,row])
    d.loc[1,'DI1_dg']=1;d.loc[1,'DI1_2']=5
    d.loc[2,'DI1_2']=1
    d.loc[3,'HE_glu']=np.nan
    assert cohort(d,'unaware')[0].index.tolist()==[0]
    assert cohort(d,'untreated')[0].index.tolist()==[0,1]
    d.loc[0,'DE1_dg']=9
    assert cohort(d,'unaware')[0].empty

def test_saved_model_and_api():
    from pathlib import Path
    from metabolic.predict import Predictor
    from metabolic.api import app
    from fastapi.testclient import TestClient
    path=Path(__file__).resolve().parents[1]/'artifacts/local_parallel_v1_run2'
    if not (path/'network.pt').exists():
        pytest.skip('Train the documented reference run to enable integration test')
    state={'age':29,'sex':1,'HE_BMI':25.}
    predictor=Predictor(path)
    result=predictor.predict(state)
    assert 0 <= result['answers']['risk']['noul'] <= 1
    assert result==predictor.predict(state)
    with TestClient(app) as client:
        assert client.get('/health').json()['status']=='loaded'
        response=client.post('/v1/decide',json={'state':state})
        assert response.status_code==200
        assert response.json()['answers']['screening']['choice']==result['answers']['screening']['choice']
        assert response.json()['answers']['risk']['noul']==pytest.approx(result['answers']['risk']['noul'],abs=1e-6)
        assert client.post('/v1/decide',json={'state':state|{'HE_glu':110}}).status_code==422
        assert client.post('/v1/decide',json={'state':state,'questions':{}}).status_code==422
        custom={'risk100':{'type':'score','concept':'abnormality_count','minimum':0,'maximum':100}}
        assert client.post('/v1/decide',json={'state':state,'questions':custom}).status_code==200
