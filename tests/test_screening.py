import numpy as np
import pandas as pd
import pytest
from metabolic.screening import screening_metrics,select_screening_threshold,net_benefit,SurveyBootstrap,age_household_groups,interval

def test_weighted_screening_burden_arithmetic():
    y=np.array([0,0,1,1]);p=np.array([.1,.3,.2,.9]);w=np.array([1,2,3,4])
    r=screening_metrics(y,p,.2,w)
    assert r['sensitivity']==1 and r['specificity']==pytest.approx(1/3)
    assert r['ppv']==pytest.approx(7/9) and r['referral_rate']==.9
    assert r['detected_per_1000']==700 and r['unnecessary_referrals_per_1000']==200
    assert r['missed_per_1000']==0 and r['referrals_per_1000']==900
    assert r['unweighted_confusion']=={'tn':1,'fp':1,'fn':0,'tp':2}
    assert net_benefit(y,p,w,.2)==pytest.approx(.65)

def test_no_referral_is_not_infinite_precision():
    r=screening_metrics([0,1],[.1,.9],1)
    assert r['ppv'] is None and r['tests_per_detected'] is None
    assert r['sensitivity']==0 and r['referral_rate']==0

def test_95_target_and_no_threshold_leakage():
    y=np.array([0,1,1,1]);p=np.array([.5,.2,.6,.9]);w=np.ones(4)
    assert select_screening_threshold(y,p,w,.95)==.2
    assert select_screening_threshold(y,p,w,.90)==.2
    with pytest.raises(ValueError):select_screening_threshold(y,p,w,.8)
    with pytest.raises(ValueError):select_screening_threshold(y,p,np.array([1,0,0,0]),.9)
    with pytest.raises(ValueError):screening_metrics(y,p*2,.2)

def test_full_frame_bootstrap_keeps_zero_domain_psus():
    full=pd.DataFrame({'survey_year':[2024]*4,'kstrata':[1,1,2,2],'group':['A','B','C','D']})
    domain=pd.DataFrame({'survey_year':[2024],'kstrata':[1],'group':['A'],'wt_itvex':[1.]})
    boot=SurveyBootstrap(full,domain)
    assert boot.audit['full_frame_psus']==4 and boot.audit['domain_psus']==1
    rng=np.random.default_rng(42);values=[boot.draw(rng)[0] for _ in range(1000)]
    assert set(values)=={0.,2.}
    assert np.mean(values)==pytest.approx(1,abs=.1)

def test_exact_twenties_and_thirties():
    d=pd.DataFrame({'age':[19,20,29,30,39],'living_alone':[1,1,0,1,0],'sex':[1,2,1,2,1]})
    g=age_household_groups(d)
    assert g['age20_29'].tolist()==[False,True,True,False,False]
    assert g['age30_39'].tolist()==[False,False,False,True,True]
    assert not (g['age20_29']&g['age19']).any()
    assert g['age30_39_living1'].sum()==1

def test_interval_reports_insufficient_replicates():
    assert interval([None,np.nan])=={'ci95':None,'valid_repeats':0}
    assert interval([1,2,3])['valid_repeats']==3

def test_screening_saved_api_and_95_policy():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from metabolic.api import app
    model=Path(__file__).resolve().parents[1]/'artifacts/screening_v5/models/expanded_NN/network.npz'
    if not model.exists():pytest.skip('Run the screening experiment first')
    state={'age':24,'sex':1,'HE_BMI':24.,'living_alone':1}
    with TestClient(app) as client:
        r=client.post('/v5/screen',json={'state':state,'policy':'sensitivity95'})
        assert r.status_code==200 and r.json()['development_sensitivity_target']==.95
        assert r.json()['research_only'] is True
        assert 'not a guarantee' in r.json()['validation_notice']
        assert client.post('/v5/screen',json={'state':state,'input_set':'base'}).status_code==200
        assert client.post('/v5/screen',json={'state':state,'policy':'youden'}).status_code==422
        assert client.post('/v5/screen',json={'state':state|{'HE_glu':100}}).status_code==422
