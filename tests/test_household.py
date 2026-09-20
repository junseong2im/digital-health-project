import numpy as np
import pandas as pd
import pytest
from metabolic.household import living_alone,household_features,HouseholdPatient
from metabolic.comparison import youden_threshold,preprocessor,restore_parts,paired_ci

def test_household_codes_and_unknowns():
    v=living_alone(pd.Series([1,2,6,9,np.nan,0,7]))
    assert v.iloc[:3].tolist()==[1.,0.,0.]
    assert v.iloc[3:].isna().all()
    with pytest.raises(ValueError):HouseholdPatient(age=29,sex=1,HE_BMI=22,cfam=9)
    assert HouseholdPatient(age=29,sex=1,HE_BMI=22,cfam=None).cfam is None

def test_youden_finite_and_threshold_role():
    y=np.array([0,0,1,1]);p=np.array([.1,.3,.4,.9]);w=np.ones(4)
    assert youden_threshold(y,p,w)==.4

def test_household_feature_separation():
    d=pd.DataFrame({'age':[25,35], 'HE_BMI':[20.,25.],'HE_wc':[70.,90.],'HE_ht':[170.,180.],
        'sex':[1,2],'incm':[1,2],'edu':[3,4],'sm_presnt':[0,1],'dr_month':[1,0],
        'pa_aerobic':[1,0],'cfam':[1,9],'HE_glu':[100,99]})
    x=household_features(d,True)
    assert x.living_alone.iloc[0]==1 and pd.isna(x.living_alone.iloc[1])
    assert 'HE_glu' not in x and 'cfam' not in x
    assert 'living_alone' not in household_features(d,False)
    pre=preprocessor(True).fit(x)
    assert np.isfinite(pre.transform(x)).all()

def test_split_restoration_refuses_leakage():
    import hashlib
    d=pd.DataFrame({'ID':['a','b'],'survey_year':[2021,2021],'group':['A','A']})
    ids=[hashlib.sha256(f'2021:{v}'.encode()).hexdigest() for v in ['a','b']]
    with pytest.raises(ValueError,match='PSU leakage'):restore_parts(d,{'train':[ids[0]],'test':[ids[1]]})

def test_paired_bootstrap_identical_predictions_zero():
    d=pd.DataFrame({'survey_year':[2024]*8,'kstrata':[1]*4+[2]*4,
        'group':['A','A','B','B','C','C','D','D'],'target':[0,1]*4,'wt_itvex':[1]*8})
    p=np.array([.1,.8,.2,.7,.3,.9,.4,.6])
    r=paired_ci(d,p,p,repeats=5)
    for item in r.values():assert item['ci95']==[0.,0.]

def test_household_checkpoint_and_http():
    from pathlib import Path
    from metabolic.household import HouseholdPredictor
    from metabolic.api import app
    from fastapi.testclient import TestClient
    base=Path(__file__).resolve().parents[1]/'artifacts/comparison_household_v1/models'
    if not (base/'household_seed42/network.pt').exists():pytest.skip('Run comparison first')
    predictor=HouseholdPredictor(base/'household_seed42')
    state={'age':29,'sex':1,'HE_BMI':24.,'cfam':1}
    r=predictor.predict(state)
    assert r['household_group']=='single_person' and r['exploratory_extension']
    assert predictor.predict(state|{'cfam':2})['household_group']=='multi_person'
    with pytest.raises(ValueError):predictor.predict(state|{'cfam':None})
    if (base/'specialist_living_1/network.pt').exists():
        with pytest.raises(ValueError):HouseholdPredictor(base/'specialist_living_1').predict(state|{'cfam':2})
    with TestClient(app) as client:
        assert client.post('/v1/decide-household',json={'state':state}).json()==r
        assert client.post('/v1/decide-household',json={'state':state,'policy':'youden'}).status_code==200
        assert client.post('/v1/decide-household',json={'state':state,'policy':'madeup'}).status_code==422
        assert client.post('/v1/decide-household',json={'state':state|{'cfam':None}}).status_code==422
        assert client.post('/v1/decide-household',json={'state':state|{'HE_glu':100}}).status_code==422
