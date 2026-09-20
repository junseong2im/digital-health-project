import json
import numpy as np
import pandas as pd
import pytest
import torch
from metabolic.expanded import extended_features,ExpandedPreprocessor,FactorialNet,ExpandedPredictor,ExpandedPatient,EXTRA

def fixture():
    row={'age':29,'sex':1,'incm':2,'edu':4,'HE_BMI':24.,'HE_wc':85.,'HE_ht':175.,
        'sm_presnt':0,'dr_month':0,'pa_aerobic':1,'cfam':1,'EC1_1':1,'BP1':2,
        'BD1':1,'BD1_11':8,'BD2_1':8,'BE3_31':1,'BE5_1':6,'BE8_1':8,'BE8_2':30,'HE_fh':1}
    return pd.DataFrame([row,row,row])

def test_sentinel_codes_and_structural_non_drinking():
    d=fixture();d.loc[1,['BD1','BD1_11','BD2_1']]=[9,8,8]
    d.loc[1,['BE3_31','BE5_1','BE8_1','HE_fh','EC1_1']]=[99,9,99,9,9]
    x=extended_features(d)
    assert x.loc[0,'alcohol_frequency']==0 and x.loc[0,'alcohol_amount']==0
    assert pd.isna(x.loc[1,'alcohol_frequency']) and pd.isna(x.loc[1,'alcohol_amount'])
    for key in ['walking_days','strength_days','sedentary_hours','family_history','employed']:assert pd.isna(x.loc[1,key])
    assert x.loc[0,'sedentary_hours']==8.5 and x.loc[0,'walking_days']==0 and x.loc[0,'strength_days']==5
    d.loc[2,['BE8_1','BE8_2']]=[24,30]
    assert pd.isna(extended_features(d).loc[2,'sedentary_hours'])

def test_expanded_preprocessing_training_only_and_roundtrip(tmp_path):
    x=extended_features(fixture()).to_numpy();pre=ExpandedPreprocessor(True).fit(x[:2])
    x[2,10]=20
    assert pre.mean[0]==8.5
    for row in x:np.testing.assert_array_equal(pre.transform(row),pre.transform(row[None,:])[0])
    pre.save(tmp_path);loaded=ExpandedPreprocessor.load(tmp_path)
    np.testing.assert_array_equal(loaded.transform(x),pre.transform(x))
    assert set(sum(pre.groups,[]))==set(range(pre.output_dim))

@pytest.mark.parametrize('multi',[False,True])
def test_export_and_untrained_task_protection(tmp_path,multi):
    x=extended_features(fixture()).to_numpy();pre=ExpandedPreprocessor(True).fit(x);z=pre.transform(x).astype('float32')
    net=FactorialNet(pre,multi).eval();pre.save(tmp_path)
    np.savez(tmp_path/'network.npz',**{k:v.detach().numpy() for k,v in net.state_dict().items()})
    (tmp_path/'config.json').write_text(json.dumps({'name':'test','multitask':multi,'temperature':1.,'pattern_temperature':1.,'thresholds':{'youden':.5}}))
    predictor=ExpandedPredictor(tmp_path)
    with torch.inference_mode():expected=net(torch.tensor(z)).numpy()
    np.testing.assert_allclose(predictor.logits(z),expected,atol=1e-6)
    result=predictor.predict({'age':29,'sex':1,'HE_BMI':24.,'living_alone':1})
    if multi:assert 'component_probabilities' in result
    else:assert result['component_predictions_available'] is False and 'component_probabilities' not in result

def test_no_lab_fields_and_invalid_codes_in_serving():
    with pytest.raises(ValueError):ExpandedPatient(age=29,sex=1,HE_BMI=24.,HE_glu=100)
    with pytest.raises(ValueError):ExpandedPatient(age=29,sex=1,HE_BMI=24.,stress_level=9)
    with pytest.raises(ValueError):ExpandedPatient(age=29,sex=1,HE_BMI=24.,sedentary_hours=99)

def test_alcohol_consistency_validation():
    with pytest.raises(ValueError):ExpandedPatient(age=29,sex=1,HE_BMI=24.,alcohol_frequency=0,alcohol_amount=5)
    with pytest.raises(ValueError):ExpandedPatient(age=29,sex=1,HE_BMI=24.,dr_month=0,alcohol_frequency=4)
    assert ExpandedPatient(age=29,sex=1,HE_BMI=24.,dr_month=0,alcohol_frequency=1,alcohol_amount=2).alcohol_amount==2

def test_expanded_saved_api():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from metabolic.api import app
    path=Path(__file__).resolve().parents[1]/'artifacts/expanded_multitask_v4/models/expanded_multi_s42'
    if not (path/'network.npz').exists():pytest.skip('Run factorial experiment first')
    state={'age':29,'sex':1,'HE_BMI':24.,'living_alone':1,'walking_days':3,'family_history':1}
    expected=ExpandedPredictor(path).predict(state)
    with TestClient(app) as client:
        assert client.post('/v4/decide-expanded',json={'state':state}).json()==expected
        assert client.post('/v4/decide-expanded',json={'state':state|{'alcohol_amount':8}}).status_code==422
        assert client.post('/v4/decide-expanded',json={'state':state|{'HE_TG':160}}).status_code==422
        assert client.post('/v4/decide-expanded',json={'state':state,'policy':'test_best'}).status_code==422
