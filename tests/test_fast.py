from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import pytest
from scipy.special import softmax
from metabolic.fast import ArrayPreprocessor,ArrayNetwork,FastPredictor,FastLogistic,patient_vector
from metabolic.predict import Predictor
from metabolic.data import features
from metabolic.decisions import Patient

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/local_parallel_v1_run2'
pytestmark=pytest.mark.skipif(
    not (OLD/'network.pt').exists() or not (ROOT/'artifacts/comparison_household_v1/models/logistic_regression/pipeline.joblib').exists(),
    reason='Run the documented local experiments to create private model artifacts',
)

def test_exported_preprocessor_and_network_match_with_missing_inputs():
    reference=Predictor(OLD);pre=ArrayPreprocessor(reference.preprocessor)
    ck=torch.load(OLD/'network.pt',map_location='cpu',weights_only=True);net=ArrayNetwork(ck['state_dict'])
    rng=np.random.default_rng(7);states=[]
    for i in range(100):
        state={'age':int(rng.integers(19,40)),'sex':int(rng.integers(1,3)),
               'HE_BMI':float(rng.uniform(16,40)),'HE_wc':float(rng.uniform(60,120)),
               'HE_ht':float(rng.uniform(145,190)),'incm':int(rng.integers(1,5)),
               'edu':int(rng.integers(1,5)),'sm_presnt':i%2,'dr_month':i%2,'pa_aerobic':i%2}
        for key in ['incm','edu','HE_wc','HE_ht','sm_presnt','dr_month','pa_aerobic']:
            if rng.random()<.3:state[key]=None
        states.append(state)
    frame=pd.DataFrame(states).astype(float)
    x=reference.preprocessor.transform(features(frame))
    vectors=np.stack([patient_vector(Patient.model_validate(s)) for s in states])
    np.testing.assert_allclose(pre.transform(vectors),x,rtol=1e-12,atol=1e-12)
    with torch.inference_mode():expected=reference.model(torch.tensor(x,dtype=torch.float32)).numpy()
    np.testing.assert_allclose(net.logits(x),expected,rtol=1e-5,atol=3e-6)
    fast=FastPredictor(OLD)
    for state in states[:10]:
        a=fast.predict(state);b=reference.predict(state)
        assert abs(a['answers']['risk']['noul']-b['answers']['risk']['noul'])<1e-6
        assert a['answers']['screening']['choice']==b['answers']['screening']['choice']

def test_fast_logistic_matches_same_model():
    from scipy.special import expit,logit
    path=ROOT/'artifacts/comparison_household_v1/models/logistic_regression'
    fast=FastLogistic(path);bundle=joblib.load(path/'pipeline.joblib')
    patient=Patient(age=30,sex=2,HE_BMI=25)
    frame=pd.DataFrame([patient.model_dump()]).astype(float)
    p=bundle['model'].predict_proba(bundle['preprocessor'].transform(features(frame)))[0,1]
    p=expit(logit(np.clip(p,1e-7,1-1e-7))/fast.config['temperature'])
    np.testing.assert_allclose(fast.probability(patient_vector(patient)),p,atol=1e-12)
