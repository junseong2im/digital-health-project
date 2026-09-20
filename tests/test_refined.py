import numpy as np
import pytest
import torch
from metabolic.refined import CleanPreprocessor,RiskPatternNet,joint_from_logits

def raw():
    return np.array([[20,20,70,.42,1,1,3,0,1,1],[30,30,90,.5,2,2,4,1,0,0],[25,np.nan,80,.45,1,np.nan,2,0,0,1]],dtype=float)

def test_preprocessing_training_only_and_missing_flags(tmp_path):
    x=raw();pre=CleanPreprocessor('hinge').fit(x[:2]);med=pre.median.copy();knots=pre.knots.copy()
    z=pre.transform(x[2]);assert np.isfinite(z).all() and z[5]==1
    _=pre.transform(np.array([39,70,150,.9,2,4,4,1,1,0]))
    np.testing.assert_array_equal(pre.median,med);np.testing.assert_array_equal(pre.knots,knots)
    pre.save(tmp_path/'pre.npz');restored=CleanPreprocessor.load(tmp_path/'pre.npz')
    np.testing.assert_array_equal(pre.transform(x),restored.transform(x))
    x[2,5]=9
    with pytest.raises(ValueError):pre.transform(x)

def test_coherent_primary_conditional_factorization():
    rng=np.random.default_rng(42);z=rng.normal(size=(100,16))*5
    p=joint_from_logits(z,.8,1.2)
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
    from scipy.special import expit
    np.testing.assert_allclose(p[:,1:].sum(axis=1),expit(z[:,0]/.8),atol=1e-12)
    assert (p>=0).all() and (p<=1).all()

def test_array_neural_export_equivalence(tmp_path):
    from metabolic.refined import RefinedPredictor
    import json
    x=raw();pre=CleanPreprocessor().fit(x);z=pre.transform(x).astype('float32')
    net=RiskPatternNet(z.shape[1],8).eval();pre.save(tmp_path/'preprocessor.npz')
    np.savez(tmp_path/'network.npz',**{k:v.detach().numpy() for k,v in net.state_dict().items()})
    (tmp_path/'config.json').write_text(json.dumps({'risk_temperature':1,'pattern_temperature':1,'thresholds':{'youden':.5},'name':'test'}))
    predictor=RefinedPredictor(tmp_path)
    with torch.inference_mode():expected=net(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(predictor.logits(z),expected,atol=1e-6)

def test_scalar_and_batch_preprocessing_identical():
    for basis in ['standard','hinge']:
        pre=CleanPreprocessor(basis).fit(raw())
        for row in raw():np.testing.assert_allclose(pre.transform(row),pre.transform(row[None,:])[0],atol=1e-12)

def test_refined_endpoint_and_probability_invariants():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from metabolic.api import app
    path=Path(__file__).resolve().parents[1]/'artifacts/optimized_v2/model_complete/network.npz'
    if not path.exists():pytest.skip('Run optimization first')
    with TestClient(app) as client:
        r=client.post('/v2/decide',json={'state':{'age':28,'sex':1,'HE_BMI':23},'policy':'youden'})
        assert r.status_code==200
        d=r.json();p=d['answers']['risk']['noul']
        assert 0<=p<=1
        assert max(d['component_probabilities'].values())<=p+1e-7
        assert sum(d['answers']['burden']['probabilities'].values())==pytest.approx(1)
        assert client.post('/v2/decide',json={'state':{'age':28,'sex':1,'HE_BMI':23,'HE_glu':100}}).status_code==422
        assert client.post('/v2/decide',json={'state':{'age':28,'sex':1,'HE_BMI':23},'policy':'test_selected'}).status_code==422
