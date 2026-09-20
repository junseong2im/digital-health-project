import json
from pathlib import Path
import numpy as np
import pytest
import torch
from metabolic.specialized import MetabolicInteractionNet,SpecializedPredictor,pattern_features,joint_numpy,group_indices
from metabolic.refined import CleanPreprocessor
from metabolic.data import BITS

def test_group_indices_cover_expected_domains():
    body,context,behavior=group_indices(40)
    assert set(body).isdisjoint(context) and set(body).isdisjoint(behavior)
    assert set(body+context+behavior)==set(range(40))
    assert 1 in body and 0 in context and 18 in behavior
    with pytest.raises(ValueError):group_indices(30)

@pytest.mark.parametrize('pairwise',[False,True])
def test_joint_constraints_and_torch_numpy_consistency(pairwise):
    model=MetabolicInteractionNet(40,4,'grouped',pairwise).eval()
    x=torch.randn(30,40)
    with torch.inference_mode():z=model(x);p=model.joint(z).numpy()
    q=joint_numpy(z.numpy(),pairwise)
    np.testing.assert_allclose(p,q,atol=2e-7)
    np.testing.assert_allclose(q.sum(axis=1),1,atol=2e-7)
    assert np.all(q@BITS <= q[:,1:].sum(axis=1)[:,None]+1e-6)
    assert pattern_features(pairwise).shape==(15,10 if pairwise else 4)

@pytest.mark.parametrize('mode',['flat','grouped'])
def test_exported_model_matches_torch(tmp_path,mode):
    raw=np.array([[20,20,70,.42,1,1,3,0,1,1],[30,30,90,.5,2,2,4,1,0,0]],dtype=float)
    pre=CleanPreprocessor('hinge').fit(raw);x=pre.transform(raw).astype('float32')
    model=MetabolicInteractionNet(40,4,mode,True).eval()
    pre.save(tmp_path/'preprocessor.npz')
    np.savez(tmp_path/'network.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    (tmp_path/'config.json').write_text(json.dumps({'name':'test','mode':mode,'pairwise':True,'risk_temperature':1,'pattern_temperature':1,'thresholds':{'youden':.4}}))
    runtime=SpecializedPredictor(tmp_path)
    with torch.inference_mode():expected=model(torch.tensor(x)).numpy()
    np.testing.assert_allclose(runtime.logits(x),expected,atol=1e-6)
    result=runtime.predict({'age':28,'sex':1,'HE_BMI':24},policy='youden')
    assert 0<=result['answers']['risk']['noul']<=1
    assert 0<=result['answers']['burden']['score']<=4

def test_oof_manifest_excludes_own_training_rows_and_test_year():
    root=Path(__file__).resolve().parents[1]
    path=root/'artifacts/specialized_v3/split_manifest.json'
    if not path.exists():pytest.skip('Run specialized training first')
    manifest=json.loads(path.read_text())
    held=[]
    for fold in manifest['folds']:
        assert set(fold['train']).isdisjoint(fold['validation']);held.extend(fold['validation'])
    assert sorted(held)==list(range(3062))
    assert set(manifest['calibration_oof_rows']).isdisjoint(manifest['threshold_oof_rows'])
    assert sorted(manifest['calibration_oof_rows']+manifest['threshold_oof_rows'])==list(range(3062))

def test_saved_specialized_endpoint():
    from fastapi.testclient import TestClient
    from metabolic.api import app
    root=Path(__file__).resolve().parents[1]
    model_path=root/'artifacts/specialized_v3/model'
    if not (model_path/'network.npz').exists():pytest.skip('Run training first')
    model=SpecializedPredictor(model_path)
    state={'age':30,'sex':2,'HE_BMI':24.,'HE_wc':80.,'HE_ht':165.}
    expected=model.predict(state,policy='youden')
    with TestClient(app) as client:
        response=client.post('/v3/decide',json={'state':state,'policy':'youden'})
        assert response.status_code==200 and response.json()==expected
        assert client.post('/v3/decide',json={'state':state|{'HE_TG':155}}).status_code==422
        assert client.post('/v3/decide',json={'state':state|{'age':40}}).status_code==422
        assert client.post('/v3/decide',json={'state':state,'policy':'invalid'}).status_code==422
