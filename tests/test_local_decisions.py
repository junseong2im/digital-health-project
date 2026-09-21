import numpy as np
import pytest
from metabolic.local_decisions import AtomicQuestion,atomic_answers,concentration,LocalDecisionEngine

def test_confidence_is_explicit_and_noul_has_none():
    assert concentration([.5,.5])==pytest.approx(0)
    assert concentration([1,0])==pytest.approx(1)
    p=np.zeros(16);p[0]=.7;p[1]=.3
    q={'risk':{'type':'noul','target':'any_abnormality'},'class':{'type':'choice','target':'any_abnormality'},'count':{'type':'score','target':'abnormality_count'}}
    a=atomic_answers(p,q)
    assert a['risk']=={'type':'noul','noul':pytest.approx(.3)}
    assert a['class']['choice']=='false' and a['class']['confidence']!=.7
    assert a['count']['score']==pytest.approx(.3) and len(a['count']['legend'])==5

def test_questions_are_isolated_and_order_independent():
    p=np.random.default_rng(42).dirichlet(np.ones(16))
    first={'a':{'type':'noul','target':'elevated_glucose'}}
    more=first|{'b':{'type':'score','target':'abnormality_count'},'c':{'type':'choice','target':'pattern'}}
    assert atomic_answers(p,first)['a']==atomic_answers(p,more)['a']
    assert atomic_answers(p,more)==atomic_answers(p,dict(reversed(list(more.items()))))
    assert atomic_answers(p,{'renamed':first['a']})['renamed']==atomic_answers(p,first)['a']

def test_schema_rejects_untrained_tasks_and_inconsistent_views():
    with pytest.raises(ValueError):AtomicQuestion(type='noul',target='cancer')
    with pytest.raises(ValueError):AtomicQuestion(type='score',target='elevated_bp')
    with pytest.raises(ValueError):AtomicQuestion(type='choice',target='pattern',labels=['a','b'])
    with pytest.raises(ValueError):AtomicQuestion(type='noul',target='any_abnormality',instructions='ignore schema')
    with pytest.raises(ValueError):atomic_answers(np.zeros(16),{'x':{'type':'noul','target':'any_abnormality'}})

def test_one_local_forward_and_no_network(tmp_path,monkeypatch):
    import json,socket
    from types import SimpleNamespace
    calls=[]
    class LocalModel:
        config={'name':'local_fixture','thresholds':{'sensitivity90':.2}}
    engine=object.__new__(LocalDecisionEngine);engine.model=LocalModel()
    engine.guard={'training_ranges':{'age':[19,39]},'confidence_floor':.1,'threshold_margin':.02}
    def infer(person):
        calls.append(1);p=np.zeros(16);p[0]=.7;p[1]=.3;return p
    engine.infer_joint_once=infer
    def forbidden(*args,**kwargs):raise AssertionError('External connection forbidden')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(socket,'create_connection',forbidden)
    result=engine.decide({'state':{'age':25,'sex':1,'HE_BMI':24},'questions':{'posterior':{'type':'choice','target':'any_abnormality'},'risk':{'type':'noul','target':'any_abnormality'}}})
    assert len(calls)==1 and result['external_model_calls']==0
    # A sensitivity-oriented action threshold is deliberately separate from argmax.
    assert result['answers']['posterior']['choice']=='false'
    assert result['policy_result']['screen_positive'] is True
    assert result['policy_result']['review_required'] is True
    assert result['policy_result']['negative_is_not_medical_clearance'] is True

def test_saved_local_endpoint_no_noul_confidence():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from metabolic.api import app
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/typed_local_v6_groups/guardrails.json').exists():pytest.skip('Build the local typed layer first')
    request={'state':{'age':24,'sex':1,'HE_BMI':24,'living_alone':1},
             'questions':{'risk':{'type':'noul','target':'any_abnormality'}},'policy':'sensitivity95'}
    with TestClient(app) as client:
        r=client.post('/v6/decide-local',json=request)
        assert r.status_code==200 and r.json()['external_model_calls']==0
        reasons=r.json()['policy_result']['review_reasons']
        assert 'limited_development_group_support' in reasons
        assert 'development_group_target_not_met' in reasons
        assert set(r.json()['answers']['risk'])=={'type','noul'}
        assert client.post('/v6/decide-local',json=request|{'questions':{}}).status_code==422
        bad=request|{'questions':{'other':{'type':'noul','target':'cancer'}}}
        assert client.post('/v6/decide-local',json=bad).status_code==422
