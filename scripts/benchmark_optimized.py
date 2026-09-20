"""Apples-to-apples warm latency and detailed stage profiling."""
import json,time,platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit,softmax,logit
from threadpoolctl import threadpool_limits
from metabolic.data import features,load_cohort
from metabolic.predict import Predictor
from metabolic.fast import FastPredictor,FastLogistic,patient_vector
from metabolic.refined import RefinedPredictor,CleanPreprocessor
from metabolic.decisions import Patient,render_decisions

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/optimized_v2';OLD=ROOT/'artifacts/local_parallel_v1_run2'
state=json.loads((OLD/'example_request.json').read_text());old=Predictor(OLD);fast=FastPredictor(OLD);new=RefinedPredictor(OUT/'model_complete')
lrpath=ROOT/'artifacts/comparison_household_v1/models/logistic_regression'
lrlegacy=joblib.load(lrpath/'pipeline.joblib');lrf=FastLogistic(lrpath)
lrp=CleanPreprocessor.load(OUT/'tuned_lr/preprocessor.npz');lra=dict(np.load(OUT/'tuned_lr/network.npz'));lrc=json.loads((OUT/'tuned_lr/config.json').read_text())
torch.set_num_threads(1)
summary=lambda v:{'p50_ms':float(np.median(v)),'p95_ms':float(np.quantile(v,.95)),'repeats':len(v)}
def risk_old(s):
    p=Patient.model_validate(s);x=old.preprocessor.transform(features(pd.DataFrame([p.model_dump()]).astype(float))).astype('float32')
    with torch.inference_mode():return float(1-torch.softmax(old.model(torch.from_numpy(x))/old.config['temperature'],dim=1)[0,0])
def risk_lr_old(s):
    p=Patient.model_validate(s);x=lrlegacy['preprocessor'].transform(features(pd.DataFrame([p.model_dump()]).astype(float)))
    q=lrlegacy['model'].predict_proba(x)[0,1];return float(expit(logit(np.clip(q,1e-7,1-1e-7))/lrf.config['temperature']))
def risk_fast(s):
    v=patient_vector(Patient.model_validate(s));return float(1-softmax(fast.net.logits(fast.pre.transform(v))/fast.config['temperature'])[0])
def risk_fast_lr(s):return float(lrf.probability(patient_vector(Patient.model_validate(s))))
def risk_new(s):return new.probability(patient_vector(Patient.model_validate(s)))
def risk_tuned_lr(s):
    v=patient_vector(Patient.model_validate(s));x=lrp.transform(v)
    return float(expit((x@lra['coef']+lra['bias'])/lrc['temperature']))
functions={'v1_legacy_nn':risk_old,'v1_fast_nn':risk_fast,'v2_refined_nn':risk_new,
    'lr_legacy':risk_lr_old,'lr_fast':risk_fast_lr,'lr_tuned_fast':risk_tuned_lr}
cuts={'v1_legacy_nn':old.config['threshold'],'v1_fast_nn':old.config['threshold'],
    'v2_refined_nn':new.config['thresholds']['sensitivity90'],'lr_legacy':lrf.config['thresholds']['sensitivity90'],
    'lr_fast':lrf.config['thresholds']['sensitivity90'],'lr_tuned_fast':lrc['thresholds']['sensitivity90']}
def call(name):
    p=functions[name](state)
    return json.dumps({'probability':p,'positive':p>=cuts[name]},allow_nan=False)
record={'threads':1,'device':platform.processor(),'platform':platform.platform(),'scope':'Warm CPU, identical input validation and binary JSON output. Excludes loading and HTTP. Optimized LR uses the same array runtime strategy.', 'common':{}}
with threadpool_limits(limits=1):
    rng=np.random.default_rng(2026);samples={k:[] for k in functions}
    for k in functions:
        for _ in range(30):call(k)
    for _ in range(1000):
        for k in rng.permutation(list(functions)):
            t=time.perf_counter_ns();call(k);samples[k].append((time.perf_counter_ns()-t)/1e6)
    record['common']={k:summary(v) for k,v in samples.items()}
    record['full_schema']={}
    for name,model in [('v1_fast_nn',fast),('v2_refined_nn',new)]:
        values=[]
        for _ in range(1000):
            t=time.perf_counter_ns();json.dumps(model.predict(state),allow_nan=False);values.append((time.perf_counter_ns()-t)/1e6)
        record['full_schema'][name]=summary(values)
    stage={k:[] for k in ['validation','dataframe_and_features','sklearn_transform','torch_forward','typed_output']}
    for _ in range(300):
        t=time.perf_counter_ns();p=Patient.model_validate(state);a=time.perf_counter_ns()
        f=features(pd.DataFrame([p.model_dump()]).astype(float));b=time.perf_counter_ns()
        x=old.preprocessor.transform(f).astype('float32');c=time.perf_counter_ns()
        with torch.inference_mode():probs=torch.softmax(old.model(torch.from_numpy(x))/old.config['temperature'],dim=1)[0].numpy()
        dd=time.perf_counter_ns();json.dumps(render_decisions(probs,old.config['threshold']));e=time.perf_counter_ns()
        for key,delta in zip(stage,[a-t,b-a,c-b,dd-c,e-dd]):stage[key].append(delta/1e6)
    record['legacy_stage_profile']={k:summary(v) for k,v in stage.items()}
    # Probability parity over all original eligible records, not just the timing input.
    d,_=load_cohort(ROOT);f=features(d);x=old.preprocessor.transform(f).astype('float32');a=fast.pre.transform(f.to_numpy()).astype('float32')
    np.testing.assert_allclose(a,x,atol=1e-6,rtol=1e-6)
    with torch.inference_mode():p=torch.softmax(old.model(torch.from_numpy(x))/old.config['temperature'],dim=1).numpy()
    q=softmax(fast.net.logits(a)/fast.config['temperature'],axis=1)
    record['v1_equivalence']={'n':len(d),'max_probability_difference':float(np.max(abs(p-q))),
        'binary_changes_at_saved_threshold':int(np.sum((p[:,1:].sum(axis=1)>=old.config['threshold'])!=(q[:,1:].sum(axis=1)>=old.config['threshold'])))}
    assert record['v1_equivalence']['max_probability_difference']<2e-6
(OUT/'latency.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
