"""Warm CPU inference benchmark with a common binary-output contract."""
import json
import platform
import time
from datetime import datetime
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from metabolic.data import features
from metabolic.decisions import Patient
from metabolic.predict import Predictor

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/local_parallel_v1_run2'
COMP=ROOT/'artifacts/comparison_household_v1'
OUT=ROOT/'.benchmarks'/('metabolic_latency_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1)
state=json.loads((BASE/'example_request.json').read_text())
nn=Predictor(BASE)
models={'parallel_nn':{'pre':nn.preprocessor,'temperature':nn.config['temperature'],
    'threshold':nn.config['threshold'],'model':nn.model}}
for name in ['logistic_regression','random_forest','xgboost','lightgbm']:
    path=COMP/'models'/name;bundle=joblib.load(path/'pipeline.joblib')
    cfg=json.loads((path/'config.json').read_text())
    model=bundle['model']
    if 'n_jobs' in model.get_params():model.set_params(n_jobs=1)
    models[name]={'pre':bundle['preprocessor'],'temperature':cfg['temperature'],
        'threshold':cfg['thresholds']['sensitivity90'],'model':model}

def probability(name,x):
    spec=models[name]
    if name=='parallel_nn':
        with torch.inference_mode():
            logits=spec['model'](torch.from_numpy(np.asarray(x,dtype=np.float32)))
            return (1-torch.softmax(logits/spec['temperature'],dim=1)[:,0]).numpy()
    raw=spec['model'].predict_proba(x)[:,1]
    return expit(logit(np.clip(raw,1e-7,1-1e-7))/spec['temperature'])

def full(name):
    patient=Patient.model_validate(state)
    frame=pd.DataFrame([patient.model_dump()]).astype(float)
    x=models[name]['pre'].transform(features(frame))
    p=float(probability(name,x)[0])
    return json.dumps({'probability':p,'positive':p>=models[name]['threshold']},allow_nan=False)

def summary(values):
    return {'p50_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95)),
            'min_ms':float(np.min(values)),'repeats':len(values)}

rng=np.random.default_rng(2026)
result={'scope':'Warm local CPU, one thread per model, excludes startup/loading/HTTP. Synthetic fixed patient, not a clinical validation.',
    'common_output':'One binary risk probability and threshold decision; includes validation, derived features, preprocessing, calibration and JSON serialization.',
    'device':platform.processor(),'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,
    'repeats':300,'warmup':20,'results':{}}
with threadpool_limits(limits=1),warnings.catch_warnings():
    # LightGBM assigns feature names to ndarray training; feature order is fixed.
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    for name in models:
        for _ in range(20):full(name)
    samples={name:[] for name in models}
    for _ in range(300):
        for name in rng.permutation(list(models)):
            t=time.perf_counter_ns();full(name);samples[name].append((time.perf_counter_ns()-t)/1e6)
    for name in models:result['results'][name]={'single_full':summary(samples[name])}
    for size in [1,64]:
        prepared={name:spec['pre'].transform(features(pd.DataFrame([state]*size).astype(float))) for name,spec in models.items()}
        samples={name:[] for name in models}
        for name in models:
            for _ in range(20):probability(name,prepared[name])
        for _ in range(300):
            for name in rng.permutation(list(models)):
                t=time.perf_counter_ns();p=probability(name,prepared[name]);samples[name].append((time.perf_counter_ns()-t)/1e6)
                assert len(p)==size and np.isfinite(p).all()
        for name in models:result['results'][name][f'core_batch_{size}']=summary(samples[name])
    native=[]
    for _ in range(20):nn.predict(state)
    for _ in range(300):
        t=time.perf_counter_ns();json.dumps(nn.predict(state),allow_nan=False);native.append((time.perf_counter_ns()-t)/1e6)
    result['nn_full_choice_score_noul']=summary(native)
(OUT/'benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'path':str(OUT),'results':result['results'],'nn_full_output':result['nn_full_choice_score_noul']},indent=2))
