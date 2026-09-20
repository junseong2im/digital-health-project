"""Warm single-person benchmark of v3 and its matched optimized LR."""
import json,time,platform
from pathlib import Path
import numpy as np
from scipy.special import expit,softmax
from threadpoolctl import threadpool_limits
from metabolic.specialized import SpecializedPredictor
from metabolic.refined import RefinedPredictor,CleanPreprocessor
from metabolic.fast import FastPredictor,patient_vector
from metabolic.decisions import Patient

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/specialized_v3'
state=json.loads((ROOT/'artifacts/local_parallel_v1_run2/example_request.json').read_text())
v1=FastPredictor(ROOT/'artifacts/local_parallel_v1_run2');v2=RefinedPredictor(ROOT/'artifacts/optimized_v2/model_complete');v3=SpecializedPredictor(OUT/'model')
lp=CleanPreprocessor.load(OUT/'logistic/preprocessor.npz');params=dict(np.load(OUT/'logistic/network.npz'));cfg=json.loads((OUT/'logistic/config.json').read_text())
def one(name):
    vec=patient_vector(Patient.model_validate(state))
    if name=='v1_fast':p=float(1-softmax(v1.net.logits(v1.pre.transform(vec))/v1.config['temperature'])[0]);cut=v1.config['threshold']
    elif name=='v2_fast':p=v2.probability(vec);cut=v2.config['thresholds']['sensitivity90']
    elif name=='v3_specialized':p=v3.probability(vec);cut=v3.config['thresholds']['sensitivity90']
    else:p=float(expit((lp.transform(vec)@params['coef']+params['bias'])/cfg['temperature']));cut=cfg['thresholds']['sensitivity90']
    return json.dumps({'probability':p,'positive':p>=cut},allow_nan=False)

summarize=lambda v:{'p50_ms':float(np.median(v)),'p95_ms':float(np.quantile(v,.95)),'repeats':len(v)}
names=['v1_fast','v2_fast','v3_specialized','v3_tuned_lr'];times={k:[] for k in names};rng=np.random.default_rng(2026)
with threadpool_limits(limits=1):
    for name in names:
        for _ in range(30):one(name)
    for _ in range(1000):
        for name in rng.permutation(names):
            t=time.perf_counter_ns();one(name);times[name].append((time.perf_counter_ns()-t)/1e6)
    full=[]
    for _ in range(1000):
        t=time.perf_counter_ns();json.dumps(v3.predict(state),allow_nan=False);full.append((time.perf_counter_ns()-t)/1e6)
record={'scope':'Same synthetic state, warm CPU one thread, identical validation/features/binary JSON contract; startup/HTTP excluded',
    'device':platform.processor(),'common':{k:summarize(v) for k,v in times.items()},'v3_full_choice_score_noul':summarize(full)}
(OUT/'latency.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(OUT/'example_request.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
(OUT/'example_response.json').write_text(json.dumps(v3.predict(state),indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
