"""Shared-preprocessing five-output LR benchmark against multitask NN."""
import json,time
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from metabolic.comparison import ROOT,dump
from metabolic.expanded import ExpandedPredictor,ExpandedPreprocessor,ExpandedPatient,EXTRA
from metabolic.fast import patient_vector
from metabolic.decisions import Patient

OUT=ROOT/'artifacts/expanded_multitask_v4'
state={'age':29,'sex':1,'HE_BMI':25.,'HE_wc':87.,'HE_ht':175.,'incm':2,'edu':4,'sm_presnt':0,'dr_month':1,'pa_aerobic':1,
    'sedentary_hours':8.5,'walking_days':4,'stress_level':2,'employed':1,'alcohol_frequency':3,'alcohol_amount':2,'strength_days':2,'family_history':1,'living_alone':1}
nn=ExpandedPredictor(OUT/'models/expanded_multi_s42')
lrp=ExpandedPreprocessor.load(OUT/'models/expanded_LR5');lra=dict(np.load(OUT/'models/expanded_LR5/network.npz'))
lrc=json.loads((OUT/'models/expanded_LR5/config.json').read_text());temps=np.array(lrc['temperature'])
def call(kind):
    person=ExpandedPatient.model_validate(state);values=person.model_dump()
    base={k:v for k,v in values.items() if k not in EXTRA}
    vector=np.r_[patient_vector(Patient.model_validate(base)),np.array([values[k] for k in EXTRA],float)]
    if kind=='NN':
        z=nn.logits(nn.pre.transform(vector));risk=float(expit(z[0]/nn.config['temperature']))
        from scipy.special import softmax
        from metabolic.data import BITS
        comp=risk*softmax(z[1:]@BITS[1:].T/nn.config['pattern_temperature'])@BITS[1:]
    else:
        p=expit((lrp.transform(vector)@lra['coef'].T+lra['bias'])/temps);risk=float(p[0]);comp=p[1:]
    return json.dumps({'risk':risk,'components':comp.tolist(),'expected_count':float(comp.sum())},allow_nan=False)

times={'NN':[],'LR5':[]};rng=np.random.default_rng(2026)
with threadpool_limits(limits=1):
    for k in times:
        for _ in range(30):call(k)
    for _ in range(1000):
        for k in rng.permutation(list(times)):
            start=time.perf_counter_ns();call(k);times[k].append((time.perf_counter_ns()-start)/1e6)
report={'scope':'Warm CPU one thread, input validation + one shared preprocessing pass + 5 output probabilities + count + JSON; no loading/HTTP',
        'results':{k:{'p50_ms':float(np.median(v)),'p95_ms':float(np.quantile(v,.95)),'repeats':len(v)} for k,v in times.items()}}
dump(OUT/'latency.json',report);dump(OUT/'example_request.json',state);dump(OUT/'example_response.json',nn.predict(state))
print(json.dumps(report,indent=2))
