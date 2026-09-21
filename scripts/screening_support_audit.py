"""Check subgroup threshold support and household code consistency after fitting."""
import json
import numpy as np
from scipy.special import expit
from metabolic.comparison import ROOT,OLD,dump,restore_parts
from metabolic.data import load_cohort
from metabolic.household import living_alone
from metabolic.expanded import ExpandedPredictor,extended_features
from metabolic.screening import age_household_groups,screening_metrics
from metabolic.train import weights

d,_=load_cohort(ROOT);d=d.copy();d['living_alone']=living_alone(d.cfam)
valid=d.cfam.isin(range(1,7))&d.genertn.isin(range(1,8))
conflicts=int((valid&(d.cfam.eq(1)!=d.genertn.eq(1))).sum());assert conflicts==0
parts=restore_parts(d,json.loads((OLD/'split_manifest.json').read_text()))
threshold=parts['threshold'];out=ROOT/'artifacts/screening_v5'
model=ExpandedPredictor(out/'models/expanded_NN')
logits=model.logits(model.pre.transform(extended_features(threshold).to_numpy()))
p=expit(logits[:,0]/model.config['temperature'])
audit={'household_code_conflicts':conflicts,'threshold_groups':{},'note':'Support diagnostic only; global thresholds remain unchanged. No test-driven subgroup thresholds were added.'}
for name,mask in age_household_groups(threshold).items():
    if not mask.any():continue
    audit['threshold_groups'][name]={policy:screening_metrics(threshold.target.to_numpy()[mask],p[mask],cut,weights(threshold)[mask]) for policy,cut in model.config['thresholds'].items() if policy in ['sensitivity90','sensitivity95']}
dump(out/'subgroup_support.json',audit)
request={'state':{'age':24,'sex':1,'HE_BMI':24.,'HE_wc':84.,'HE_ht':175.,'living_alone':1},'policy':'sensitivity95','input_set':'expanded'}
dump(out/'example_request.json',request)
from metabolic.api import screen,ScreeningRequest
dump(out/'example_response.json',screen(ScreeningRequest.model_validate(request)))
print('Household code consistency verified; subgroup threshold support saved.')
