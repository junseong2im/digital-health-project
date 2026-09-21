"""Build/evaluate a local typed layer from public design principles only."""
import json,time
from collections import Counter
import numpy as np
from scipy.special import expit,softmax
from pydantic import ValidationError
from threadpoolctl import threadpool_limits
from metabolic.comparison import ROOT,OLD,dump,restore_parts
from metabolic.data import load_cohort,features,BITS,CATEGORIES
from metabolic.expanded import extended_features
from metabolic.expanded import EXTRA,ExpandedPredictor,ExpandedPatient
from metabolic.local_decisions import LocalDecisionEngine,concentration
from metabolic.screening import screening_metrics
from metabolic.train import weights

BASE_KEYS=['age','sex','incm','edu','HE_BMI','HE_wc','HE_ht','sm_presnt','dr_month','pa_aerobic']
QUESTIONS={'risk':{'type':'noul','target':'any_abnormality'},
    'glucose':{'type':'noul','target':'elevated_glucose'},'bp':{'type':'noul','target':'elevated_bp'},
    'tg':{'type':'noul','target':'elevated_tg'},'hdl':{'type':'noul','target':'low_hdl'},
    'burden':{'type':'score','target':'abnormality_count'},
    'posterior_class':{'type':'choice','target':'any_abnormality','labels':['no_abnormality','any_abnormality']}}

def states_for(d):
    f=extended_features(d);states=[]
    for idx,row in d.iterrows():
        state={}
        for key in BASE_KEYS:
            v=row[key] if key=='HE_ht' else f.loc[idx,key]
            state[key]=None if np.isnan(v) else int(v) if key in CATEGORIES or key=='age' else float(v)
        for key in EXTRA:
            v=f.loc[idx,key];state[key]=None if np.isnan(v) else float(v) if key=='sedentary_hours' else int(v)
        states.append(state)
    return states

def main(run_name='typed_local_v6_groups'):
    out=ROOT/'artifacts'/run_name;out.mkdir(parents=True,exist_ok=False)
    plan={'sources':['https://docs.typesafe.ai/introduction','https://docs.typesafe.ai/primitives','https://docs.typesafe.ai/confidence','https://docs.typesafe.ai/patterns'],
        'adaptation':'Public interface/decision architecture only. Not knowledge distillation, proprietary model replication, or RLCD.',
        'external_teacher_calls':0,'api_keys_used':False,'new_neural_training':False,
        'backbone':'screening_v5/models/expanded_NN, unchanged',
        'guardrail_design':'Feature bounds from training min/max; concentration floor = validation tenth percentile; fixed probability distance .02 from cutoff; missing inputs flagged; threshold-role subgroup positive n<20 or sensitivity below target flagged',
        'subgroup_revision':'Exploratory follow-up rule after subgroup audit; support values come only from 2023 threshold-role data; not a prospective validation',
        'scope':'Question requests select deterministic views; no natural-language task induction; no test-defined subgroup thresholds',
        'medical_boundary':'No high-confidence automatic medical clearance; review flag does not silently change screen result'}
    dump(out/'PLAN.json',plan)
    d,_=load_cohort(ROOT);parts=restore_parts(d,json.loads((OLD/'split_manifest.json').read_text()))
    train,val,test=parts['train'],parts['validation'],parts['test_2024']
    path=ROOT/'artifacts/screening_v5/models/expanded_NN';model=ExpandedPredictor(path)
    z=model.logits(model.pre.transform(extended_features(val).to_numpy()));p=expit(z[:,0]/model.config['temperature'])
    floor=float(np.quantile([concentration([1-v,v]) for v in p],.1))
    tr=extended_features(train).copy();tr['HE_ht']=train.HE_ht
    ranges={k:[float(tr[k].min()),float(tr[k].max())] for k in ['age','HE_BMI','HE_wc','HE_ht','sedentary_hours','walking_days']}
    guard={'confidence_floor':floor,'confidence_definition':'1-H(probabilities)/log(K), local definition, not vendor formula',
        'threshold_margin':.02,'training_ranges':ranges,'fit_roles':{'ranges':'train','confidence_floor':'validation'},
        'no_individual_correctness_guarantee':True};dump(out/'guardrails.json',guard)
    support=json.loads((ROOT/'artifacts/screening_v5/subgroup_support.json').read_text())['threshold_groups']
    names=['age19','age20_29_living0','age20_29_living1','age30_39_living0','age30_39_living1']
    guard['development_group_support']={group:{policy:{'positive_n':support[group][policy]['positive_n'],
        'n':support[group][policy]['n'],'observed_sensitivity':support[group][policy]['sensitivity'],
        'target_sensitivity':.9 if policy=='sensitivity90' else .95} for policy in ['sensitivity90','sensitivity95']} for group in names if group in support}
    guard['minimum_group_positive_n']=20;dump(out/'guardrails.json',guard)
    engine=LocalDecisionEngine(path,out/'guardrails.json');states=states_for(test)
    z=model.logits(model.pre.transform(extended_features(test).to_numpy()));p=expit(z[:,0]/model.config['temperature'])
    report={'plan':plan,'guardrails':guard,'policies':{},'test_n':len(test)}
    for policy in ['sensitivity90','sensitivity95']:
        reviews=np.zeros(len(test),bool);invalid=0;reasons=Counter();differences=[];changes=0
        for i,state in enumerate(states):
            try:
                result=engine.decide({'state':state,'questions':QUESTIONS,'policy':policy})
                reviews[i]=result['policy_result']['review_required'];reasons.update(result['policy_result']['review_reasons'])
                differences.append(abs(result['answers']['risk']['noul']-p[i]))
                changes+=result['policy_result']['screen_positive']!=bool(p[i]>=model.config['thresholds'][policy])
            except ValidationError:
                invalid+=1;reviews[i]=True;reasons['insufficient_or_invalid_state']+=1
        original=p>=model.config['thresholds'][policy];scenario=original|reviews
        hypothetical=screening_metrics(test.target,scenario.astype(float),.5,weights(test),False)
        for k in ['roc_auc','average_precision','brier']:hypothetical.pop(k)
        report['policies'][policy]={'valid_state_n':len(test)-invalid,'invalid_state_n':invalid,'review_n':int(reviews.sum()),
            'weighted_review_rate':float(np.average(reviews,weights=weights(test))),
            'review_reasons_unweighted':dict(reasons),'max_probability_difference':float(max(differences)),
            'screen_decision_changes_on_valid_states':int(changes),
            'original_screening':screening_metrics(test.target,p,model.config['thresholds'][policy],weights(test)),
            'hypothetical_test_every_review_case':hypothetical,
            'note':'Review-all scenario assumes every flagged case receives confirmatory tests; this is a cost scenario, not an observed clinical intervention.'}
    # Synthetic example, independent of earlier experiment artifacts.
    sample={'age':29,'sex':1,'HE_BMI':25.,'HE_wc':87.,'HE_ht':175.,'incm':2,'edu':4,
        'sm_presnt':0,'dr_month':1,'pa_aerobic':1,'sedentary_hours':8.5,'walking_days':4,
        'stress_level':2,'employed':1,'alcohol_frequency':3,'alcohol_amount':2,
        'strength_days':2,'family_history':1,'living_alone':1}
    request={'state':sample,'questions':QUESTIONS,'policy':'sensitivity90'}
    result=engine.decide(request);dump(out/'example_request.json',request);dump(out/'example_response.json',result)
    timings={}
    with threadpool_limits(limits=1):
        for label,qs in [('one_question',{'risk':QUESTIONS['risk']}),('seven_questions',QUESTIONS)]:
            req=request|{'questions':qs}
            for _ in range(30):engine.decide(req)
            values=[]
            for _ in range(1000):
                t=time.perf_counter_ns();json.dumps(engine.decide(req),allow_nan=False);values.append((time.perf_counter_ns()-t)/1e6)
            timings[label]={'p50_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95)),'repeats':1000}
    report['latency']=timings;report['latency_scope']='Warm CPU one thread, typed validation, one network pass, policy/review flags, JSON; loading/HTTP excluded'
    dump(out/'report.json',report);print(json.dumps({'output':str(out),'policies':report['policies'],'latency':timings},indent=2))

if __name__=='__main__':
    import argparse
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',default='typed_local_v6_groups');args=ap.parse_args()
    if Path(args.run_name).name!=args.run_name:ap.error('Invalid output directory')
    main(args.run_name)
