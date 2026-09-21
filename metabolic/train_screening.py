"""Separate-role re-experiment focused on sensitivity and referral burden."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit,softmax
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from .comparison import ROOT,OLD,dump,restore_parts,youden_threshold
from .data import load_cohort,BITS
from .expanded import ExpandedPreprocessor
from .train_factorial import matrix,fit_nn,logits
from .train_specialized import binary_temp
from .train import weights
from .household import living_alone
from .screening import (POLICIES,screening_metrics,select_screening_threshold,SurveyBootstrap,
                        load_full_frame,age_household_groups,net_benefit,interval)

PLAN={'date':'2026-09-21','purpose':'Current undiagnosed metabolic-abnormality screening, not future disease prediction',
    'main_cohort':'Unchanged age19-39 unaware cohort; 12h fasting and complete targets',
    'roles':'Exact v1 PSU-disjoint roles: 2021-22 train1680/validation372; 2023 calibration481/threshold529; 2024 test1006',
    'change_from_v4':'No final refit on calibration or threshold participants. Frozen architecture from past exploratory work; no claim of fresh external validation.',
    'models':['base_NN','base_LR','expanded_NN','expanded_LR'],'nn_seeds':[42,43,44],'LR_C':[.03,.1,.3,1.],
    'selection':'Validation-only average specificity at empirical sensitivity90 and sensitivity95; Brier tie-break; NN early-stop validation BCE then refit same train rows',
    'cutoffs':'2023 threshold role only; primary sensitivity90/95, secondary Youden',
    'primary_measures':['actual sensitivity','specificity','referral_rate','PPV','NPV','missed_per_1000','unnecessary_referrals_per_1000','tests_per_detected'],
    'uncertainty':'300 full-frame Rao-Wu-style PSU resamples separately for calibration, threshold and test; calibration/cutoffs refit per replicate; model fitting/selection uncertainty not included',
    'household':'20-29 and30-39, single vs multi household, sex strata; age19 separate. Global cutoff applied to all subgroups; no test-based subgroup tuning',
    'DCA':'Exploratory common probability thresholds .05-.50, refer-all and refer-none references; no empirically established clinical cost ratio',
    'sensitivity_analysis':'Frozen fitted models evaluated in >=8h and untreated cohorts; not new training or interchangeable clinical definitions',
    'limitations':['2024 and architecture-development data were inspected before; this is exploratory.',
        'No direct independent-living/self-catering variable; cfam is household-size proxy.',
        'Official design weights used, but resampled weights are not official KNHANES replicate weights and no FPC is applied.']}

RATE_KEYS=['sensitivity','specificity','ppv','npv','referral_rate','miss_rate_among_positive',
           'referrals_per_1000','detected_per_1000','missed_per_1000','unnecessary_referrals_per_1000','tests_per_detected']

def selection_score(d,p):
    measures=[]
    for target in POLICIES.values():
        cut=select_screening_threshold(d.target,p,weights(d),target)
        measures.append(screening_metrics(d.target,p,cut,weights(d),False)['specificity'])
    return float(np.mean(measures)),float(brier_score_loss(d.target,p,sample_weight=weights(d)))

def make_cuts(d,p,w=None):
    w=weights(d) if w is None else w
    return {k:select_screening_threshold(d.target,p,w,target) for k,target in POLICIES.items()}|{'youden':youden_threshold(d.target,p,w)}

def model_logits(bundle,d):
    pre=bundle['pre'];expanded=bundle['expanded']
    if bundle['kind']=='NN':return logits(pre,bundle['model'],d,expanded)[:,0]
    return bundle['model'].decision_function(pre.transform(matrix(d,expanded)))

def main(name):
    out=ROOT/'artifacts'/name;out.mkdir(parents=True,exist_ok=False);dump(out/'PLAN.json',PLAN)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    d,flow=load_cohort(ROOT);d=d.copy();d['living_alone']=living_alone(d.cfam)
    manifest=json.loads((OLD/'split_manifest.json').read_text());parts=restore_parts(d,manifest);dump(out/'split_manifest.json',manifest)
    train,val,cal,threshold,test=[parts[k] for k in ['train','validation','calibration','threshold','test_2024']]
    selected={};candidates={}
    for expanded in [False,True]:
        label='expanded' if expanded else 'base';nn_trials=[];lr_trials=[]
        for seed in PLAN['nn_seeds']:
            _,_,ep=fit_nn(train,expanded,True,seed,180,val)
            pre,model,_=fit_nn(train,expanded,True,seed,ep)
            p=expit(logits(pre,model,val,expanded)[:,0]);score,brier=selection_score(val,p)
            nn_trials.append({'seed':seed,'epochs':ep,'specificity_score':score,'brier':brier,'pre':pre,'model':model})
        for c in PLAN['LR_C']:
            pre=ExpandedPreprocessor(expanded).fit(matrix(train,expanded));model=LogisticRegression(C=c,max_iter=3000,random_state=42)
            model.fit(pre.transform(matrix(train,expanded)),train.target,sample_weight=weights(train))
            p=model.predict_proba(pre.transform(matrix(val,expanded)))[:,1];score,brier=selection_score(val,p)
            lr_trials.append({'C':c,'specificity_score':score,'brier':brier,'pre':pre,'model':model})
        for kind,trials in [('NN',nn_trials),('LR',lr_trials)]:
            choice=sorted(trials,key=lambda z:(-z['specificity_score'],z['brier']))[0]
            selected[label+'_'+kind]=choice|{'expanded':expanded,'kind':kind}
            candidates[label+'_'+kind]=[{k:v for k,v in a.items() if k not in ['pre','model']} for a in trials]
        print(label+' candidates selected using validation only',flush=True)
    dump(out/'SELECTION.json',{'candidates':candidates,'selected':{k:{a:b for a,b in v.items() if a not in ['pre','model']} for k,v in selected.items()}})
    for key,bundle in selected.items():
        folder=out/'models'/key;folder.mkdir(parents=True);bundle['pre'].save(folder)
        bundle['cal_logits']=model_logits(bundle,cal);bundle['threshold_logits']=model_logits(bundle,threshold)
        temp=binary_temp(bundle['cal_logits'],cal.target,weights(cal));bundle['temperature']=temp
        bundle['cuts']=make_cuts(threshold,expit(bundle['threshold_logits']/temp))
        cfg={k:v for k,v in bundle.items() if k not in ['pre','model','cal_logits','threshold_logits']}
        cfg.update({'name':key,'thresholds':bundle['cuts'],'multitask':bundle['kind']=='NN','train_n':len(train),'calibration_n':len(cal),'threshold_n':len(threshold)})
        if bundle['kind']=='NN':
            z=logits(bundle['pre'],bundle['model'],cal,bundle['expanded']);pos=cal.target.to_numpy()==1;j=cal.joint_target.to_numpy()[pos]-1
            fit=minimize_scalar(lambda t:float(np.average(-np.log(np.clip(softmax(z[pos,1:]@BITS[1:].T/np.exp(t),axis=1)[np.arange(len(j)),j],1e-12,1)),weights=weights(cal)[pos])),bounds=(-2.3,2.3),method='bounded')
            assert fit.success;cfg['pattern_temperature']=float(np.exp(fit.x))
            np.savez(folder/'network.npz',**{k:v.detach().numpy() for k,v in bundle['model'].state_dict().items()})
        else:np.savez(folder/'network.npz',coef=bundle['model'].coef_[0],bias=bundle['model'].intercept_[0])
        dump(folder/'config.json',cfg)
    # Final test scores are computed only after selection/calibration/thresholds freeze.
    report={'plan':PLAN,'cohort_flow':flow,'partitions':{k:len(v) for k,v in parts.items()},'models':{},'paired':{},'sensitivity_analyses':{}}
    for key,bundle in selected.items():
        bundle['test_logits']=model_logits(bundle,test);p=expit(bundle['test_logits']/bundle['temperature'])
        groups={}
        for group,mask in age_household_groups(test).items():
            if not mask.any():continue
            positive=int(test.target.to_numpy()[mask].sum());negative=int(mask.sum())-positive
            groups[group]={'n':int(mask.sum()),'positive':positive,'low_information':min(positive,negative)<20,
                'policies':{policy:screening_metrics(test.target.to_numpy()[mask],p[mask],cut,weights(test)[mask]) for policy,cut in bundle['cuts'].items()}}
        report['models'][key]={'temperature':bundle['temperature'],'cutoffs':bundle['cuts'],
            'threshold_role':{policy:screening_metrics(threshold.target,expit(bundle['threshold_logits']/bundle['temperature']),cut,weights(threshold)) for policy,cut in bundle['cuts'].items()},
            'weighted':{policy:screening_metrics(test.target,p,cut,weights(test)) for policy,cut in bundle['cuts'].items()},
            'unweighted':{policy:screening_metrics(test.target,p,cut) for policy,cut in bundle['cuts'].items()},'groups':groups}
    full=load_full_frame(ROOT)
    bc,bt,be=[SurveyBootstrap(full,frame) for frame in [cal,threshold,test]]
    report['bootstrap_design']={'calibration':bc.audit,'threshold':bt.audit,'test':be.audit}
    samples={key:{policy:{m:[] for m in RATE_KEYS} for policy in POLICIES} for key in selected}
    groups=age_household_groups(test)
    gsamples={key:{group:{policy:{m:[] for m in RATE_KEYS} for policy in POLICIES} for group,mask in groups.items() if mask.any()} for key in ['expanded_NN','expanded_LR']}
    paired={tag:{policy:{m:[] for m in ['sensitivity','specificity','referral_rate','missed_per_1000','unnecessary_referrals_per_1000']} for policy in POLICIES} for tag in ['base','expanded']}
    rng=np.random.default_rng(20260921);completed=0;attempts=0
    while completed<300 and attempts<600:
        attempts+=1;wc,wt,we=bc.draw(rng),bt.draw(rng),be.draw(rng)
        if any(w[np.asarray(frame.target)==v].sum()==0 for frame,w in [(cal,wc),(threshold,wt),(test,we)] for v in [0,1]):continue
        current={}
        for key,bundle in selected.items():
            temp=binary_temp(bundle['cal_logits'],cal.target,wc)
            pt=expit(bundle['threshold_logits']/temp);pe=expit(bundle['test_logits']/temp);current[key]={}
            for policy,target in POLICIES.items():
                cut=select_screening_threshold(threshold.target,pt,wt,target)
                values=screening_metrics(test.target,pe,cut,we,False);current[key][policy]=values
                for measure in RATE_KEYS:samples[key][policy][measure].append(values[measure])
                if key in gsamples:
                    for group,mask in groups.items():
                        if not mask.any():continue
                        v=screening_metrics(test.target.to_numpy()[mask],pe[mask],cut,we[mask],False)
                        for measure in RATE_KEYS:gsamples[key][group][policy][measure].append(v[measure] if v else None)
        for tag in paired:
            for policy in POLICIES:
                for measure in paired[tag][policy]:paired[tag][policy][measure].append(current[tag+'_NN'][policy][measure]-current[tag+'_LR'][policy][measure])
        completed+=1
        if completed%75==0:print('screening uncertainty',completed,'/300',flush=True)
    if completed<285:raise RuntimeError('Insufficient valid bootstrap replicates')
    for key in samples:
        report['models'][key]['ci95']={policy:{m:interval(v) for m,v in values.items()} for policy,values in samples[key].items()}
        if key in gsamples:
            for group in gsamples[key]:report['models'][key]['groups'][group]['ci95']={policy:{m:interval(v) for m,v in values.items()} for policy,values in gsamples[key][group].items()}
    report['paired']={tag:{policy:{m:interval(v) for m,v in values.items()} for policy,values in policies.items()} for tag,policies in paired.items()}
    report['bootstrap_repeats']=completed
    report['decision_curve']=[]
    prev=float(np.average(test.target,weights=weights(test)))
    for threshold_probability in np.arange(.05,.501,.025):
        row={'threshold_probability':float(threshold_probability),'refer_none':0.,'refer_all':prev-(1-prev)*threshold_probability/(1-threshold_probability)}
        for key,bundle in selected.items():row[key]=net_benefit(test.target,expit(bundle['test_logits']/bundle['temperature']),weights(test),threshold_probability)
        report['decision_curve'].append(row)
    for mode,fasting in [('unaware',8),('untreated',12)]:
        alt,_=load_cohort(ROOT,mode,fasting);alt=alt[alt.survey_year==2024]
        name_alt=f'{mode}_fasting{fasting}';report['sensitivity_analyses'][name_alt]={}
        for key,bundle in selected.items():
            p=expit(model_logits(bundle,alt)/bundle['temperature'])
            report['sensitivity_analyses'][name_alt][key]={policy:screening_metrics(alt.target,p,cut,weights(alt)) for policy,cut in bundle['cuts'].items() if policy in POLICIES}
    report['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'metabolic').glob('*.py')}
    dump(out/'report.json',report)
    np.savez(out/'private_predictions.npz',**{key:expit(bundle['test_logits']/bundle['temperature']) for key,bundle in selected.items()})
    print(json.dumps({'output':str(out),'screening':{key:value['weighted'] for key,value in report['models'].items()}},indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run-name',default='screening_v5');args=a.parse_args()
    if Path(args.run_name).name!=args.run_name:a.error('Invalid artifact directory')
    main(args.run_name)
