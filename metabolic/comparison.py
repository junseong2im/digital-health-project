"""Fixed-protocol peer-model comparison and household subgroup extension."""
import argparse
import copy
import hashlib
import json
import platform
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyreadstat
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax,expit,logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,log_loss,brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from .data import load_cohort,features,FEATURES,NUMERIC,CATEGORIES,TARGET_INPUTS
from .train import ROOT,weights,metrics,choose_threshold
from .network import ParallelDecisionNet
from .household import living_alone,household_features

OLD=ROOT/'artifacts/local_parallel_v1_run2'
PLAN={
    'primary_cohort':'Original unaware cohort, >=12h fasting, complete targets; unchanged',
    'partitions':'Recover exact original split_manifest; 2024 has already been inspected in prior work',
    'status':'Exploratory extension on reused temporal holdout, not a new external validation',
    'household':'cfam 1 = single_person; 2-6 = multi_person (6 top coded); 9/missing = unknown; genertn cross-check only',
    'seeds':[42,43,44], 'epochs':250, 'patience':25,
    'comparators':['logistic_regression','random_forest','xgboost','lightgbm','parallel_nn'],
    'calibration':'2023 calibration subset only; temperature for all models',
    'thresholds':['2023 threshold-set Youden J','2023 threshold-set sensitivity >=0.90'],
    'comparison':'same train rows, imputation/scaling, temporal partitions and survey weights',
    'household_ablation':'base vs additional living_alone one-hot; three paired seeds, no best-seed selection',
    'subgroups':['sex','age_19_29_vs_30_39','sex_cross_age','household','household_cross_sex_cross_age'],
    'specialist_models':'one seed42 NN per household/age subset if all partitions have >=10 per outcome and training n>=100',
    'inference':'exploratory unweighted complete-case logit, PSU-cluster SE, age/smoking and living/smoking interactions',
    'ci':'300 paired resamples of eligible PSUs inside year-strata; exploratory domain approximation',
    'promotion':'no automatic replacement of original model based on this reused holdout',
}

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def restore_parts(d,manifest):
    ids=pd.Series([hashlib.sha256(f'{r.survey_year}:{r.ID}'.encode()).hexdigest() for r in d.itertuples()],index=d.index)
    if ids.duplicated().any():raise ValueError('Nonunique participant identifier')
    lookup=dict(zip(ids,d.index));parts={}
    for name,keys in manifest.items():
        parts[name]=d.loc[[lookup[k] for k in keys]].copy()
    seen=set()
    for name,frame in parts.items():
        if seen & set(frame.index):raise ValueError('Participant leakage')
        seen.update(frame.index)
        for other,of in parts.items():
            if other!=name and set(frame.group)&set(of.group):raise ValueError('PSU leakage')
    if seen!=set(d.index):raise ValueError('Cohort differs from original frozen split')
    return parts

def preprocessor(household):
    cats=dict(CATEGORIES)
    if household:cats['living_alone']=[0,1]
    return ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),NUMERIC),
        ('category',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=-1)),
            ('encode',OneHotEncoder(categories=[[-1]+v for v in cats.values()],handle_unknown='error',sparse_output=False))]),list(cats))])

def youden_threshold(y,p,w):
    fpr,tpr,cuts=roc_curve(y,p,sample_weight=w,drop_intermediate=False)
    valid=np.flatnonzero(np.isfinite(cuts)&(cuts>=0)&(cuts<=1))
    if not len(valid):raise ValueError('No finite Youden threshold')
    return float(cuts[valid[np.argmax((tpr-fpr)[valid])]])

def thresholds(parts,prob):
    d=parts['threshold'];y=d.target.to_numpy();w=weights(d)
    return {'sensitivity90':choose_threshold(y,prob['threshold'],w,.9),
            'youden':youden_threshold(y,prob['threshold'],w)}

def net_fit(parts,household,seed,path,name,subgroup=None):
    torch.manual_seed(seed);np.random.seed(seed)
    pre=preprocessor(household).fit(household_features(parts['train'],household))
    x={k:pre.transform(household_features(v,household)).astype('float32') for k,v in parts.items()}
    net=ParallelDecisionNet(x['train'].shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    tx=torch.from_numpy(x['train']);ty=torch.tensor(parts['train'].joint_target.to_numpy(),dtype=torch.long)
    tw=torch.tensor(weights(parts['train']),dtype=torch.float32)
    vx=torch.from_numpy(x['validation']);vy=torch.tensor(parts['validation'].joint_target.to_numpy(),dtype=torch.long)
    vw=torch.tensor(weights(parts['validation']),dtype=torch.float32)
    loss_fn=torch.nn.CrossEntropyLoss(reduction='none');best=None;bestloss=float('inf');stale=0;history=[]
    for epoch in range(PLAN['epochs']):
        net.train()
        for ix in torch.randperm(len(tx)).split(128):
            opt.zero_grad();loss=(loss_fn(net(tx[ix]),ty[ix])*tw[ix]).mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        net.eval()
        with torch.no_grad():v=float((loss_fn(net(vx),vy)*vw).sum()/vw.sum())
        history.append(v)
        if v<bestloss-1e-5:bestloss=v;best=copy.deepcopy(net.state_dict());stale=0;bestepoch=epoch+1
        else:stale+=1
        if stale>=PLAN['patience']:break
    net.load_state_dict(best);net.eval()
    with torch.inference_mode():logits={k:net(torch.from_numpy(v)).numpy() for k,v in x.items()}
    yc=parts['calibration'].joint_target.to_numpy();wc=weights(parts['calibration'])
    objective=lambda lt:float(np.average(-np.log(np.clip(softmax(logits['calibration']/np.exp(lt),axis=1)[np.arange(len(yc)),yc],1e-12,1)),weights=wc))
    fit=minimize_scalar(objective,bounds=(-2.3,2.3),method='bounded')
    if not fit.success:raise RuntimeError('Temperature fit failed')
    temp=float(np.exp(fit.x));prob={k:1-softmax(v/temp,axis=1)[:,0] for k,v in logits.items()}
    cuts=thresholds(parts,prob)
    path.mkdir(parents=True,exist_ok=False)
    torch.save({'state_dict':net.state_dict(),'input_dim':x['train'].shape[1]},path/'network.pt')
    joblib.dump(pre,path/'preprocessor.joblib')
    config={'name':name,'seed':seed,'include_household':household,'temperature':temp,'thresholds':cuts,
            'best_epoch':bestepoch,'subgroup':subgroup,'n_train':len(tx),'validation_history':history,
            'training':'supervised weighted joint NLL; not Jev/RLCD'}
    dump(path/'config.json',config)
    return prob,config

def load_frozen_nn(parts):
    pre=joblib.load(OLD/'preprocessor.joblib');ck=torch.load(OLD/'network.pt',map_location='cpu',weights_only=True)
    net=ParallelDecisionNet(ck['input_dim']);net.load_state_dict(ck['state_dict']);net.eval()
    config=json.loads((OLD/'config.json').read_text());temp=config['temperature']
    with torch.inference_mode():
        prob={k:1-softmax(net(torch.tensor(pre.transform(features(v)),dtype=torch.float32)).numpy()/temp,axis=1)[:,0] for k,v in parts.items()}
    cuts=thresholds(parts,prob)
    if not np.isclose(cuts['sensitivity90'],config['threshold'],atol=1e-7):raise ValueError('Frozen cutoff mismatch')
    return prob,{'temperature':temp,'thresholds':cuts,'source':str(OLD),'seed':42,'include_household':False}

def baseline_fit(parts,name,estimator,path):
    pre=preprocessor(False).fit(features(parts['train']))
    x={k:pre.transform(features(v)) for k,v in parts.items()}
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');estimator.fit(x['train'],parts['train'].target,sample_weight=weights(parts['train']))
    raw={k:estimator.predict_proba(v)[:,1] for k,v in x.items()}
    d=parts['calibration']
    objective=lambda lt:log_loss(d.target,expit(logit(np.clip(raw['calibration'],1e-7,1-1e-7))/np.exp(lt)),sample_weight=weights(d),labels=[0,1])
    fit=minimize_scalar(objective,bounds=(-2.3,2.3),method='bounded')
    if not fit.success:raise RuntimeError('Calibration failed')
    temp=float(np.exp(fit.x));prob={k:expit(logit(np.clip(v,1e-7,1-1e-7))/temp) for k,v in raw.items()}
    conf={'name':name,'temperature':temp,'thresholds':thresholds(parts,prob),'warnings':[str(z.message) for z in ws],
          'parameters':{k:('NaN (missing-value sentinel)' if isinstance(v,float) and np.isnan(v) else v) for k,v in estimator.get_params().items()}}
    path.mkdir(parents=True,exist_ok=False);joblib.dump({'preprocessor':pre,'model':estimator},path/'pipeline.joblib')
    dump(path/'config.json',conf)
    return prob,conf

def subgroup_masks(d):
    groups={'all':np.ones(len(d),dtype=bool)}
    for name,mask in [('male',d.sex.eq(1)),('female',d.sex.eq(2)),('age_19_29',d.age.lt(30)),('age_30_39',d.age.ge(30)),
                      ('living_1',d.living_alone.eq(1)),('living_0',d.living_alone.eq(0)),('living_unknown',d.living_alone.isna())]:
        groups[name]=np.asarray(mask)
    for sex in [1,2]:
        for age,am in [('19_29',d.age.lt(30)),('30_39',d.age.ge(30))]:
            mask=d.sex.eq(sex)&am;groups[f'sex_{sex}_age_{age}']=np.asarray(mask)
            for live in [0,1]:groups[f'living_{live}_sex_{sex}_age_{age}']=np.asarray(mask&d.living_alone.eq(live))
    return groups

def evaluate(parts,prob,conf):
    d=parts['test_2024'];w=weights(d);y=d.target.to_numpy();p=prob['test_2024']
    results={'thresholds':conf['thresholds'],'policies':{},'subgroups':{}}
    for policy,cut in conf['thresholds'].items():
        results['policies'][policy]={'weighted':metrics(y,p,cut,w),'unweighted':metrics(y,p,cut)}
    for name,m in subgroup_masks(d).items():
        if not m.any():continue
        npos=int(y[m].sum());nneg=int(m.sum())-npos
        results['subgroups'][name]={'n':int(m.sum()),'positive':npos,'negative':nneg,'low_information':min(npos,nneg)<20,
            'weighted_prevalence':float(np.average(y[m],weights=w[m])),
            'metrics':{policy:metrics(y[m],p[m],cut,w[m]) for policy,cut in conf['thresholds'].items()}}
    return results

def bootstrap_indexes(d,repeats=300,seed=2026):
    rng=np.random.default_rng(seed);strata=[]
    for _,s in d.groupby(['survey_year','kstrata']):
        strata.append([np.flatnonzero(d.group.to_numpy()==g) for g in s.group.unique()])
    for _ in range(repeats):
        yield np.concatenate([np.concatenate([cs[i] for i in rng.integers(0,len(cs),len(cs))]) for cs in strata])

def paired_ci(d,a,b,repeats=300):
    y=d.target.to_numpy();w=weights(d);vals={'delta_auc':[],'delta_ap':[],'delta_brier':[]}
    for ix in bootstrap_indexes(d,repeats):
        if len(np.unique(y[ix]))<2:continue
        for key,fn in [('delta_auc',roc_auc_score),('delta_ap',average_precision_score),('delta_brier',brier_score_loss)]:
            vals[key].append(float(fn(y[ix],a[ix],sample_weight=w[ix])-fn(y[ix],b[ix],sample_weight=w[ix])))
    return {k:{'mean_bootstrap':float(np.mean(v)),'ci95':np.quantile(v,[.025,.975]).tolist(),'valid_repeats':len(v)} for k,v in vals.items()}

def prevalence(d):
    groups=subgroup_masks(d);w=weights(d);y=d.target.to_numpy();out={}
    for name,m in groups.items():
        if not m.any():continue
        out[name]={'n':int(m.sum()),'positive':int(y[m].sum()),'unweighted':float(y[m].mean()),'weighted':float(np.average(y[m],weights=w[m]))}
    boot={k:[] for k in out}
    for ix in bootstrap_indexes(d):
        for name in out:
            select=ix[groups[name][ix]]
            if len(select):boot[name].append(float(np.average(y[select],weights=w[select])))
    for name in out:out[name]['ci95_exploratory']=np.quantile(boot[name],[.025,.975]).tolist()
    return out

def peer_audit():
    raw=[]
    for yr in range(2021,2025):
        d,_=pyreadstat.read_sav(str(ROOT/f'data/raw/knhanes/HN{str(yr)[2:]}_ALL(SPSS)/HN{str(yr)[2:]}_ALL.sav'))
        d['survey_year']=yr;raw.append(d)
    raw=pd.concat(raw,ignore_index=True);young=raw[raw.age.between(19,39)].copy()
    study=young[~young[['DI1_pt','DI2_pt','DE1_pt']].eq(1).any(axis=1)].copy()
    study['WHtR']=study.HE_wc/study.HE_ht
    # Reconstruct the peer code's labels only to quantify missing-target errors.
    study['peer_target']=((study.HE_glu>=100)|(study.HE_sbp>=130)|(study.HE_dbp>=85)|(study.HE_TG>=150)|
        ((study.sex==1)&(study.HE_HDL_st2<40))|((study.sex==2)&(study.HE_HDL_st2<50))).astype(int)
    clean=study.dropna(subset=FEATURES)
    missing=clean[TARGET_INPUTS].isna().any(axis=1)
    n=json.loads((ROOT/'data/reference/peer/my_model.ipynb').read_text(encoding='utf-8'))
    outputs={str(i):''.join(''.join(o.get('text',[])) for o in n['cells'][i].get('outputs',[]) if o.get('output_type')=='stream') for i in [3,4,5,6,9,12]}
    audit={'commit':json.loads((ROOT/'data/reference/peer/commit.json').read_text())['sha'],
        'published_text':{'young':5640,'clean':4678,'train':3274,'test':1404,'youden':.347},
        'notebook_stored_outputs':outputs,'current_data_reconstruction':{'all':len(raw),'young':len(young),'after_treatment_exclusion':len(study),
            'feature_complete':len(clean),'positive':int(clean.peer_target.sum()),'missing_any_target_measurement':int(missing.sum()),
            'missing_target_coded_negative':int((missing&clean.peer_target.eq(0)).sum())},
        'method_differences':['Test outcomes used to choose Youden cutoff (cell 9).',
            'Age-specific AUC is fitted and evaluated on the same data (cell 12).',
            'Target OR comparisons make missing measurements False before dropna on Target.',
            '*_pt describes treatment, not the dedicated medication variables.',
            'incm is personal income quartile, not quintile.',
            'AUC is not accuracy. Odds ratios are not risk ratios; coefficients do not establish causal mechanisms.',
            'No survey weighting or PSU group split in peer pipeline.']}
    return audit,raw

def associations(d):
    import statsmodels.formula.api as smf
    a=features(d)
    a['living_alone']=d.living_alone;a['target']=d.target;a['group']=d.group
    a['survey_year']=d.survey_year;a['age30']=d.age.ge(30).astype(int)
    needed=['age','sex','incm','edu','HE_wc','sm_presnt','dr_month','pa_aerobic','living_alone']
    a=a.dropna(subset=needed).copy()
    formula='target ~ living_alone + age + C(sex) + C(incm) + C(edu) + HE_wc + sm_presnt + dr_month + pa_aerobic + C(survey_year)'
    out={'n':len(a),'method':'Unweighted complete-case maximum-likelihood logit; year-PSU clustered SE; not full survey design inference','models':{}}
    for name,form in [('adjusted',formula),('interactions',formula+' + age30 + age30:sm_presnt + living_alone:sm_presnt')]:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            fit=smf.logit(form,a).fit(disp=False,maxiter=200,cov_type='cluster',cov_kwds={'groups':a.group})
        if not fit.mle_retvals['converged']:raise RuntimeError('Association model did not converge')
        ci=fit.conf_int()
        out['models'][name]={'formula':form,'warnings':[str(w.message) for w in ws],
            'terms':{k:{'beta':float(fit.params[k]),'aOR':float(np.exp(fit.params[k])),
                'ci95':np.exp(ci.loc[k].to_numpy()).tolist(),'p':float(fit.pvalues[k])} for k in fit.params.index}}
    return out

def main(run_name):
    out=ROOT/'artifacts'/run_name;out.mkdir(parents=True,exist_ok=False);dump(out/'PLAN.json',PLAN)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    audit,raw=peer_audit();dump(out/'peer_audit.json',audit)
    d,flow=load_cohort(ROOT);d=d.copy();d['living_alone']=living_alone(d.cfam)
    valid=d.cfam.isin([1,2,3,4,5,6])&d.genertn.isin([1,2,3,4,5,6,7])
    household_audit={'definition':PLAN['household'],'by_year':{},'cohort_conflicts':int((valid&(d.cfam.eq(1)!=d.genertn.eq(1))).sum())}
    for yr,sub in raw.loc[raw.age.between(19,39)].groupby('survey_year'):
        valid=sub.cfam.isin([1,2,3,4,5,6])&sub.genertn.isin([1,2,3,4,5,6,7])
        household_audit['by_year'][str(yr)]={'young_n':len(sub),'single':int(sub.cfam.eq(1).sum()),'unknown':int(living_alone(sub.cfam).isna().sum()),
             'cfam_genertn_conflict':int((valid&(sub.cfam.eq(1)!=sub.genertn.eq(1))).sum())}
    dump(out/'household_audit.json',household_audit)
    if household_audit['cohort_conflicts']:raise ValueError('Inconsistent household codes require review')
    manifest=json.loads((OLD/'split_manifest.json').read_text());parts=restore_parts(d,manifest);dump(out/'split_manifest.json',manifest)
    probs={};configs={};report={'plan':PLAN,'cohort_flow':flow,'models':{},'specialists':{}}
    probs['base_seed42'],configs['base_seed42']=load_frozen_nn(parts)
    print('Frozen baseline loaded; original split verified.',flush=True)
    for name,est in {'logistic_regression':LogisticRegression(max_iter=2000,C=1,random_state=42),
        'random_forest':RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=2),
        'xgboost':XGBClassifier(eval_metric='logloss',random_state=42,n_jobs=2),
        'lightgbm':LGBMClassifier(random_state=42,verbosity=-1,n_jobs=2)}.items():
        probs[name],configs[name]=baseline_fit(parts,name,est,out/'models'/name);print(name+' trained',flush=True)
    for seed in PLAN['seeds']:
        for living in [False,True]:
            name=f'{"household" if living else "base"}_seed{seed}'
            if name in probs:continue
            probs[name],configs[name]=net_fit(parts,living,seed,out/'models'/name,name);print(name+' trained',flush=True)
    # All fits/cutoffs fixed before reporting any current test results.
    for name,prob in probs.items():report['models'][name]=evaluate(parts,prob,configs[name])
    report['paired_differences']={}
    for seed in PLAN['seeds']:
        report['paired_differences'][f'household_minus_base_seed{seed}']=paired_ci(parts['test_2024'],probs[f'household_seed{seed}']['test_2024'],probs[f'base_seed{seed}']['test_2024'])
    for name in ['logistic_regression','random_forest','xgboost','lightgbm']:
        report['paired_differences'][f'base_seed42_minus_{name}']=paired_ci(parts['test_2024'],probs['base_seed42']['test_2024'],probs[name]['test_2024'])
    # Specialist eligibility is prespecified, not based on test scores.
    for group in ['living_1','living_0','age_19_29','age_30_39']:
        sub={k:v.loc[subgroup_masks(v)[group]].copy() for k,v in parts.items()}
        counts={k:{'n':len(v),'positive':int(v.target.sum()),'negative':int((v.target==0).sum())} for k,v in sub.items()}
        if len(sub['train'])<100 or any(min(c['positive'],c['negative'])<10 for c in counts.values()):
            report['specialists'][group]={'status':'insufficient_samples','counts':counts};continue
        prob,conf=net_fit(sub,False,42,out/'models'/('specialist_'+group),'specialist_'+group,group)
        common={k:probs['base_seed42'][k][subgroup_masks(parts[k])[group]] for k in parts}
        report['specialists'][group]={'status':'trained_exploratory','counts':counts,
            'specialist':evaluate(sub,prob,conf),'pooled_on_same_rows':evaluate(sub,common,configs['base_seed42']),
            'paired_specialist_minus_pooled':paired_ci(sub['test_2024'],prob['test_2024'],common['test_2024'])}
        print(group+' specialist complete',flush=True)
    report['prevalence_all_years']=prevalence(d)
    report['prevalence_2024']=prevalence(parts['test_2024'])
    report['associations']=associations(d)
    report['partitions']={k:{'n':len(v),'single':int(v.living_alone.eq(1).sum()),'multi':int(v.living_alone.eq(0).sum()),'unknown':int(v.living_alone.isna().sum())} for k,v in parts.items()}
    report['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'metabolic').glob('*.py')}
    report['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/raw/knhanes').glob('*.zip')}
    report['python']=platform.python_version()
    dump(out/'report.json',report)
    pred=pd.DataFrame({'row_in_cohort':parts['test_2024'].index,'target':parts['test_2024'].target.to_numpy(),
        'weight':weights(parts['test_2024']),'living_alone':parts['test_2024'].living_alone.to_numpy(),
        'sex':parts['test_2024'].sex.to_numpy(),'age':parts['test_2024'].age.to_numpy(),
        **{k:v['test_2024'] for k,v in probs.items()}})
    pred.to_json(out/'test_predictions.json',orient='records',indent=2)
    from .household import HouseholdPredictor
    state={'age':29,'sex':1,'HE_BMI':25.,'HE_wc':87.,'HE_ht':175.,'incm':2,'edu':4,'sm_presnt':0,'dr_month':1,'pa_aerobic':1,'cfam':1}
    dump(out/'example_request.json',state);dump(out/'example_response.json',HouseholdPredictor(out/'models/household_seed42').predict(state))
    print(json.dumps({'output':str(out),'models':{k:{policy:v['weighted']['roc_auc'] for policy,v in x['policies'].items()} for k,x in report['models'].items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',default='comparison_household_v1');args=ap.parse_args()
    if Path(args.run_name).name!=args.run_name:ap.error('Use a simple output directory name')
    main(args.run_name)
