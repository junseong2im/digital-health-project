"""Age-specific household follow-up, selection audit and weighted associations."""
import argparse,json,hashlib,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from threadpoolctl import threadpool_limits
from .data import load_cohort,features
from .household import living_alone
from .comparison import ROOT,dump
from .screening import load_full_frame,SurveyBootstrap,age_household_groups,interval

PLAN={'date':'2026-09-21','primary_ages':['20-29','30-39'],'age19':'Reported separately, not labelled twenties',
    'exposure':'cfam=1 versus cfam=2..6; genertn consistency check; not direct independent living/self-catering',
    'analyses':['age x household x sex weighted prevalence with full-frame PSU uncertainty',
        'never-married (marri_1=2) sensitivity analysis','descriptive selection/retention by age and household',
        'survey-weighted logistic association with age interaction and standardized predicted prevalence',
        'separate screening_v5 global-cutoff group evaluation'],
    'adjustment_A':'within-band age, sex, income quartile, education, survey year, ever-married status, employment',
    'adjustment_B':'A plus waist circumference, current smoking, monthly drinking and aerobic activity; possible overadjustment, not causal',
    'uncertainty':'300 full-frame stratified rescaled PSU replicates; GLM refit; no naive frequency-weight Wald p-values',
    'inference':'Cross-sectional descriptive association. No claim that living alone protects or harms health. Multiple contrasts exploratory.'}

def weighted_rate(y,w):return float(np.average(y,weights=w)) if len(y) and np.sum(w)>0 else None

def prevalence_tables(d,full):
    masks=age_household_groups(d)
    for age in ['20_29','30_39']:
        for status in [0,1]:masks[f'never_married_age{age}_living{status}']=masks[f'age{age}_living{status}']&d.marri_1.eq(2).to_numpy()
    y=d.target.to_numpy();w=d.wt_itvex.to_numpy();bootstrap=SurveyBootstrap(full,d)
    rows={};samples={};gap_samples={'20_29':[],'30_39':[]}
    for name,mask in masks.items():
        if not mask.any():continue
        count=int(mask.sum());positive=int(y[mask].sum())
        rows[name]={'n':count,'positive':positive,'negative':count-positive,'weighted_prevalence':weighted_rate(y[mask],w[mask]),
            'unweighted_prevalence':float(y[mask].mean()),'low_information':min(positive,count-positive)<20}
        samples[name]=[]
    rng=np.random.default_rng(20260922)
    for _ in range(300):
        rw=bootstrap.draw(rng)
        for name in rows:samples[name].append(weighted_rate(y[masks[name]],rw[masks[name]]))
        for band in gap_samples:
            a=samples['age'+band+'_living1'][-1];b=samples['age'+band+'_living0'][-1]
            gap_samples[band].append(a-b if a is not None and b is not None else None)
    for name in rows:rows[name]['ci95']=interval(samples[name])
    gaps={band:{'point':rows['age'+band+'_living1']['weighted_prevalence']-rows['age'+band+'_living0']['weighted_prevalence'],**interval(v)} for band,v in gap_samples.items()}
    return {'groups':rows,'living1_minus_living0':gaps,'survey_design':bootstrap.audit}

def selection_audit(full,d):
    raw=full[full.age.between(20,39)].copy();raw['living_alone']=living_alone(raw.cfam)
    keys=set(zip(d.survey_year,d.ID));raw['included']=[(r.survey_year,r.ID) in keys for r in raw.itertuples()]
    known_no_diagnosis=raw[['DI1_dg','DI2_dg','DE1_dg']].eq(0).all(axis=1)
    no_med=raw.DI1_2.isin([5,8])&raw.DI2_2.isin([5,8])&raw.DE1_31.isin([0,8])&raw.DE1_32.isin([0,8])
    raw['pre_exam_eligible']=known_no_diagnosis&no_med&raw.HE_dprg.isna()
    out={}
    for name,mask in age_household_groups(raw).items():
        if '_sex' in name or name=='age19' or not mask.any():continue
        x=raw.loc[mask];pre=x[x.pre_exam_eligible]
        label='all_20_39' if name=='all_19_39' else name
        out[label]={'all_survey_young':len(x),'pre_exam_eligible':len(pre),'included':int(pre.included.sum()),
            'not_included_after_exam_filters':int((~pre.included).sum()),'retention_unweighted':float(pre.included.mean()) if len(pre) else None}
    return out

def association_tables(d,full):
    # Same complete-covariate domain for both adjustment specifications.
    a=features(d);a['target']=d.target;a['living_alone']=d.living_alone;a['survey_year']=d.survey_year
    a['group']=d.group;a['kstrata']=d.kstrata;a['wt_itvex']=d.wt_itvex
    a['ever_married']=d.marri_1.map({1:1,2:0});a['employed']=d.EC1_1.map({1:1,2:0})
    a=a.loc[a.age>=20].copy();a['age30']=(a.age>=30).astype(int);a['age_within_band']=a.age-np.where(a.age30,30,20)
    required=['target','living_alone','age','sex','incm','edu','HE_wc','sm_presnt','dr_month','pa_aerobic','ever_married','employed']
    a=a.dropna(subset=required).copy()
    formula='target ~ living_alone * age30 + age_within_band + C(sex) + C(incm) + C(edu) + C(survey_year) + ever_married + employed'
    formulas={'demographic':formula,'behavior_waist':formula+' + HE_wc + sm_presnt + dr_month + pa_aerobic'}
    boot=SurveyBootstrap(full,a);rng=np.random.default_rng(20260923);weights=a.wt_itvex.to_numpy()
    prepared={};out={}
    def summarize(beta,x0,x1,w):
        from scipy.special import expit
        p0=expit(x0@beta);p1=expit(x1@beta);values={}
        for age in [0,1]:
            mask=a.age30.to_numpy()==age;label='20_29' if age==0 else '30_39'
            r0=weighted_rate(p0[mask],w[mask]);r1=weighted_rate(p1[mask],w[mask])
            values[label]={'predicted_multi':r0,'predicted_single':r1,'difference':r1-r0,'ratio':r1/r0}
        values['difference_interaction']=values['30_39']['difference']-values['20_29']['difference']
        return values
    for name,f in formulas.items():
        yy,xx=patsy.dmatrices(f,a,return_type='dataframe');x=xx.to_numpy();y=yy.to_numpy().ravel()
        frames=[]
        for status in [0,1]:
            changed=a.copy();changed['living_alone']=status
            frames.append(np.asarray(patsy.build_design_matrices([xx.design_info],changed)[0]))
        fit=sm.GLM(y,x,family=sm.families.Binomial(),freq_weights=weights/weights.mean()).fit(maxiter=100)
        if not fit.converged:raise RuntimeError('Association point model did not converge')
        prepared[name]=(x,y,*frames);point=summarize(fit.params,*frames,weights)
        out[name]={'formula':f,'n':len(a),'point':point,'coef_names':xx.columns.tolist(),'coefficients':fit.params.tolist()}
    samples={name:{band:{key:[] for key in ['predicted_multi','predicted_single','difference','ratio']} for band in ['20_29','30_39']} for name in formulas}
    interactions={name:[] for name in formulas};valid={name:0 for name in formulas};failures={name:0 for name in formulas}
    for rep in range(300):
        rw=boot.draw(rng)
        for name,(x,y,x0,x1) in prepared.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error',PerfectSeparationWarning)
                    fit=sm.GLM(y,x,family=sm.families.Binomial(),freq_weights=rw/rw.mean()).fit(maxiter=100)
                if not fit.converged or not np.isfinite(fit.params).all():raise ValueError('Failed replicate')
                result=summarize(fit.params,x0,x1,rw)
                for band in samples[name]:
                    for key in samples[name][band]:samples[name][band][key].append(result[band][key])
                interactions[name].append(result['difference_interaction']);valid[name]+=1
            except (ValueError,np.linalg.LinAlgError,Warning):failures[name]+=1
        if (rep+1)%100==0:print('household adjusted uncertainty',rep+1,'/300',flush=True)
    for name in formulas:
        if valid[name]<285:raise RuntimeError('Too few converged association replicates')
        out[name]['ci95']={band:{key:interval(v) for key,v in values.items()} for band,values in samples[name].items()}
        out[name]['difference_interaction_ci95']=interval(interactions[name]);out[name]['failed_replicates']=failures[name]
    return {'models':out,'design':boot.audit,'interpretation':'Survey-weighted observational standardization, not a causal effect or a population-wide estimate beyond this selected complete-case cohort.'}

def main(name):
    out=ROOT/'artifacts'/name;out.mkdir(parents=True,exist_ok=False);dump(out/'PLAN.json',PLAN)
    d,_=load_cohort(ROOT);d=d.copy();d['living_alone']=living_alone(d.cfam);full=load_full_frame(ROOT)
    report={'plan':PLAN,'age19_n':int(d.age.eq(19).sum()),'age20_39_n':int(d.age.between(20,39).sum()),
        'descriptive':prevalence_tables(d,full),'selection_audit':selection_audit(full,d)}
    print('Age-specific household prevalence and selection audit done.',flush=True)
    with threadpool_limits(limits=1):report['adjusted']=association_tables(d,full)
    report['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'metabolic').glob('*.py')}
    dump(out/'report.json',report)
    print(json.dumps({'output':str(out),'age20_39_n':report['age20_39_n'],'prevalence':{k:v for k,v in report['descriptive']['groups'].items() if k in ['age20_29_living0','age20_29_living1','age30_39_living0','age30_39_living1']}},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='household_v5');a=p.parse_args()
    if Path(a.run_name).name!=a.run_name:p.error('Invalid artifact directory')
    main(a.run_name)
