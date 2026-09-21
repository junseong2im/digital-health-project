"""Screening-oriented metrics and full-frame stratified PSU resampling."""
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from .train import choose_threshold

POLICIES={'sensitivity90':.90,'sensitivity95':.95}

def screening_metrics(y,p,cut,w=None,discrimination=True):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    w=np.ones(len(y)) if w is None else np.asarray(w,dtype=float)
    if not len(y) or w.sum()<=0:return None
    if y.shape!=p.shape or y.shape!=w.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or not np.isfinite(w).all() or (w<0).any() or (p<0).any() or (p>1).any():raise ValueError('Invalid screening data')
    if not 0<=cut<=1:raise ValueError('Invalid cutoff')
    z=p>=cut;pos=y==1;neg=~pos
    tp=w[pos&z].sum();fn=w[pos&~z].sum();fp=w[neg&z].sum();tn=w[neg&~z].sum();total=w.sum()
    div=lambda a,b:float(a/b) if b>0 else None
    prev=float(w[pos].sum()/total)
    return {'n':len(y),'positive_n':int(pos.sum()),'negative_n':int(neg.sum()),'cutoff':float(cut),
        'prevalence':prev,'sensitivity':div(tp,tp+fn),'specificity':div(tn,tn+fp),'ppv':div(tp,tp+fp),'npv':div(tn,tn+fn),
        'miss_rate_among_positive':div(fn,tp+fn),'referral_rate':div(tp+fp,total),
        'referrals_per_1000':float(1000*(tp+fp)/total),'detected_per_1000':float(1000*tp/total),
        'missed_per_1000':float(1000*fn/total),'unnecessary_referrals_per_1000':float(1000*fp/total),
        'tests_per_detected':div(tp+fp,tp),'roc_auc':float(roc_auc_score(y,p,sample_weight=w)) if discrimination and tp+fn>0 and tn+fp>0 else None,
        'average_precision':float(average_precision_score(y,p,sample_weight=w)) if discrimination and tp+fn>0 else None,
        'brier':float(brier_score_loss(y,p,sample_weight=w)),
        'unweighted_confusion':{'tn':int((neg&~z).sum()),'fp':int((neg&z).sum()),'fn':int((pos&~z).sum()),'tp':int((pos&z).sum())}}

def select_screening_threshold(y,p,w,target):
    if target not in [.90,.95]:raise ValueError('Prespecified targets are 90% or 95%')
    y,p,w=np.asarray(y),np.asarray(p),np.asarray(w)
    if any(w[y==v].sum()<=0 for v in [0,1]):raise ValueError('Both outcomes need positive weight')
    return choose_threshold(y,p,w,target)

def net_benefit(y,p,w,threshold):
    if not 0<threshold<1:raise ValueError('Decision threshold must be between 0 and 1')
    y=np.asarray(y);p=np.asarray(p);w=np.asarray(w);selected=p>=threshold;ratio=threshold/(1-threshold)
    return float((w[selected&(y==1)].sum()-ratio*w[selected&(y==0)].sum())/w.sum())

def age_household_groups(d):
    age=d.age.to_numpy();live=d.living_alone.to_numpy();sex=d.sex.to_numpy()
    groups={'all_19_39':np.ones(len(d),bool),'age19':age==19,'age20_29':(age>=20)&(age<30),'age30_39':age>=30}
    for name,am in [('20_29',(age>=20)&(age<30)),('30_39',age>=30)]:
        for status in [0,1]:
            mask=am&(live==status);groups[f'age{name}_living{status}']=mask
            for s in [1,2]:groups[f'age{name}_living{status}_sex{s}']=mask&(sex==s)
    return groups

def load_full_frame(root):
    frames=[]
    for year in range(2021,2025):
        path=root/f'data/raw/knhanes/HN{str(year)[2:]}_ALL(SPSS)/HN{str(year)[2:]}_ALL.sav'
        d,_=pyreadstat.read_sav(str(path));d=d.copy();d['survey_year']=year;d['group']=str(year)+':'+d.psu.astype(str);frames.append(d)
    return pd.concat(frames,ignore_index=True)

class SurveyBootstrap:
    """Rao-Wu-style rescaled PSU bootstrap, no finite-population correction.

    Use all survey PSUs, including those with zero eligible domain observations.
    Singleton strata are held fixed and counted in metadata. This is an
    approximation, not an official KNHANES replicate-weight product.
    """
    def __init__(self,full,domain):
        years=set(domain.survey_year.unique());full=full[full.survey_year.isin(years)]
        units=full[['survey_year','kstrata','group']].dropna().drop_duplicates()
        if units.group.duplicated().any():raise ValueError('PSU mapped to multiple strata')
        self.units=units.reset_index(drop=True);self.lookup={g:i for i,g in enumerate(self.units.group)}
        self.strata=[group.index.to_numpy() for _,group in self.units.groupby(['survey_year','kstrata'])]
        self.row_unit=np.array([self.lookup[g] for g in domain.group]);self.weight=domain.wt_itvex.to_numpy(dtype=float)
        self.audit={'full_frame_psus':len(units),'domain_psus':domain.group.nunique(),
                    'strata':len(self.strata),'singleton_strata':sum(len(ix)==1 for ix in self.strata),
                    'method':'n_h-1 draws with replacement; multiplicity*n_h/(n_h-1); full survey PSUs; no FPC'}
    def draw(self,rng):
        multiplier=np.zeros(len(self.units))
        for ix in self.strata:
            n=len(ix)
            if n==1:multiplier[ix]=1;continue
            sampled=rng.choice(ix,size=n-1,replace=True);counts=np.bincount(sampled,minlength=len(self.units))
            multiplier[ix]=counts[ix]*n/(n-1)
        return self.weight*multiplier[self.row_unit]

def interval(values):
    v=np.array([x for x in values if x is not None and np.isfinite(x)],dtype=float)
    return {'ci95':np.quantile(v,[.025,.975]).tolist(),'valid_repeats':len(v)} if len(v) else {'ci95':None,'valid_repeats':0}
