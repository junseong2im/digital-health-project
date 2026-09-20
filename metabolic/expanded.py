"""Cross-year non-invasive feature harmonization and factorial NN runtime."""
import json
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit,softmax
from pydantic import Field,model_validator
from .data import features,BITS,FEATURES
from .decisions import Patient,render_decisions
from .fast import patient_vector
from .refined import CleanPreprocessor
from .specialized import group_indices

EXTRA_NUM=['sedentary_hours','walking_days']
EXTRA_CAT={'stress_level':[1,2,3,4],'employed':[0,1],'alcohol_frequency':[0,1,2,3,4,5],
           'alcohol_amount':[0,1,2,3,4,5],'strength_days':[0,1,2,3,4,5],
           'family_history':[0,1],'living_alone':[0,1]}
EXTRA=EXTRA_NUM+list(EXTRA_CAT)
SOURCE=['cfam','EC1_1','BP1','BD1','BD1_11','BD2_1','BE3_31','BE5_1','BE8_1','BE8_2','HE_fh']

def extended_features(d):
    x=features(d)
    h=d.BE8_1;m=d.BE8_2
    sedentary=(h+m/60).where(h.between(0,24)&m.between(0,59)&(h+m/60).le(24))
    x['sedentary_hours']=sedentary
    x['walking_days']=(d.BE3_31-1).where(d.BE3_31.isin(range(1,9)))
    x['stress_level']=d.BP1.where(d.BP1.isin([1,2,3,4]))
    x['employed']=d.EC1_1.map({1:1,2:0})
    frequency=(d.BD1_11-1).where(d.BD1_11.isin(range(1,7)))
    frequency=frequency.mask(d.BD1.eq(1)&d.BD1_11.eq(8),0)
    amount=d.BD2_1.where(d.BD2_1.isin(range(1,6)))
    non_drinker=d.BD1.eq(1)|d.BD1_11.eq(1)
    amount=amount.mask(non_drinker&d.BD2_1.eq(8),0)
    x['alcohol_frequency']=frequency;x['alcohol_amount']=amount
    x['strength_days']=(d.BE5_1-1).where(d.BE5_1.isin(range(1,7)))
    x['family_history']=d.HE_fh.where(d.HE_fh.isin([0,1]))
    x['living_alone']=d.cfam.eq(1).astype(float).where(d.cfam.isin(range(1,7)))
    return x[FEATURES+EXTRA].astype(float)

class ExpandedPreprocessor:
    def __init__(self,expanded=True):self.expanded=expanded
    def fit(self,x):
        x=np.asarray(x,float);self.base=CleanPreprocessor('hinge').fit(x[:,:10])
        if self.expanded:
            n=x[:,10:12];self.median=np.nanmedian(n,axis=0)
            if not np.isfinite(self.median).all():raise ValueError('Empty extra numeric field')
            n=np.where(np.isnan(n),self.median,n);self.mean=n.mean(axis=0);self.scale=n.std(axis=0);self.scale[self.scale<1e-8]=1
        else:self.median=self.mean=np.zeros(2);self.scale=np.ones(2)
        self._compile();return self
    def _compile(self):
        body,context,behavior=group_indices(40);self.groups=[body,context,behavior]
        self.lookups=[];cursor=44
        if self.expanded:
            behavior.extend([40,41,42,43])
            for name,values in EXTRA_CAT.items():
                mapping={values[0]:None};mapping.update({v:cursor+i for i,v in enumerate(values[1:])})
                self.lookups.append((mapping,cursor+len(values)-1))
                group=context if name in ['employed','family_history','living_alone'] else behavior
                group.extend(range(cursor,cursor+len(values)));cursor+=len(values)
        self.output_dim=cursor if self.expanded else 40
    def transform(self,x):
        x=np.asarray(x,float);single=x.ndim==1
        if single:
            expected=19 if self.expanded else 10
            if x.shape!=(expected,) or np.isinf(x).any():raise ValueError('Invalid feature vector')
            base=self.base.transform(x[:10])
            if not self.expanded:return base
            out=np.zeros(self.output_dim);out[:40]=base
            n=x[10:12];missing=np.isnan(n);out[40:42]=(np.where(missing,self.median,n)-self.mean)/self.scale;out[42:44]=missing
            for j,(mapping,misscol) in enumerate(self.lookups):
                val=x[12+j]
                if np.isnan(val):out[misscol]=1
                elif val not in mapping:raise ValueError('Unknown extra category')
                elif mapping[val] is not None:out[mapping[val]]=1
            return out
        return np.stack([self.transform(row) for row in x])
    def save(self,path):
        p=Path(path);self.base.save(p/'base_preprocessor.npz')
        np.savez(p/'extra_preprocessor.npz',expanded=self.expanded,median=self.median,mean=self.mean,scale=self.scale)
    @classmethod
    def load(cls,path):
        p=Path(path);a=dict(np.load(p/'extra_preprocessor.npz'));obj=cls(bool(a['expanded']))
        obj.base=CleanPreprocessor.load(p/'base_preprocessor.npz')
        for k in ['median','mean','scale']:setattr(obj,k,a[k])
        obj._compile();return obj

class FactorialNet(nn.Module):
    def __init__(self,pre,multitask):
        super().__init__();self.multitask=multitask;self.input_dim=pre.output_dim
        for name,ix in zip(['body','context','behavior'],pre.groups):
            self.register_buffer(name+'_indices',torch.tensor(ix,dtype=torch.long));setattr(self,name,nn.Linear(len(ix),4))
        self.output=nn.Linear(pre.output_dim+20,5 if multitask else 1)
        self.register_buffer('bits',torch.tensor(BITS[1:]))
    def forward(self,x):
        a,c,b=[torch.tanh(getattr(self,n)(x[...,getattr(self,n+'_indices')])) for n in ['body','context','behavior']]
        return self.output(torch.cat([x,a,c,b,a*c,a*b],dim=-1))
    def initialize(self,lr_models,x):
        with torch.no_grad():
            self.output.weight.zero_();self.output.bias.zero_()
            for j,lr in enumerate(lr_models):
                self.output.weight[j,:self.input_dim]=torch.tensor(lr.coef_[0],dtype=torch.float32);self.output.bias[j]=float(lr.intercept_[0])
            # Never-seen indicator columns must not acquire random effects at serving.
            constant=np.std(x,axis=0)<1e-12
            for name in ['body','context','behavior']:
                ix=getattr(self,name+'_indices').numpy();getattr(self,name).weight[:,constant[ix]]=0
    def joint(self,z):
        if not self.multitask:raise ValueError('Single-task model has no trained component output')
        r=torch.sigmoid(z[:,0]);q=torch.softmax(z[:,1:]@self.bits.T,dim=-1)
        return torch.cat([(1-r)[:,None],r[:,None]*q],dim=-1)

class ExpandedPatient(Patient):
    sedentary_hours:float|None=Field(default=None,ge=0,le=24)
    walking_days:Literal[0,1,2,3,4,5,6,7]|None=None
    stress_level:Literal[1,2,3,4]|None=None
    employed:Literal[0,1]|None=None
    alcohol_frequency:Literal[0,1,2,3,4,5]|None=None
    alcohol_amount:Literal[0,1,2,3,4,5]|None=None
    strength_days:Literal[0,1,2,3,4,5]|None=None
    family_history:Literal[0,1]|None=None
    living_alone:Literal[0,1]|None=None

    @model_validator(mode='after')
    def consistent_alcohol_information(self):
        if self.alcohol_frequency==0 and self.alcohol_amount not in (None,0):
            raise ValueError('No alcohol use in the past year requires amount 0 or unknown')
        if self.alcohol_frequency is not None and self.dr_month is not None:
            if int(self.alcohol_frequency>=2)!=self.dr_month:
                raise ValueError('Monthly drinking and alcohol frequency are inconsistent')
        return self

class ExpandedPredictor:
    def __init__(self,artifact):
        path=Path(artifact);self.config=json.loads((path/'config.json').read_text());self.pre=ExpandedPreprocessor.load(path)
        self.p=dict(np.load(path/'network.npz'));self.fused=np.zeros((12,self.pre.output_dim),dtype='float32')
        self.bias=np.concatenate([self.p[n+'.bias'] for n in ['body','context','behavior']])
        for k,n in enumerate(['body','context','behavior']):self.fused[4*k:4*k+4,self.p[n+'_indices']]=self.p[n+'.weight']
    def logits(self,x):
        x=np.asarray(x,dtype='float32');h=np.tanh(x@self.fused.T+self.bias);a,c,b=h[...,:4],h[...,4:8],h[...,8:]
        return np.concatenate([x,a,c,b,a*c,a*b],axis=-1)@self.p['output.weight'].T+self.p['output.bias']
    def predict(self,state,policy='youden'):
        schema=ExpandedPatient if self.pre.expanded else Patient;person=schema.model_validate(state)
        data=person.model_dump();base={k:v for k,v in data.items() if k not in EXTRA}
        vector=patient_vector(Patient.model_validate(base))
        if self.pre.expanded:vector=np.r_[vector,np.array([data[k] for k in EXTRA],dtype=float)]
        if policy not in self.config['thresholds']:raise ValueError('Unknown threshold policy')
        z=self.logits(self.pre.transform(vector));r=float(expit(z[0]/self.config['temperature']))
        if self.config['multitask']:
            q=softmax(z[1:]@BITS[1:].T/self.config['pattern_temperature']);p=np.r_[1-r,r*q]
            result=render_decisions(p,self.config['thresholds'][policy])
        else:
            result={'probability':r,'positive':r>=self.config['thresholds'][policy],
                    'component_predictions_available':False,'research_only':True}
        result.update({'model':self.config['name'],'policy':policy,'missing_inputs':[k for k,v in data.items() if v is None],
                       'exploratory_extension':True,'living_alone_definition':'Single-person household; not verified independent living'})
        return result

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--artifact',required=True);a.add_argument('--input',required=True)
    a.add_argument('--policy',choices=['youden','sensitivity90'],default='youden');args=a.parse_args()
    payload=json.loads(Path(args.input).read_text(encoding='utf-8-sig'))
    print(json.dumps(ExpandedPredictor(args.artifact).predict(payload.get('state',payload),args.policy),ensure_ascii=False,indent=2,allow_nan=False))
