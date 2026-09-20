"""Train-only preprocessing and a primary-risk/conditional-pattern neural model."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit,softmax
from .data import FEATURES,CATEGORIES
from .decisions import Patient,render_decisions
from .fast import patient_vector

class CleanPreprocessor:
    def __init__(self,basis='standard'):
        if basis not in ['standard','hinge']:raise ValueError('Unknown basis')
        self.basis=basis

    def fit(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape[1]!=10:raise ValueError('Ten features required')
        self.median=np.nanmedian(x[:,:4],axis=0)
        if not np.isfinite(self.median).all():raise ValueError('Empty numeric training column')
        n=np.where(np.isnan(x[:,:4]),self.median,x[:,:4])
        self.mean=n.mean(axis=0);self.scale=n.std(axis=0);self.scale[self.scale<1e-8]=1
        z=(n-self.mean)/self.scale
        self.knots=np.quantile(z,[.25,.5,.75],axis=0).T
        self._compile()
        return self

    def _compile(self):
        cursor=8;self.indexes=[]
        for values in CATEGORIES.values():
            mapping={values[0]:None};mapping.update({v:cursor+i for i,v in enumerate(values[1:])})
            self.indexes.append((mapping,cursor+len(values)-1));cursor+=len(values)
        self.base_dim=cursor;self.output_dim=cursor+(16 if self.basis=='hinge' else 0)

    def transform(self,x):
        x=np.asarray(x,dtype=float);single=x.ndim==1
        if single:
            if x.shape!=(10,) or np.isinf(x).any():raise ValueError('Invalid features')
            n=x[:4];missing=np.isnan(n);n=(np.where(missing,self.median,n)-self.mean)/self.scale
            out=np.zeros(self.output_dim,dtype=float);out[:4]=n;out[4:8]=missing
            for j,(mapping,missing_col) in enumerate(self.indexes):
                value=x[4+j]
                if np.isnan(value):out[missing_col]=1
                elif value not in mapping:raise ValueError('Unknown category')
                elif mapping[value] is not None:out[mapping[value]]=1
            if self.basis=='hinge':
                out[self.base_dim:self.base_dim+12]=np.maximum(n[:,None]-self.knots,0).ravel()
                out[self.base_dim+12:]=n*(x[4]==1)
            return out
        if x.shape[1]!=10 or np.isinf(x).any():raise ValueError('Invalid features')
        n=x[:,:4];missing=np.isnan(n);n=(np.where(missing,self.median,n)-self.mean)/self.scale
        blocks=[n,missing.astype(float)]
        for j,(name,values) in enumerate(CATEGORIES.items()):
            c=x[:,4+j]
            if (~(np.isnan(c)|np.isin(c,values))).any():raise ValueError(f'Unknown category: {name}')
            levels=values[1:]
            blocks += [(c[:,None]==np.asarray(levels)).astype(float),np.isnan(c)[:,None].astype(float)]
        if self.basis=='hinge':
            blocks.append(np.maximum(n[:,:,None]-self.knots[None,:,:],0).reshape(len(x),-1))
            # Prespecified sex-dependent slopes, not data-selected interactions.
            blocks.append(n*(x[:,4]==1)[:,None])
        out=np.concatenate(blocks,axis=1)
        return out[0] if single else out

    def save(self,path):
        np.savez(path,basis=self.basis,median=self.median,mean=self.mean,scale=self.scale,knots=self.knots)

    @classmethod
    def load(cls,path):
        a=np.load(path,allow_pickle=False);obj=cls(str(a['basis']))
        for key in ['median','mean','scale','knots']:setattr(obj,key,a[key])
        obj._compile()
        return obj

class RiskPatternNet(nn.Module):
    def __init__(self,input_dim,hidden):
        super().__init__();self.hidden=nn.Linear(input_dim,hidden);self.output=nn.Linear(input_dim+hidden,16)

    def forward(self,x):
        return self.output(torch.cat([x,torch.tanh(self.hidden(x))],dim=1))

    def initialize_risk(self,coef,bias):
        with torch.no_grad():
            self.output.weight[0].zero_();self.output.weight[0,:len(coef)]=torch.tensor(coef,dtype=torch.float32)
            self.output.bias[0]=float(bias)

def joint_from_logits(logits,risk_temperature=1.,pattern_temperature=1.):
    a=np.asarray(logits);r=expit(a[...,0]/risk_temperature)
    c=softmax(a[...,1:]/pattern_temperature,axis=-1)
    return np.concatenate([(1-r)[...,None],r[...,None]*c],axis=-1)

class RefinedPredictor:
    def __init__(self,artifact):
        path=Path(artifact);self.config=json.loads((path/'config.json').read_text())
        self.pre=CleanPreprocessor.load(path/'preprocessor.npz')
        self.p=dict(np.load(path/'network.npz',allow_pickle=False))

    def logits(self,x):
        x=np.asarray(x,dtype=np.float32);p=self.p
        h=np.tanh(x@p['hidden.weight'].T+p['hidden.bias'])
        return np.concatenate([x,h],axis=-1)@p['output.weight'].T+p['output.bias']

    def probability(self,vector):
        return float(expit(self.logits(self.pre.transform(vector))[0]/self.config['risk_temperature']))

    def predict(self,state,questions=None,policy='sensitivity90'):
        patient=Patient.model_validate(state)
        if policy not in self.config['thresholds']:raise ValueError('Unknown threshold policy')
        logits=self.logits(self.pre.transform(patient_vector(patient)))
        p=joint_from_logits(logits,self.config['risk_temperature'],self.config['pattern_temperature'])
        result=render_decisions(p,self.config['thresholds'][policy],questions)
        result.update({'model':self.config['name'],'missing_inputs':[k for k,v in patient.model_dump().items() if v is None],
                       'threshold_policy':policy,'exploratory_extension':True})
        return result
