"""Equivalent array-only inference for the frozen v1 network and logistic baseline."""
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.special import expit,softmax
from .data import NUMERIC,CATEGORIES,FEATURES
from .decisions import Patient,render_decisions

def patient_vector(patient):
    d=patient.model_dump()
    ratio=d['HE_wc']/d['HE_ht'] if d['HE_wc'] is not None and d['HE_ht'] is not None else np.nan
    numeric=[d['age'],d['HE_BMI'],d['HE_wc'],ratio]
    return np.asarray(numeric+[d[k] for k in CATEGORIES],dtype=float)

class ArrayPreprocessor:
    def __init__(self,pre):
        n=pre.named_transformers_['numeric'];c=pre.named_transformers_['category']
        self.median=n['impute'].statistics_.copy()
        self.indicators=n['impute'].indicator_.features_.copy()
        self.mean=n['scale'].mean_.copy();self.scale=n['scale'].scale_.copy()
        self.cats=[a.copy() for a in c['encode'].categories_]
        cursor=len(self.mean);self.indexes=[]
        for values in self.cats:
            self.indexes.append({float(v):cursor+i for i,v in enumerate(values)});cursor+=len(values)
        self.output_dim=cursor

    def transform(self,x):
        x=np.asarray(x,dtype=float);single=x.ndim==1
        if single:
            if x.shape!=(len(FEATURES),):raise ValueError('Unexpected feature count')
            missing=np.isnan(x[:4]);n=np.where(missing,self.median,x[:4])
            out=np.zeros(self.output_dim,dtype=float)
            n=np.concatenate([n,missing[self.indicators]])
            out[:len(n)]=(n-self.mean)/self.scale
            for j,lookup in enumerate(self.indexes):
                value=x[4+j];value=-1 if np.isnan(value) else float(value)
                if value not in lookup:raise ValueError('Unknown category')
                out[lookup[value]]=1
            return out
        if x.shape[1]!=len(FEATURES):raise ValueError('Unexpected feature count')
        n=x[:,:4];missing=np.isnan(n);n=np.where(missing,self.median,n)
        n=np.concatenate([n,missing[:,self.indicators]],axis=1)
        n=(n-self.mean)/self.scale
        cat=x[:,4:];cat=np.where(np.isnan(cat),-1,cat)
        encoded=np.concatenate([(cat[:,j,None]==values).astype(float) for j,values in enumerate(self.cats)],axis=1)
        if np.any(encoded.sum(axis=1)!=len(self.cats)):raise ValueError('Unknown category')
        out=np.concatenate([n,encoded],axis=1)
        return out[0] if single else out

class ArrayNetwork:
    def __init__(self,state):
        self.p={k:v.detach().cpu().numpy().copy() for k,v in state.items()}

    def logits(self,x):
        p=self.p;x=np.asarray(x,dtype=np.float32)
        x=x@p['net.0.weight'].T+p['net.0.bias']
        x=(x-x.mean(axis=-1,keepdims=True))/np.sqrt(x.var(axis=-1,keepdims=True)+1e-5)
        x=x*p['net.1.weight']+p['net.1.bias'];x=x*expit(x)
        x=x@p['net.4.weight'].T+p['net.4.bias'];x=x*expit(x)
        return x@p['net.7.weight'].T+p['net.7.bias']

class FastPredictor:
    def __init__(self,artifact):
        path=Path(artifact);self.config=json.loads((path/'config.json').read_text())
        self.pre=ArrayPreprocessor(joblib.load(path/'preprocessor.joblib'))
        ck=torch.load(path/'network.pt',map_location='cpu',weights_only=True)
        self.net=ArrayNetwork(ck['state_dict'])

    def predict(self,state,questions=None):
        patient=Patient.model_validate(state)
        x=self.pre.transform(patient_vector(patient))
        p=softmax(self.net.logits(x)/self.config['temperature'])
        result=render_decisions(p,self.config['threshold'],questions)
        result['missing_inputs']=[k for k,v in patient.model_dump().items() if v is None]
        result['model']=self.config['run_name']
        return result

class FastLogistic:
    def __init__(self,artifact):
        path=Path(artifact);self.config=json.loads((path/'config.json').read_text())
        b=joblib.load(path/'pipeline.joblib');self.pre=ArrayPreprocessor(b['preprocessor'])
        self.coef=b['model'].coef_[0].copy();self.bias=float(b['model'].intercept_[0])

    def probability(self,vector):
        x=self.pre.transform(vector)
        # Equivalent to probability clipping followed by logit/temperature.
        z=np.clip(x@self.coef+self.bias,-16.118095550958316,16.118095550958316)
        return expit(z/self.config['temperature'])
