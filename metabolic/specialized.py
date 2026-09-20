"""Domain-grouped neural screening with coherent four-finding interactions."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import softmax,expit
from .data import BITS
from .decisions import Patient,render_decisions
from .fast import patient_vector
from .refined import CleanPreprocessor

PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

def pattern_features(pairwise=True):
    bits=BITS[1:]
    return np.concatenate([bits,np.stack([bits[:,a]*bits[:,b] for a,b in PAIRS],axis=1)],axis=1) if pairwise else bits.copy()

def group_indices(input_dim):
    # CleanPreprocessor contract: 4 numeric, 4 missing flags, 16 categorical.
    body=[1,2,3,5,6,7]
    context=[0,4,*range(8,18)]
    behavior=list(range(18,24))
    if input_dim==40:
        body += list(range(27,36))+[37,38,39]
        context += [24,25,26,36]
    elif input_dim!=24:raise ValueError('Unexpected preprocessor layout')
    return body,context,behavior

class MetabolicInteractionNet(nn.Module):
    def __init__(self,input_dim,hidden=4,mode='grouped',pairwise=True):
        super().__init__()
        if mode not in ['flat','grouped']:raise ValueError('Invalid mode')
        self.mode=mode;self.pairwise=pairwise;self.input_dim=input_dim
        self.register_buffer('patterns',torch.tensor(pattern_features(pairwise)))
        self.register_buffer('positive_bits',torch.tensor(BITS[1:]))
        if mode=='flat':
            self.flat=nn.Linear(input_dim,hidden);dim=input_dim+hidden
        else:
            body,context,behavior=group_indices(input_dim)
            for name,ix in [('body',body),('context',context),('behavior',behavior)]:
                self.register_buffer(name+'_indices',torch.tensor(ix,dtype=torch.long))
                setattr(self,name,nn.Linear(len(ix),hidden))
            dim=input_dim+5*hidden
        self.output=nn.Linear(dim,1+self.patterns.shape[1])

    def forward(self,x):
        if self.mode=='flat':z=torch.cat([x,torch.tanh(self.flat(x))],dim=-1)
        else:
            a=torch.tanh(self.body(x[...,self.body_indices]))
            c=torch.tanh(self.context(x[...,self.context_indices]))
            b=torch.tanh(self.behavior(x[...,self.behavior_indices]))
            z=torch.cat([x,a,c,b,a*c,a*b],dim=-1)
        return self.output(z)

    def initialize(self,primary,components):
        with torch.no_grad():
            self.output.weight.zero_();self.output.bias.zero_()
            self.output.weight[0,:self.input_dim]=torch.tensor(primary.coef_[0],dtype=torch.float32)
            self.output.bias[0]=float(primary.intercept_[0])
            for j,model in enumerate(components):
                self.output.weight[j+1,:self.input_dim]=torch.tensor(model.coef_[0],dtype=torch.float32)
                self.output.bias[j+1]=float(model.intercept_[0])

    def joint(self,logits):
        r=torch.sigmoid(logits[...,0])
        q=torch.softmax(logits[...,1:]@self.patterns.T,dim=-1)
        return torch.cat([(1-r).unsqueeze(-1),r.unsqueeze(-1)*q],dim=-1)

def joint_numpy(logits,pairwise=True,risk_temperature=1.,pattern_temperature=1.):
    r=expit(logits[...,0]/risk_temperature)
    q=softmax(logits[...,1:]@pattern_features(pairwise).T/pattern_temperature,axis=-1)
    return np.concatenate([(1-r)[...,None],r[...,None]*q],axis=-1)

class SpecializedPredictor:
    def __init__(self,artifact):
        p=Path(artifact);self.config=json.loads((p/'config.json').read_text())
        self.pre=CleanPreprocessor.load(p/'preprocessor.npz')
        self.p=dict(np.load(p/'network.npz',allow_pickle=False))
        if self.config['mode']=='grouped':
            hidden=self.p['body.bias'].shape[0]
            self.fused_weight=np.zeros((3*hidden,self.pre.output_dim),dtype=np.float32)
            self.fused_bias=np.concatenate([self.p[n+'.bias'] for n in ['body','context','behavior']])
            for k,name in enumerate(['body','context','behavior']):
                self.fused_weight[k*hidden:(k+1)*hidden,self.p[name+'_indices']]=self.p[name+'.weight']
            self.hidden_width=hidden

    def logits(self,x):
        p=self.p;x=np.asarray(x,dtype='float32')
        if self.config['mode']=='flat':
            z=np.concatenate([x,np.tanh(x@p['flat.weight'].T+p['flat.bias'])],axis=-1)
        else:
            fused=np.tanh(x@self.fused_weight.T+self.fused_bias);h=self.hidden_width
            a,c,b=fused[...,:h],fused[...,h:2*h],fused[...,2*h:]
            z=np.concatenate([x,a,c,b,a*c,a*b],axis=-1)
        return z@p['output.weight'].T+p['output.bias']

    def probability(self,vector):
        z=self.logits(self.pre.transform(vector))
        return float(expit(z[0]/self.config['risk_temperature']))

    def predict(self,state,questions=None,policy='sensitivity90'):
        patient=Patient.model_validate(state)
        if policy not in self.config['thresholds']:raise ValueError('Unknown threshold policy')
        z=self.logits(self.pre.transform(patient_vector(patient)))
        p=joint_numpy(z,self.config['pairwise'],self.config['risk_temperature'],self.config['pattern_temperature'])
        result=render_decisions(p,self.config['thresholds'][policy],questions)
        result.update({'model':self.config['name'],'threshold_policy':policy,'exploratory_extension':True,
            'missing_inputs':[k for k,v in patient.model_dump().items() if v is None],
            'calibration_method':'Out-of-fold development predictions; final refit calibration transfer requires external validation'})
        return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--input',required=True)
    parser.add_argument('--policy',choices=['youden','sensitivity90'],default='sensitivity90');args=parser.parse_args()
    payload=json.loads(Path(args.input).read_text(encoding='utf-8-sig'))
    print(json.dumps(SpecializedPredictor(args.artifact).predict(payload.get('state',payload),payload.get('questions'),args.policy),ensure_ascii=False,indent=2,allow_nan=False))
