"""Local architecture adaptation: typed atomic views over one network pass.

No vendor SDK, API key, remote inference, teacher outputs or RLCD implementation.
"""
import json
from pathlib import Path
from typing import Literal
import numpy as np
from scipy.special import expit,softmax
from pydantic import BaseModel,ConfigDict,Field,model_validator
from .data import BITS,COMPONENTS
from .expanded import ExpandedPredictor,ExpandedPatient,EXTRA
from .decisions import Patient
from .fast import patient_vector

Target=Literal['any_abnormality','elevated_glucose','elevated_bp','elevated_tg','low_hdl','abnormality_count','pattern']

class AtomicQuestion(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    type:Literal['choice','score','noul']
    target:Target
    labels:list[str]|None=None
    minimum:float=Field(default=0,ge=-1e6,le=1e6)
    maximum:float=Field(default=4,ge=-1e6,le=1e6)
    @model_validator(mode='after')
    def valid_contract(self):
        if self.type=='score':
            if self.target!='abnormality_count' or not self.maximum>self.minimum:raise ValueError('Score requires count target and increasing finite range')
            if self.labels is not None:raise ValueError('Score levels are defined by the count legend')
        elif self.type=='noul':
            if self.target in ['abnormality_count','pattern'] or self.labels is not None:raise ValueError('Noul supports only trained binary events')
        else:
            expected=16 if self.target=='pattern' else 2
            if self.target=='abnormality_count':raise ValueError('Use Score for ordered count')
            if self.labels is not None and (len(self.labels)!=expected or len(set(self.labels))!=expected or any(not s.strip() for s in self.labels)):
                raise ValueError(f'Expected {expected} distinct nonempty labels')
        return self

class LocalDecisionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    state:ExpandedPatient
    questions:dict[str,AtomicQuestion]=Field(min_length=1,max_length=32)
    policy:Literal['sensitivity90','sensitivity95']='sensitivity90'

def concentration(probabilities):
    """Explicit local statistic: 1 minus normalized Shannon entropy.

    This is not the unpublished TypeSafe confidence formula or a correctness
    probability; Noul deliberately has no confidence field.
    """
    p=np.asarray(probabilities,float)
    if p.ndim!=1 or len(p)<2 or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-6):
        raise ValueError('Expected a probability distribution')
    p=p/p.sum();positive=p[p>0]
    return float(np.clip(1+np.sum(positive*np.log(positive))/np.log(len(p)),0,1))

def atomic_answers(joint,questions):
    p=np.asarray(joint,float)
    if p.shape!=(16,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-6):raise ValueError('Invalid joint probability distribution')
    p=p/p.sum();risk=float(p[1:].sum());marginals=p@BITS
    binary={'any_abnormality':risk,**dict(zip(COMPONENTS,marginals.tolist()))}
    counts=np.bincount(BITS.sum(axis=1).astype(int),weights=p,minlength=5)
    answers={}
    if not 1<=len(questions)<=32:raise ValueError('Expected 1 to 32 atomic questions')
    for identifier,spec in questions.items():
        q=spec if isinstance(spec,AtomicQuestion) else AtomicQuestion.model_validate(spec)
        if q.type=='noul':answers[identifier]={'type':'noul','noul':binary[q.target]}
        elif q.type=='score':
            levels=np.linspace(q.minimum,q.maximum,5)
            answers[identifier]={'type':'score','score':float(counts@levels),
                'legend':{str(float(level)):f'{i} of 4 abnormal findings' for i,level in enumerate(levels)},
                'probabilities':{str(float(level)):float(v) for level,v in zip(levels,counts)},
                'confidence':concentration(counts),'confidence_method':'one_minus_normalized_entropy'}
        else:
            probs=p if q.target=='pattern' else np.array([1-binary[q.target],binary[q.target]])
            labels=q.labels or ([f'pattern_{i:04b}' for i in range(16)] if q.target=='pattern' else ['false','true'])
            answers[identifier]={'type':'choice','choice':labels[int(np.argmax(probs))],
                'probabilities':dict(zip(labels,probs.tolist())),'confidence':concentration(probs),
                'confidence_method':'one_minus_normalized_entropy'}
    return answers

class LocalDecisionEngine:
    def __init__(self,model_path,guardrail_path):
        self.model=ExpandedPredictor(model_path)
        self.guard=json.loads(Path(guardrail_path).read_text(encoding='utf-8'))
    def infer_joint_once(self,person):
        state=person.model_dump();base={k:v for k,v in state.items() if k not in EXTRA}
        vector=np.r_[patient_vector(Patient.model_validate(base)),np.array([state[k] for k in EXTRA],float)]
        z=self.model.logits(self.model.pre.transform(vector))
        risk=expit(z[0]/self.model.config['temperature'])
        conditional=softmax(z[1:]@BITS[1:].T/self.model.config['pattern_temperature'])
        return np.r_[1-risk,risk*conditional]
    def decide(self,request):
        req=request if isinstance(request,LocalDecisionRequest) else LocalDecisionRequest.model_validate(request)
        # Question IDs, ordering, number and views never enter the learned model.
        joint=self.infer_joint_once(req.state);answers=atomic_answers(joint,req.questions)
        risk=float(joint[1:].sum());cut=self.model.config['thresholds'][req.policy]
        state=req.state.model_dump();missing=[k for k,v in state.items() if v is None]
        outside=[k for k,(lo,hi) in self.guard['training_ranges'].items() if state.get(k) is not None and not lo<=state[k]<=hi]
        confidence=concentration([1-risk,risk]);reasons=[]
        if missing:reasons.append('missing_inputs')
        if outside:reasons.append('outside_observed_training_range')
        if confidence<self.guard['confidence_floor']:reasons.append('low_distribution_concentration')
        if abs(risk-cut)<=self.guard['threshold_margin']:reasons.append('near_screening_cutoff')
        group='age19' if req.state.age==19 else f'age{"20_29" if req.state.age<30 else "30_39"}_living{req.state.living_alone}' if req.state.living_alone is not None else None
        support=self.guard.get('development_group_support',{}).get(group,{}).get(req.policy)
        if support is not None:
            if support['positive_n']<self.guard.get('minimum_group_positive_n',20):reasons.append('limited_development_group_support')
            if support['observed_sensitivity'] is None or support['observed_sensitivity']+1e-10<support['target_sensitivity']:
                reasons.append('development_group_target_not_met')
        return {'answers':answers,'policy_result':{'policy':req.policy,'screen_positive':bool(risk>=cut),
            'threshold':cut,'review_required':bool(reasons),'review_reasons':reasons,
            'development_group':group,'development_group_support':support,
            'review_is_not_a_diagnosis':True,'negative_is_not_medical_clearance':True},
            'input_quality':{'missing':missing,'outside_observed_training_range':outside},
            'model':self.model.config['name'],'schema_version':'local-atomic-v1','forward_passes':1,
            'research_only':True,'external_model_calls':0,
            'confidence_note':'Choice/Score confidence is an entropy statistic, not calibrated correctness or epistemic uncertainty. Noul has no separate confidence.'}
