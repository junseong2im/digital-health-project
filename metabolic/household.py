"""Household-size proxy, explicitly distinct from independent living (자취)."""
import argparse
import json
from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
import torch
from .data import features
from .decisions import Patient, render_decisions
from .network import ParallelDecisionNet

def living_alone(cfam):
    s = pd.to_numeric(cfam, errors='coerce')
    result = pd.Series(np.nan, index=s.index, dtype=float)
    result.loc[s.isin([1,2,3,4,5,6])] = s.loc[s.isin([1,2,3,4,5,6])].eq(1).astype(float)
    return result

def household_features(frame, include_household=False):
    x=features(frame)
    if include_household:
        x['living_alone']=living_alone(frame['cfam'])
    return x

class HouseholdPatient(Patient):
    cfam: Literal[1,2,3,4,5,6] | None = None

class HouseholdPredictor:
    """Load a comparison NN; never replaces the original clinical research model."""
    def __init__(self, artifact):
        p=Path(artifact)
        self.config=json.loads((p/'config.json').read_text(encoding='utf-8'))
        self.pre=joblib.load(p/'preprocessor.joblib')
        ck=torch.load(p/'network.pt',map_location='cpu',weights_only=True)
        self.model=ParallelDecisionNet(ck['input_dim'])
        self.model.load_state_dict(ck['state_dict']);self.model.eval()

    def predict(self,state,questions=None,policy='sensitivity90'):
        patient=HouseholdPatient.model_validate(state)
        if policy not in self.config['thresholds']:
            raise ValueError('Use sensitivity90 or youden policy')
        frame=pd.DataFrame([patient.model_dump()]).astype(float)
        living=living_alone(frame.cfam).iloc[0]
        if self.config['include_household'] and pd.isna(living):
            raise ValueError('Household size is required for this augmented model; use the original model if unknown')
        subgroup=self.config.get('subgroup')
        if subgroup in ['living_0','living_1'] and (pd.isna(living) or int(living)!=int(subgroup[-1])):
            raise ValueError('Patient outside the subgroup used to train this model')
        if subgroup=='age_19_29' and patient.age>=30 or subgroup=='age_30_39' and patient.age<30:
            raise ValueError('Patient outside the age subgroup used to train this model')
        x=self.pre.transform(household_features(frame,self.config['include_household'])).astype('float32')
        with torch.inference_mode():
            p=torch.softmax(self.model(torch.from_numpy(x))/self.config['temperature'],dim=1)[0].numpy()
        result=render_decisions(p,self.config['thresholds'][policy],questions)
        result.update({'household_group':'unknown' if pd.isna(living) else ('single_person' if living else 'multi_person'),
            'household_definition':'cfam == 1; household-size proxy, not verified independent living or student living arrangement',
            'model':self.config['name'],'threshold_policy':policy,
            'missing_inputs':[k for k,v in patient.model_dump().items() if v is None],
            'exploratory_extension':True})
        return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--artifact',required=True);a.add_argument('--input',required=True)
    a.add_argument('--policy',choices=['sensitivity90','youden'],default='sensitivity90');args=a.parse_args()
    request=json.loads(Path(args.input).read_text(encoding='utf-8-sig'))
    print(json.dumps(HouseholdPredictor(args.artifact).predict(request.get('state',request),request.get('questions'),args.policy),ensure_ascii=False,indent=2,allow_nan=False))
