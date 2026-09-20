import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .data import features
from .decisions import Patient, render_decisions
from .network import ParallelDecisionNet

class Predictor:
    def __init__(self, artifact):
        path = Path(artifact)
        self.config = json.loads((path/'config.json').read_text(encoding='utf-8'))
        self.preprocessor = joblib.load(path/'preprocessor.joblib')
        checkpoint = torch.load(path/'network.pt',map_location='cpu',weights_only=True)
        self.model = ParallelDecisionNet(checkpoint['input_dim'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def predict(self, state, questions=None):
        patient = Patient.model_validate(state)
        values = patient.model_dump()
        frame = pd.DataFrame([values]).astype(float)
        x = self.preprocessor.transform(features(frame)).astype(np.float32)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(torch.from_numpy(x))/self.config['temperature'],dim=1)[0].numpy()
        response = render_decisions(probabilities,self.config['threshold'],questions)
        response['missing_inputs'] = [k for k,v in values.items() if v is None]
        response['model'] = self.config['run_name']
        return response

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--artifact',required=True)
    ap.add_argument('--input',required=True)
    args=ap.parse_args()
    payload=json.loads(Path(args.input).read_text(encoding='utf-8-sig'))
    state=payload.get('state',payload)
    from .fast import FastPredictor
    print(json.dumps(FastPredictor(args.artifact).predict(state,payload.get('questions')),ensure_ascii=False,indent=2,allow_nan=False))
