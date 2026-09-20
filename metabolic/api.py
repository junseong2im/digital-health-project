"""Optional local HTTP interface. No external services are called."""
import os
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .decisions import Patient, Question
from .predict import Predictor
from .fast import FastPredictor
from .refined import RefinedPredictor
from .specialized import SpecializedPredictor
from .expanded import ExpandedPatient, ExpandedPredictor
from .household import HouseholdPatient, HouseholdPredictor

app = FastAPI(title='Young-adult metabolic screening research model')

class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    state: Patient
    questions: dict[str, Question] | None = Field(default=None, min_length=1, max_length=32)

@lru_cache
def predictor():
    default = Path(__file__).resolve().parents[1] / 'artifacts/local_parallel_v1_run2'
    return FastPredictor(os.environ.get('METABOLIC_ARTIFACT', str(default)))

@app.post('/v1/decide')
def decide(request: DecisionRequest):
    try:
        return predictor().predict(request.state.model_dump(), request.questions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get('/health')
def health():
    return {'status':'loaded','model':predictor().config['run_name'],'research_only':True}

class HouseholdDecisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    state: HouseholdPatient
    questions: dict[str, Question] | None = Field(default=None, min_length=1, max_length=32)
    policy: str = 'sensitivity90'

@lru_cache
def household_predictor():
    default = Path(__file__).resolve().parents[1] / 'artifacts/comparison_household_v1/models/household_seed42'
    return HouseholdPredictor(os.environ.get('METABOLIC_HOUSEHOLD_ARTIFACT',str(default)))

@app.post('/v1/decide-household')
def decide_household(request: HouseholdDecisionRequest):
    try:
        return household_predictor().predict(request.state.model_dump(),request.questions,request.policy)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc

class RefinedDecisionRequest(DecisionRequest):
    policy: str = 'sensitivity90'

@lru_cache
def refined_predictor():
    default=Path(__file__).resolve().parents[1]/'artifacts/optimized_v2/model_complete'
    return RefinedPredictor(os.environ.get('METABOLIC_REFINED_ARTIFACT',str(default)))

@app.post('/v2/decide')
def decide_refined(request: RefinedDecisionRequest):
    try:
        return refined_predictor().predict(request.state.model_dump(),request.questions,request.policy)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc

@lru_cache
def specialized_predictor():
    default=Path(__file__).resolve().parents[1]/'artifacts/specialized_v3/model'
    return SpecializedPredictor(os.environ.get('METABOLIC_SPECIALIZED_ARTIFACT',str(default)))

@app.post('/v3/decide')
def decide_specialized(request: RefinedDecisionRequest):
    try:
        return specialized_predictor().predict(request.state.model_dump(),request.questions,request.policy)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc

class ExpandedDecisionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    state: ExpandedPatient
    policy: str = 'youden'

@lru_cache
def expanded_predictor():
    default=Path(__file__).resolve().parents[1]/'artifacts/expanded_multitask_v4/models/expanded_multi_s42'
    return ExpandedPredictor(os.environ.get('METABOLIC_EXPANDED_ARTIFACT',str(default)))

@app.post('/v4/decide-expanded')
def decide_expanded(request: ExpandedDecisionRequest):
    try:
        return expanded_predictor().predict(request.state.model_dump(),request.policy)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
