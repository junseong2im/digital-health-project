import math
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .data import BITS, COMPONENTS

class Patient(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    age: int = Field(ge=19, le=39, strict=True)
    sex: Literal[1, 2]
    incm: Literal[1, 2, 3, 4] | None = None
    edu: Literal[1, 2, 3, 4] | None = None
    HE_BMI: float | None = Field(default=None, ge=10, le=80)
    HE_wc: float | None = Field(default=None, ge=30, le=200)
    HE_ht: float | None = Field(default=None, ge=100, le=230)
    sm_presnt: Literal[0, 1] | None = None
    dr_month: Literal[0, 1] | None = None
    pa_aerobic: Literal[0, 1] | None = None

    @model_validator(mode='after')
    def has_measurements(self):
        if self.HE_BMI is None and (self.HE_wc is None or self.HE_ht is None):
            raise ValueError('Provide BMI or both waist circumference and height')
        return self

class Question(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    type: Literal['choice', 'score', 'noul']
    # This trained model supports these study outputs, not arbitrary text questions.
    concept: Literal['any_metabolic_abnormality', 'abnormality_count']
    labels: list[str] = Field(default_factory=lambda: ['screen_negative', 'screen_positive'])
    minimum: float = 0
    maximum: float = 4

    @model_validator(mode='after')
    def valid_schema(self):
        if self.type == 'score':
            if self.concept != 'abnormality_count' or not self.maximum > self.minimum:
                raise ValueError('Score requires abnormality_count and an increasing range')
        elif self.concept != 'any_metabolic_abnormality':
            raise ValueError('Choice/Noul require any_metabolic_abnormality')
        if self.type == 'choice' and (len(self.labels) != 2 or len(set(self.labels)) != 2 or any(not x for x in self.labels)):
            raise ValueError('This binary screening model requires exactly two distinct labels')
        return self

DEFAULT_QUESTIONS = {
    'screening': Question(type='choice', concept='any_metabolic_abnormality'),
    'risk': Question(type='noul', concept='any_metabolic_abnormality'),
    'burden': Question(type='score', concept='abnormality_count'),
}

def render_decisions(probabilities, threshold, questions=None):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (16,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-6):
        raise ValueError('Expected a finite normalized 16-class distribution')
    p = p / p.sum()  # Remove float32 rounding drift in the serving contract.
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid decision threshold')
    questions = DEFAULT_QUESTIONS if questions is None else questions
    if not questions or len(questions) > 32:
        raise ValueError('Expected 1 to 32 supported questions')
    risk = float(p[1:].sum())
    positive = risk >= threshold
    counts = BITS.sum(axis=1).astype(int)
    count_p = np.bincount(counts, weights=p, minlength=5)
    answers = {}
    for key, q in questions.items():
        q = q if isinstance(q, Question) else Question.model_validate(q)
        if q.type == 'choice':
            answers[key] = {'type': 'choice', 'choice': q.labels[int(positive)],
                'probabilities': {q.labels[0]: float(p[0]), q.labels[1]: risk},
                'confidence': risk if positive else float(p[0])}
        elif q.type == 'noul':
            answers[key] = {'type': 'noul', 'value': bool(positive), 'noul': risk,
                'confidence': risk if positive else float(p[0]),
                'probabilities': {'false': float(p[0]), 'true': risk}}
        else:
            levels = np.linspace(q.minimum, q.maximum, 5)
            answers[key] = {'type': 'score', 'score': float(np.dot(levels, count_p)),
                'confidence': float(count_p.max()),
                'confidence_definition': 'Probability of the modal count category, not confidence that the expected score is exact.',
                'minimum': q.minimum, 'maximum': q.maximum,
                'probabilities': {str(float(level)): float(v) for level, v in zip(levels, count_p)}}
    return {'answers': answers, 'component_probabilities': dict(zip(COMPONENTS, (p @ BITS).tolist())),
        'threshold': threshold, 'research_only': True,
        'confidence_definition': 'Probability assigned to the chosen category; not epistemic certainty or a guarantee of correctness.'}
