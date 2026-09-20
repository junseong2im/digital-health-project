from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

NUMERIC = ['age', 'HE_BMI', 'HE_wc', 'WHtR']
CATEGORIES = {'sex': [1, 2], 'incm': [1, 2, 3, 4], 'edu': [1, 2, 3, 4],
              'sm_presnt': [0, 1], 'dr_month': [0, 1], 'pa_aerobic': [0, 1]}
FEATURES = NUMERIC + list(CATEGORIES)
COMPONENTS = ['elevated_glucose', 'elevated_bp', 'elevated_tg', 'low_hdl']
BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32)
TARGET_INPUTS = ['HE_glu', 'HE_sbp', 'HE_dbp', 'HE_TG', 'HE_HDL_st2', 'sex']

def features(frame):
    x = frame.copy()
    x['WHtR'] = x['HE_wc'] / x['HE_ht'].where(x['HE_ht'] > 0)
    for c, values in CATEGORIES.items():
        x[c] = x[c].where(x[c].isin(values))
    return x[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)

def targets(frame):
    if frame[TARGET_INPUTS].isna().any().any():
        raise ValueError('Incomplete target measurements cannot be coded as normal')
    if not frame.sex.isin([1, 2]).all():
        raise ValueError('Unknown sex cannot define HDL threshold')
    flags = np.column_stack([
        frame.HE_glu >= 100,
        (frame.HE_sbp >= 130) | (frame.HE_dbp >= 85),
        frame.HE_TG >= 150,
        frame.HE_HDL_st2 < np.where(frame.sex == 1, 40, 50),
    ]).astype(int)
    return flags @ (2 ** np.arange(4))

def cohort(frame, mode='unaware', fasting_hours=12):
    if mode not in ('unaware', 'untreated'):
        raise ValueError('Unknown cohort mode')
    d = frame.copy()
    flow = {'all': len(d)}
    def keep(name, mask):
        nonlocal d
        d = d.loc[mask].copy()
        flow[name] = len(d)
    keep('age_19_39', d.age.between(19, 39))
    diag = ['DI1_dg', 'DI2_dg', 'DE1_dg']
    keep('known_diagnosis_status', d[diag].isin([0, 1]).all(axis=1))
    if mode == 'unaware':
        keep('no_prior_diagnosis', d[diag].eq(0).all(axis=1))
    # Code 8 means skipped/not applicable, permitted for these adult medication items.
    keep('no_medication', d.DI1_2.isin([5, 8]) & d.DI2_2.isin([5, 8]) &
         d.DE1_31.isin([0, 8]) & d.DE1_32.isin([0, 8]))
    keep('not_recorded_pregnant', d.HE_dprg.isna())
    keep('fasting', d.HE_fst.ge(fasting_hours))
    keep('complete_target', d[TARGET_INPUTS].notna().all(axis=1) & d.sex.isin([1, 2]))
    keep('valid_survey_design', d.wt_itvex.gt(0) & d.psu.notna() & d.kstrata.notna())
    d['joint_target'] = targets(d)
    d['target'] = (d.joint_target > 0).astype(int)
    return d, flow

def load_cohort(root: Path, mode='unaware', fasting_hours=12):
    frames, flow = [], {}
    for year in range(2021, 2025):
        path = root / f'data/raw/knhanes/HN{str(year)[2:]}_ALL(SPSS)/HN{str(year)[2:]}_ALL.sav'
        d, _ = pyreadstat.read_sav(str(path), apply_value_formats=False)
        d['survey_year'] = year
        d['group'] = str(year) + ':' + d.psu.astype(str)
        d, flow[str(year)] = cohort(d, mode, fasting_hours)
        frames.append(d)
    return pd.concat(frames, ignore_index=True), flow
