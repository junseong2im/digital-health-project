"""Temporal, group-separated experiment. Run: python -m metabolic.train."""
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logit, expit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix, matthews_corrcoef)
from .data import load_cohort, features, NUMERIC, CATEGORIES, BITS
from .network import ParallelDecisionNet

ROOT = Path(__file__).resolve().parents[1]

def weights(d):
    # All rows in each temporal set cover equally long survey years. A common
    # 1/number_of_years integration factor cancels in normalized losses/metrics.
    w = d.wt_itvex.to_numpy(dtype=float)
    return w / w.mean()

def preprocessing():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                             ('scale', StandardScaler())]), NUMERIC),
        ('category', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=-1)),
            ('encode', OneHotEncoder(categories=[[-1] + v for v in CATEGORIES.values()],
                                     handle_unknown='error', sparse_output=False))]), list(CATEGORIES))])

def metrics(y, p, threshold, w=None):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-8, 1-1e-8)
    w = np.ones(len(y)) if w is None else np.asarray(w)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1], sample_weight=w).ravel()
    ratio = lambda a, b: float(a/b) if b else None
    bins = np.minimum((p * 10).astype(int), 9)
    reliability = []
    ece = 0.
    for b in range(10):
        m = bins == b
        if m.any():
            observed, predicted = np.average(y[m], weights=w[m]), np.average(p[m], weights=w[m])
            ece += w[m].sum()/w.sum()*abs(observed-predicted)
            reliability.append({'bin': b, 'n': int(m.sum()), 'predicted': float(predicted), 'observed': float(observed)})
    return {'n': len(y), 'prevalence': float(np.average(y, weights=w)),
        'roc_auc': float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y)) > 1 else None,
        'pr_auc_ap': float(average_precision_score(y,p,sample_weight=w)) if np.any(y) else None,
        'brier': float(brier_score_loss(y,p,sample_weight=w)),
        'log_loss': float(log_loss(y,p,sample_weight=w,labels=[0,1])),
        'ece_10_bins': float(ece), 'sensitivity': ratio(tp,tp+fn), 'specificity': ratio(tn,tn+fp),
        'ppv': ratio(tp,tp+fp), 'npv': ratio(tn,tn+fn),
        'mcc': float(matthews_corrcoef(y,pred,sample_weight=w)), 'threshold': float(threshold),
        'reliability': reliability}

def choose_threshold(y, p, w, minimum_sensitivity=0.90):
    if not np.any(y == 1) or not np.any(y == 0):
        raise ValueError('Threshold set must include both outcomes')
    # Highest threshold meeting the prespecified sensitivity on the threshold set.
    candidates = np.unique(np.r_[0., p])
    valid = [t for t in candidates if np.average((p[y==1] >= t), weights=w[y==1]) >= minimum_sensitivity]
    return float(max(valid))

def bootstrap_ci(d, p, threshold, repeats=200, seed=42):
    # Resample PSUs inside strata. Descriptive bootstrap CI, not an official
    # KNHANES domain-variance estimator (excluded domain rows are not retained).
    rng = np.random.default_rng(seed)
    strata = []
    for _, group in d.groupby('kstrata'):
        strata.append([np.flatnonzero(d.group.to_numpy() == g) for g in group.group.unique()])
    values = {k: [] for k in ['roc_auc','pr_auc_ap','brier','sensitivity','specificity']}
    y, w = d.target.to_numpy(), weights(d)
    for _ in range(repeats):
        ix = np.concatenate([np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))]) for clusters in strata])
        if len(np.unique(y[ix])) < 2:
            continue
        r = metrics(y[ix], p[ix], threshold, w[ix])
        for k in values:
            values[k].append(r[k])
    return {k: np.quantile(v,[.025,.975]).tolist() for k,v in values.items() if v}

def run(args):
    out = ROOT / 'artifacts' / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    d, flow = load_cohort(ROOT, args.cohort, args.fasting_hours)
    development = d[d.survey_year <= 2022].copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=args.seed)
    train_i, val_i = next(splitter.split(development, groups=development.group))
    train, val = development.iloc[train_i], development.iloc[val_i]
    year23 = d[d.survey_year == 2023].copy()
    ci, ti = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=args.seed).split(year23, groups=year23.group))
    parts = {'train': train, 'validation': val, 'calibration': year23.iloc[ci],
             'threshold': year23.iloc[ti], 'test_2024': d[d.survey_year == 2024].copy()}
    for a, da in parts.items():
        assert da.target.nunique() == 2, f'{a} lacks one outcome'
        for b, db in parts.items():
            if a != b:
                assert set(da.group).isdisjoint(db.group), f'Group leakage {a}/{b}'
    # Only training rows fit imputation, scaling, and the model.
    pre = preprocessing()
    pre.fit(features(train))
    x = {k: pre.transform(features(v)).astype(np.float32) for k,v in parts.items()}
    y = {k: v.joint_target.to_numpy().astype('int64') for k,v in parts.items()}
    w = {k: weights(v).astype('float32') for k,v in parts.items()}
    model = ParallelDecisionNet(x['train'].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    tensors = {k: (torch.from_numpy(x[k]),torch.from_numpy(y[k]),torch.from_numpy(w[k])) for k in ['train','validation']}
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    best_loss, stale, best_epoch, history = float('inf'), 0, 0, []
    best = None
    tx, ty, tw = tensors['train']
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tx))
        for ix in order.split(128):
            opt.zero_grad()
            loss = (loss_fn(model(tx[ix]), ty[ix]) * tw[ix]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx, vy, vw = tensors['validation']
            vloss = float((loss_fn(model(vx),vy)*vw).sum()/vw.sum())
        history.append({'epoch':epoch+1,'validation_joint_nll':vloss})
        if vloss < best_loss - 1e-5:
            best_loss, stale, best_epoch = vloss, 0, epoch+1
            best = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 25:
            break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        logits = {k:model(torch.from_numpy(v)).numpy() for k,v in x.items()}
    def calibration_loss(log_temp):
        pp = softmax(logits['calibration']/np.exp(log_temp),axis=1)
        return float(np.average(-np.log(np.clip(pp[np.arange(len(pp)), y['calibration']],1e-12,1)),weights=w['calibration']))
    result = minimize_scalar(calibration_loss, bounds=(-2.3,2.3), method='bounded')
    if not result.success:
        raise RuntimeError('Temperature fitting failed')
    temp = float(np.exp(result.x))
    joint = {k:softmax(v/temp,axis=1) for k,v in logits.items()}
    probs = {k:1-v[:,0] for k,v in joint.items()}
    truth = {k:(v>0).astype(int) for k,v in y.items()}
    threshold = choose_threshold(truth['threshold'],probs['threshold'],w['threshold'],args.sensitivity)
    # Freeze checkpoint, calibration and cutoff before computing test metrics.
    torch.save({'state_dict':model.state_dict(),'input_dim':x['train'].shape[1]},out/'network.pt')
    joblib.dump(pre,out/'preprocessor.joblib')
    config = vars(args) | {'temperature':temp,'threshold':threshold,'best_epoch':best_epoch,
        'architecture':'MLP joint categorical distribution over 16 metabolic patterns',
        'training':'Survey-weighted supervised categorical log loss + held-out temperature scaling; not RLCD',
        'features':NUMERIC+list(CATEGORIES), 'seed':args.seed}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    reports = {'parallel_nn':{k:metrics(truth[k],probs[k],threshold,w[k]) for k in ['validation','calibration','threshold','test_2024']}}
    reports['parallel_nn']['test_unweighted'] = metrics(truth['test_2024'],probs['test_2024'],threshold)
    reports['parallel_nn']['test_uncalibrated'] = metrics(truth['test_2024'],1-softmax(logits['test_2024'],axis=1)[:,0],threshold,w['test_2024'])
    reports['parallel_nn']['test_ci95_exploratory'] = bootstrap_ci(parts['test_2024'],probs['test_2024'],threshold,seed=args.seed)
    reports['parallel_nn']['component_test'] = {}
    components = joint['test_2024'] @ BITS
    from .data import COMPONENTS
    for j,name in enumerate(COMPONENTS):
        reports['parallel_nn']['component_test'][name] = metrics(BITS[y['test_2024'],j],components[:,j],.5,w['test_2024'])
    reports['parallel_nn']['count_test_mae'] = float(np.average(abs(joint['test_2024'] @ BITS.sum(axis=1)-BITS[y['test_2024']].sum(axis=1)),weights=w['test_2024']))
    baseline_predictions = {}
    for name, estimator in {
        'logistic_regression':LogisticRegression(C=1.,max_iter=2000,random_state=args.seed),
        'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,l2_regularization=5,early_stopping=False,random_state=args.seed)
    }.items():
        estimator.fit(x['train'],truth['train'],sample_weight=w['train'])
        bp = {k:estimator.predict_proba(v)[:,1] for k,v in x.items()}
        # Baselines receive the same held-out calibration and threshold rows.
        rr = minimize_scalar(lambda lt: log_loss(truth['calibration'], expit(logit(np.clip(bp['calibration'],1e-7,1-1e-7))/np.exp(lt)),sample_weight=w['calibration'],labels=[0,1]),bounds=(-2.3,2.3),method='bounded')
        if not rr.success:
            raise RuntimeError('Baseline calibration failed')
        bt = float(np.exp(rr.x))
        bp = {k:expit(logit(np.clip(v,1e-7,1-1e-7))/bt) for k,v in bp.items()}
        cut = choose_threshold(truth['threshold'],bp['threshold'],w['threshold'],args.sensitivity)
        reports[name] = {'temperature':bt,'threshold':cut,'test_2024':metrics(truth['test_2024'],bp['test_2024'],cut,w['test_2024'])}
        joblib.dump({'model':estimator,'temperature':bt,'threshold':cut},out/f'{name}.joblib')
        baseline_predictions[name] = bp['test_2024']
    report = {'cohort_flow':flow,'partitions':{k:{'n':len(v),'positive':int(v.target.sum()),'psus':v.group.nunique(),'years':sorted(v.survey_year.unique().tolist())} for k,v in parts.items()},
        'models':reports,'training_history':history,
        'limitations':['Cross-sectional screening, not prospective disease prediction.',
            'No diagnosis is a proxy for unawareness of these lower-threshold metabolic findings.',
            'Complete laboratory cases with >= fasting_hours; selection bias is possible.',
            'KNHANES measured anthropometry is not validation of self-measured inputs.',
            'Probability calibration and 90% sensitivity are not guaranteed on held-out data.',
            'Bootstrap intervals resample observed eligible PSUs within strata; not official design-based domain intervals.',
            'Single fixed architecture/seed; exploratory experiment, not clinical deployment.']}
    # Aggregate-only public report; individual predictions stay in the local artifact.
    pd.DataFrame({'row_in_cohort':parts['test_2024'].index,'target':truth['test_2024'],
        'probability':probs['test_2024'],'weight':w['test_2024'],**baseline_predictions}).to_json(out/'test_predictions.json',orient='records',indent=2)
    split_ids = {k:[hashlib.sha256(f'{r.survey_year}:{r.ID}'.encode()).hexdigest() for r in v.itertuples()] for k,v in parts.items()}
    (out/'split_manifest.json').write_text(json.dumps(split_ids,indent=2),encoding='utf-8')
    # Benchmark the complete local prediction path separately from model forward.
    from .predict import Predictor
    predictor = Predictor(out)
    sample = {'age':29,'sex':1,'HE_BMI':25.,'HE_wc':87.,'HE_ht':175.,'incm':2,'edu':4,'sm_presnt':0,'dr_month':1,'pa_aerobic':1}
    predictor.predict(sample)
    timings = []
    for _ in range(100):
        start = time.perf_counter()
        example = predictor.predict(sample)
        timings.append((time.perf_counter()-start)*1000)
    report['latency_ms'] = {'scope':'Warm CPU preprocessing + single forward + typed output; excludes startup, transport',
        'p50':float(np.median(timings)),'p95':float(np.quantile(timings,.95)),'repeats':100}
    report['environment'] = {'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':2}
    report['source_hashes'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'metabolic').glob('*.py')}
    report['input_hashes'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/raw/knhanes').glob('*.zip')}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'example_request.json').write_text(json.dumps(sample,indent=2),encoding='utf-8')
    (out/'example_response.json').write_text(json.dumps(example,indent=2),encoding='utf-8')
    print(json.dumps({'artifact':str(out),'partitions':report['partitions'],
        'metrics':{k:v['test_2024'] for k,v in reports.items()},'latency':report['latency_ms']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-name',default='local_parallel_v1')
    ap.add_argument('--cohort',choices=['unaware','untreated'],default='unaware')
    ap.add_argument('--fasting-hours',type=int,choices=[8,12],default=12)
    ap.add_argument('--epochs',type=int,default=250)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--sensitivity',type=float,default=.90)
    args=ap.parse_args()
    if not 0 < args.sensitivity <= 1 or args.epochs < 1 or Path(args.run_name).name != args.run_name:
        ap.error('Invalid sensitivity, epochs or run name')
    run(args)
