"""Paired multi-output uncertainty and development-only extra-feature importance."""
import json
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from metabolic.data import load_cohort,BITS
from metabolic.expanded import extended_features,EXTRA,FEATURES
from metabolic.comparison import ROOT,dump,bootstrap_indexes
from metabolic.train import weights
from metabolic.train_factorial import fit_nn

OUT=ROOT/'artifacts/expanded_multitask_v4'
r=json.loads((OUT/'report.json').read_text());pred=json.loads((OUT/'test_predictions.json').read_text())
d,_=load_cohort(ROOT);dev=d[d.survey_year<=2023];test=d[d.survey_year==2024];y=BITS[test.joint_target.to_numpy()];w=weights(test)
comparisons={'expanded_multi_minus_LR5':('expanded_multi_s42','expanded_LR5'),
             'expanded_minus_base_multi':('expanded_multi_s42','base_multi_s42')}
ci={}
for label,(aa,bb) in comparisons.items():
    a=np.array(pred[aa]['components']);b=np.array(pred[bb]['components']);values={k:[] for k in ['macro_auc','macro_ap','macro_brier','count_mae']}
    for ix in bootstrap_indexes(test,300):
        if any(len(np.unique(y[ix,j]))<2 for j in range(4)):continue
        for key,fn in [('macro_auc',roc_auc_score),('macro_ap',average_precision_score),('macro_brier',brier_score_loss)]:
            values[key].append(float(np.mean([fn(y[ix,j],a[ix,j],sample_weight=w[ix])-fn(y[ix,j],b[ix,j],sample_weight=w[ix]) for j in range(4)])))
        values['count_mae'].append(float(np.average(abs(a[ix].sum(axis=1)-y[ix].sum(axis=1))-abs(b[ix].sum(axis=1)-y[ix].sum(axis=1)),weights=w[ix])))
    ci[label]={k:{'ci95':np.quantile(v,[.025,.975]).tolist(),'bootstrap_mean':float(np.mean(v)),'valid_repeats':len(v)} for k,v in values.items()}
dump(OUT/'multioutput_uncertainty.json',ci)

# Refit the frozen fold configurations to inspect extra information without using 2024.
manifest=json.loads((OUT/'split_manifest.json').read_text());cv=json.loads((OUT/'cv_progress.json').read_text())
epochs=cv['expanded_multi_s42']['best_epochs'];rng=np.random.default_rng(2026);torch.set_num_threads(2)
blocks={k:[10+EXTRA.index(k)] for k in EXTRA if k not in ['alcohol_frequency','alcohol_amount']}
blocks['alcohol_detail']=[14,15]
values={k:[] for k in blocks};baseline=[]
for fold,split in enumerate(manifest['folds']):
    train,val=dev.iloc[split['train']],dev.iloc[split['validation']]
    pre,net,_=fit_nn(train,True,True,42+fold,epochs[fold]);raw=extended_features(val).to_numpy();yv=val.target.to_numpy();wv=weights(val)
    def auc(x):
        with torch.inference_mode():p=expit(net(torch.tensor(pre.transform(x),dtype=torch.float32)).numpy()[:,0])
        return roc_auc_score(yv,p,sample_weight=wv)
    score=auc(raw);baseline.append(score)
    for name,cols in blocks.items():
        for _ in range(10):
            z=raw.copy()
            if name=='alcohol_detail':
                # Keep monthly-drinking category unchanged while shuffling its details.
                strata=np.where(np.isnan(raw[:,8]),-1,raw[:,8])
                for level in np.unique(strata):
                    ix=np.flatnonzero(strata==level);z[np.ix_(ix,cols)]=raw[np.ix_(rng.permutation(ix),cols)]
            else:z[:,cols]=raw[rng.permutation(len(raw))][:,cols]
            values[name].append(float(score-auc(z)))
    print('OOF importance fold',fold+1,'done',flush=True)
dump(OUT/'extra_feature_importance.json',{'fold_baseline_auc':baseline,
    'blocks':{k:{'mean_auc_drop':float(np.mean(v)),'sd_across_folds_and_permutations':float(np.std(v,ddof=1)),'repeats':len(v)} for k,v in values.items()},
    'method':'Frozen seed42 expanded-multitask fold models; 10 permutations each fold; alcohol details shuffled within monthly-drinking strata; no test-based feature selection',
    'caution':'Permutation importance is not a causal effect. Negative or tiny changes do not establish harmful/protective effects.'})
print('Diagnostics saved.')
