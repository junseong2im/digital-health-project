"""Post-fit subgroup uncertainty and validation-only permutation importance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from metabolic.data import load_cohort,features
from metabolic.household import living_alone
from metabolic.comparison import restore_parts,bootstrap_indexes,subgroup_masks,dump,OLD
from metabolic.train import weights
from metabolic.network import ParallelDecisionNet

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/comparison_household_v1'
d,_=load_cohort(ROOT);d=d.copy();d['living_alone']=living_alone(d.cfam)
parts=restore_parts(d,json.loads((OUT/'split_manifest.json').read_text()))
report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
pred=pd.read_json(OUT/'test_predictions.json')
test=parts['test_2024'];assert np.array_equal(test.index,pred.row_in_cohort)
y=test.target.to_numpy();w=weights(test);groups=subgroup_masks(test)
cis={name:{g:{'auc':[],'sensitivity':[],'specificity':[]} for g in groups if groups[g].any()} for name in ['base_seed42','household_seed42']}
for ix in bootstrap_indexes(test):
    for name,sub in cis.items():
        p=pred[name].to_numpy();cut=report['models'][name]['thresholds']['youden']
        for group,buckets in sub.items():
            rows=ix[groups[group][ix]]
            if not len(rows) or len(np.unique(y[rows]))<2:continue
            pos=rows[y[rows]==1];neg=rows[y[rows]==0]
            buckets['auc'].append(roc_auc_score(y[rows],p[rows],sample_weight=w[rows]))
            buckets['sensitivity'].append(np.average(p[pos]>=cut,weights=w[pos]))
            buckets['specificity'].append(np.average(p[neg]<cut,weights=w[neg]))
ciout={name:{group:{k:{'ci95':np.quantile(v,[.025,.975]).tolist(),'valid_repeats':len(v)} for k,v in buckets.items() if v} for group,buckets in sub.items()} for name,sub in cis.items()}
dump(OUT/'subgroup_uncertainty.json',ciout)

# Permute only validation data: not used for selecting the already trained model.
torch.set_num_threads(2)
pre=joblib.load(OLD/'preprocessor.joblib');ck=torch.load(OLD/'network.pt',map_location='cpu',weights_only=True)
net=ParallelDecisionNet(ck['input_dim']);net.load_state_dict(ck['state_dict']);net.eval()
temp=json.loads((OLD/'config.json').read_text())['temperature']
v=parts['validation'];x=features(v);yv=v.target.to_numpy();wv=weights(v)
def score(f):
    with torch.inference_mode():p=1-torch.softmax(net(torch.tensor(pre.transform(f),dtype=torch.float32))/temp,dim=1)[:,0].numpy()
    return roc_auc_score(yv,p,sample_weight=wv)
original=score(x);rng=np.random.default_rng(2026)
blocks={c:[c] for c in x.columns};blocks['anthropometry_joint']=['HE_BMI','HE_wc','WHtR']
importance={}
for name,cols in blocks.items():
    drops=[]
    for _ in range(20):
        shuffled=x.copy();shuffled.loc[:,cols]=x.iloc[rng.permutation(len(x))][cols].to_numpy()
        drops.append(original-score(shuffled))
    importance[name]={'mean_auc_drop':float(np.mean(drops)),'sd':float(np.std(drops,ddof=1)),'columns':cols}
dump(OUT/'validation_importance.json',{'n':len(v),'baseline_auc':original,'importance':importance,
    'note':'Validation-set permutation only; correlated anthropometry may mask individual contributions; not causal effects.'})
terms=report['associations']['models']['interactions']['terms']
names=['age30:sm_presnt','living_alone:sm_presnt'];q=multipletests([terms[k]['p'] for k in names],method='fdr_bh')[1]
dump(OUT/'interaction_multiplicity.json',{k:terms[k]|{'q_bh_two_prespecified_interactions':float(z)} for k,z in zip(names,q)})
print('Subgroup CIs, validation importance, and interaction multiplicity saved.')
