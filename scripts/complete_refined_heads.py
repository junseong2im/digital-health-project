"""Fit conditional-pattern output while keeping the selected primary risk fixed."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from metabolic.data import load_cohort,features,BITS,COMPONENTS
from metabolic.train import weights,metrics
from metabolic.comparison import restore_parts,OLD,dump
from metabolic.refined import RefinedPredictor,joint_from_logits

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/optimized_v2'
DEST=OUT/'model_complete';DEST.mkdir(exist_ok=False)
torch.set_num_threads(2);torch.manual_seed(2026)
d,_=load_cohort(ROOT);parts=restore_parts(d,json.loads((OLD/'split_manifest.json').read_text()))
dev=d[d.survey_year<=2022];model=RefinedPredictor(OUT/'model')
def embedding(frame):
    x=model.pre.transform(features(frame).to_numpy()).astype('float32')
    h=np.tanh(x@model.p['hidden.weight'].T+model.p['hidden.bias'])
    return np.concatenate([x,h],axis=1)

positive=dev.target==1;z=embedding(dev.loc[positive]);y=(dev.loc[positive].joint_target.to_numpy()-1).astype('int64')
w=weights(dev.loc[positive]).astype('float32')
head=torch.nn.Linear(z.shape[1],15)
with torch.no_grad():
    head.weight.zero_();prior=np.bincount(y,weights=w,minlength=15)+1
    head.bias.copy_(torch.tensor(np.log(prior/prior.sum()),dtype=torch.float32))
opt=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=0)
tz=torch.from_numpy(z);ty=torch.from_numpy(y);tw=torch.from_numpy(w);history=[]
for _ in range(250):
    opt.zero_grad();loss=(torch.nn.functional.cross_entropy(head(tz),ty,reduction='none')*tw).sum()/tw.sum()+.01*head.weight.square().sum()
    loss.backward();opt.step();history.append(float(loss.detach()))
parameters={k:v.copy() for k,v in model.p.items()}
parameters['output.weight'][1:]=head.weight.detach().numpy();parameters['output.bias'][1:]=head.bias.detach().numpy()
assert np.array_equal(parameters['output.weight'][0],model.p['output.weight'][0])
assert parameters['output.bias'][0]==model.p['output.bias'][0]
cal=parts['calibration'];pos=cal.target.to_numpy()==1;yc=cal.joint_target.to_numpy()[pos]-1;wc=weights(cal)[pos]
cal_logits=embedding(cal)@parameters['output.weight'].T+parameters['output.bias']
opt_temp=minimize_scalar(lambda lt:float(np.average(-np.log(np.clip(softmax(cal_logits[pos,1:]/np.exp(lt),axis=1)[np.arange(len(yc)),yc],1e-12,1)),weights=wc)),bounds=(-2.3,2.3),method='bounded')
assert opt_temp.success
config=model.config.copy();config['pattern_temperature']=float(np.exp(opt_temp.x))
config['pattern_training']='Frozen primary branch; conditional multinomial weighted CE + 0.01 L2, fixed 250 epochs, 2021-2022 positive cases only'
np.savez(DEST/'network.npz',**parameters);model.pre.save(DEST/'preprocessor.npz');dump(DEST/'config.json',config)
test=parts['test_2024'];z=embedding(test)@parameters['output.weight'].T+parameters['output.bias']
p=joint_from_logits(z,config['risk_temperature'],config['pattern_temperature'])
primary_before=np.array([model.probability(v) for v in features(test).to_numpy()])
assert np.max(np.abs(primary_before-p[:,1:].sum(axis=1)))<1e-6
component=p@BITS;truth=BITS[test.joint_target.to_numpy()]
report={'fixed_primary_max_difference':float(np.max(np.abs(primary_before-p[:,1:].sum(axis=1)))),
    'training_loss_first':history[0],'training_loss_last':history[-1],
    'count_weighted_mae':float(np.average(abs(p@BITS.sum(axis=1)-truth.sum(axis=1)),weights=weights(test))),
    'components':{name:metrics(truth[:,j],component[:,j],.5,weights(test)) for j,name in enumerate(COMPONENTS)},
    'note':'Conditional head completed separately so binary early stopping does not leave auxiliary outputs effectively untrained. No change to selected binary classifier, calibration, or cutoff.'}
dump(OUT/'pattern_report.json',report);print('Completed conditional heads; primary risk unchanged.',report['fixed_primary_max_difference'])
