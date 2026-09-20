"""Expanded-development study: domain architecture and cross-fitted calibration."""
import argparse
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from scipy.special import expit,softmax
from scipy.optimize import minimize_scalar
from .data import load_cohort,features,BITS,COMPONENTS
from .refined import CleanPreprocessor
from .specialized import MetabolicInteractionNet,pattern_features,joint_numpy
from .train import weights,metrics,choose_threshold
from .comparison import ROOT,dump,youden_threshold,subgroup_masks,paired_ci
from .household import living_alone

GRID=[{'mode':'flat','hidden':8,'pairwise':True,'aux':.2},
      {'mode':'grouped','hidden':4,'pairwise':True,'aux':.2},
      {'mode':'grouped','hidden':8,'pairwise':True,'aux':.2},
      {'mode':'grouped','hidden':4,'pairwise':False,'aux':.2},
      {'mode':'grouped','hidden':4,'pairwise':True,'aux':0.},
      {'mode':'grouped','hidden':4,'pairwise':True,'aux':.5}]
PLAN={'development_years':[2021,2022,2023],'test_year':2024,
    'status':'Reused 2024 exploratory temporal evaluation; never used for this architecture selection',
    'cohort':'Existing complete-lab, no-diagnosis, no-medication, >=12h fasting cohort',
    'grid':GRID,'basis':'hinge','primary_initialization':'Weighted C=0.1 logistic, same development rows; hybrid neural initialization is disclosed',
    'component_initialization':'Four weighted C=0.1 single-finding logistic models, training rows only',
    'nn_selection':'Mean 3-fold group ROC-AUC, then Brier, then parameter count; all candidates reported',
    'early_stop':'Fold-internal PSU-group 20% split, primary BCE + 0.1 component BCE; refit outer training fold at selected epoch',
    'loss':'Weighted primary BCE + aux * mean component BCE + 0.05 positive conditional NLL + primary-skip anchor',
    'max_epochs':220,'patience':25,'seed':42,
    'calibration':'Selected out-of-fold logits split by PSU into 60% calibration and 40% threshold roles; no in-sample final-model predictions used',
    'lr_grid':[{'basis':b,'C':c} for b in ['standard','hinge'] for c in [.03,.1,.3,1.]],
    'limitations':['Selection and calibration reuse development outcomes; internal estimates are not unbiased external validation.',
        'OOF models train on two thirds of development; calibration transfers to a full refit and is not guaranteed.',
        'Interaction paths encode modeling hypotheses, not known causal biological mechanisms.']}

def prefit(frame,basis='hinge'):
    return CleanPreprocessor(basis).fit(features(frame).to_numpy())

def fit_lr(pre,frame,y=None,C=.1):
    m=LogisticRegression(C=C,max_iter=3000,random_state=42)
    m.fit(pre.transform(features(frame).to_numpy()),frame.target if y is None else y,sample_weight=weights(frame));return m

def tensors(pre,d):
    x=torch.tensor(pre.transform(features(d).to_numpy()),dtype=torch.float32)
    y=torch.tensor(d.target.to_numpy(),dtype=torch.float32)
    joint=torch.tensor(d.joint_target.to_numpy(),dtype=torch.long)
    bits=torch.tensor(BITS[d.joint_target.to_numpy()],dtype=torch.float32)
    w=torch.tensor(weights(d),dtype=torch.float32)
    return x,y,joint,bits,w

def fit_nn(train,cfg,seed,epochs,valid=None):
    torch.manual_seed(seed);np.random.seed(seed)
    pre=prefit(train);x,y,j,bits,w=tensors(pre,train)
    primary=fit_lr(pre,train)
    comps=[fit_lr(pre,train,y=BITS[train.joint_target.to_numpy(),k]) for k in range(4)]
    net=MetabolicInteractionNet(x.shape[1],cfg['hidden'],cfg['mode'],cfg['pairwise']);net.initialize(primary,comps)
    anchor=net.output.weight[0,:x.shape[1]].detach().clone()
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01)
    vdata=tensors(pre,valid) if valid is not None else None
    bestloss=float('inf');bestepoch=epochs;stale=0;history=[]
    for epoch in range(epochs):
        net.train();opt.zero_grad();logits=net(x);prob=net.joint(logits)
        primary_loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0],y,reduction='none')
        marginal=prob[:,1:]@net.positive_bits
        comp_loss=torch.nn.functional.binary_cross_entropy(marginal.clamp(1e-6,1-1e-6),bits,reduction='none').mean(dim=1)
        pos=y.bool();nll=torch.nn.functional.cross_entropy(logits[pos,1:]@net.patterns.T,j[pos]-1,reduction='none')
        loss=((primary_loss+cfg['aux']*comp_loss)*w).sum()/w.sum()+.05*(nll*w[pos]).sum()/w.sum()
        loss=loss+.05*(net.output.weight[0,:x.shape[1]]-anchor).square().sum()
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if vdata is not None:
            net.eval();vx,vy,vj,vbits,vw=vdata
            with torch.no_grad():
                vz=net(vx);vp=net.joint(vz);vm=vp[:,1:]@net.positive_bits
                primary_v=torch.nn.functional.binary_cross_entropy_with_logits(vz[:,0],vy,reduction='none')
                component_v=torch.nn.functional.binary_cross_entropy(vm.clamp(1e-6,1-1e-6),vbits,reduction='none').mean(dim=1)
                value=float(((primary_v+.1*component_v)*vw).sum()/vw.sum())
            history.append(value)
            if value<bestloss-1e-5:bestloss=value;bestepoch=epoch+1;stale=0
            else:stale+=1
            if stale>=25:break
    net.eval();return pre,net,bestepoch,history

def predict_logits(pre,net,d):
    with torch.inference_mode():return net(torch.tensor(pre.transform(features(d).to_numpy()),dtype=torch.float32)).numpy()

def binary_temp(z,y,w):
    fit=minimize_scalar(lambda t:log_loss(y,expit(z/np.exp(t)),sample_weight=w,labels=[0,1]),bounds=(-2.3,2.3),method='bounded')
    if not fit.success:raise RuntimeError('Calibration failed')
    return float(np.exp(fit.x))

def calibrate(z,dev,cal_ix,threshold_ix,pairwise=None):
    y=dev.target.to_numpy();w=weights(dev)
    risk_z=z if z.ndim==1 else z[:,0]
    rt=binary_temp(risk_z[cal_ix],y[cal_ix],w[cal_ix]);pt=None
    if pairwise is not None:
        pos=cal_ix[y[cal_ix]==1];j=dev.joint_target.to_numpy()[pos]-1;matrix=pattern_features(pairwise)
        fit=minimize_scalar(lambda t:float(np.average(-np.log(np.clip(softmax(z[pos,1:]@matrix.T/np.exp(t),axis=1)[np.arange(len(j)),j],1e-12,1)),weights=w[pos])),bounds=(-2.3,2.3),method='bounded')
        if not fit.success:raise RuntimeError('Conditional calibration failed')
        pt=float(np.exp(fit.x))
    p=expit(risk_z[threshold_ix]/rt);yt=y[threshold_ix];wt=w[threshold_ix]
    cuts={'youden':youden_threshold(yt,p,wt),'sensitivity90':choose_threshold(yt,p,wt,.9)}
    return rt,pt,cuts

def eval_model(test,p,cuts):
    y=test.target.to_numpy();w=weights(test)
    out={'policies':{k:{'weighted':metrics(y,p,v,w),'unweighted':metrics(y,p,v)} for k,v in cuts.items()},'subgroups':{}}
    for name,mask in subgroup_masks(test).items():
        if mask.any():out['subgroups'][name]={'n':int(mask.sum()),'positive':int(y[mask].sum()),
            'metrics':{k:metrics(y[mask],p[mask],v,w[mask]) for k,v in cuts.items()}}
    return out

def main(name):
    out=ROOT/'artifacts'/name;out.mkdir(parents=True,exist_ok=False);dump(out/'PLAN.json',PLAN)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    d,flow=load_cohort(ROOT);d=d.copy();d['living_alone']=living_alone(d.cfam)
    dev=d[d.survey_year<=2023].copy();test=d[d.survey_year==2024].copy()
    assert len(dev)==3062 and set(dev.group).isdisjoint(test.group)
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026).split(dev,dev.target,dev.group))
    calibration_ix,threshold_ix=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=773).split(dev,groups=dev.group))
    assert set(dev.iloc[calibration_ix].group).isdisjoint(dev.iloc[threshold_ix].group)
    cv={'nn':[],'lr':[]};oof={};identifiers=[hashlib.sha256(f'{r.survey_year}:{r.ID}'.encode()).hexdigest() for r in dev.itertuples()]
    manifest={'ids':identifiers,'folds':[{'train':tr.tolist(),'validation':va.tolist()} for tr,va in folds],
        'calibration_oof_rows':calibration_ix.tolist(),'threshold_oof_rows':threshold_ix.tolist()};dump(out/'split_manifest.json',manifest)
    for kind,grid in [('lr',PLAN['lr_grid']),('nn',GRID)]:
        for index,cfg in enumerate(grid):
            outputs=None;foldmetrics=[];epochs=[];count=None
            for fold,(tr,va) in enumerate(folds):
                train,valid=dev.iloc[tr],dev.iloc[va];assert set(train.group).isdisjoint(valid.group)
                if kind=='lr':
                    pre=prefit(train,cfg['basis']);model=fit_lr(pre,train,C=cfg['C']);z=model.decision_function(pre.transform(features(valid).to_numpy()));p=expit(z);count=len(model.coef_[0])+1
                else:
                    ia,ib=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=100+fold).split(train,groups=train.group))
                    _,_,best,_=fit_nn(train.iloc[ia],cfg,42+fold,220,train.iloc[ib]);epochs.append(best)
                    pre,model,_,_=fit_nn(train,cfg,42+fold,best);z=predict_logits(pre,model,valid);p=expit(z[:,0]);count=sum(v.numel() for v in model.parameters())
                if outputs is None:outputs=np.full((len(dev),)+z.shape[1:],np.nan)
                outputs[va]=z
                foldmetrics.append({'auc':float(roc_auc_score(valid.target,p,sample_weight=weights(valid))),
                    'ap':float(average_precision_score(valid.target,p,sample_weight=weights(valid))),
                    'brier':float(brier_score_loss(valid.target,p,sample_weight=weights(valid))),'n':len(valid)})
            assert np.isfinite(outputs).all()
            info={'id':f'{kind}_{index}','config':cfg,'folds':foldmetrics,'epochs':epochs,'parameters':count,
                'mean_auc':float(np.mean([x['auc'] for x in foldmetrics])),'mean_brier':float(np.mean([x['brier'] for x in foldmetrics]))}
            cv[kind].append(info);oof[info['id']]=outputs;np.save(out/(info['id']+'_oof.npy'),outputs)
            dump(out/'cv_progress.json',cv);print(info['id'],cfg,'CV',round(info['mean_auc'],5),'epochs',epochs,flush=True)
    selected={k:sorted(v,key=lambda a:(-a['mean_auc'],a['mean_brier'],a['parameters']))[0] for k,v in cv.items()}
    dump(out/'SELECTION.json',selected);dump(out/'cv_results.json',cv)
    chosen=selected['nn'];cfg=chosen['config'];epochs=max(1,int(np.median(chosen['epochs'])))
    rt,pt,cuts=calibrate(oof[chosen['id']],dev,calibration_ix,threshold_ix,cfg['pairwise'])
    pre,net,_,_=fit_nn(dev,cfg,42,epochs);modeldir=out/'model';modeldir.mkdir();pre.save(modeldir/'preprocessor.npz')
    np.savez(modeldir/'network.npz',**{k:v.detach().numpy() for k,v in net.state_dict().items()})
    config=cfg|{'name':name,'risk_temperature':rt,'pattern_temperature':pt,'thresholds':cuts,'train_n':len(dev),'epochs':epochs,
        'calibration_method':'3-fold out-of-fold predictions split by PSU into calibration/threshold roles','parameters':sum(v.numel() for v in net.parameters())}
    dump(modeldir/'config.json',config)
    lc=selected['lr']['config'];lp=prefit(dev,lc['basis']);lm=fit_lr(lp,dev,C=lc['C']);lt,_,lcut=calibrate(oof[selected['lr']['id']],dev,calibration_ix,threshold_ix)
    lrdir=out/'logistic';lrdir.mkdir();lp.save(lrdir/'preprocessor.npz');np.savez(lrdir/'network.npz',coef=lm.coef_[0],bias=lm.intercept_[0])
    dump(lrdir/'config.json',lc|{'temperature':lt,'thresholds':lcut,'train_n':len(dev)})
    # All architecture choices, refits, calibration and cutoffs are now frozen.
    z=predict_logits(pre,net,test);joint=joint_numpy(z,cfg['pairwise'],rt,pt);p=joint[:,1:].sum(axis=1)
    lrprob=expit(lm.decision_function(lp.transform(features(test).to_numpy()))/lt)
    report={'plan':PLAN,'selection':selected,'cohort_flow':flow,'development_n':len(dev),'test_n':len(test),
        'oof_calibration_n':len(calibration_ix),'oof_threshold_n':len(threshold_ix),
        'models':{'specialized_nn':eval_model(test,p,cuts),'tuned_lr':eval_model(test,lrprob,lcut)},
        'paired_nn_minus_lr':paired_ci(test,p,lrprob),
        'components':{c:metrics(BITS[test.joint_target.to_numpy(),j],(joint@BITS)[:,j],.5,weights(test)) for j,c in enumerate(COMPONENTS)},
        'count_mae':float(np.average(abs(joint@BITS.sum(axis=1)-BITS[test.joint_target.to_numpy()].sum(axis=1)),weights=weights(test))),
        'source_hashes':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'metabolic').glob('*.py')},
        'raw_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'data/raw/knhanes').glob('*.zip')}}
    dump(out/'report.json',report);dump(out/'test_predictions.json',{'target':test.target.tolist(),'nn':p.tolist(),'lr':lrprob.tolist()})
    print(json.dumps({'selected':selected,'test':{k:{z:v['policies']['youden']['weighted'][z] for z in ['roc_auc','pr_auc_ap','brier','sensitivity','specificity']} for k,v in report['models'].items()}},indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run-name',default='specialized_v3');args=a.parse_args()
    if Path(args.run_name).name!=args.run_name:a.error('Invalid artifact directory name')
    main(args.run_name)
