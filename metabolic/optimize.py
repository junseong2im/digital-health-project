"""Development-only group CV selection, then frozen temporal evaluation."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import joblib
from scipy.optimize import minimize_scalar
from scipy.special import expit,softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from .data import load_cohort,features,FEATURES
from .comparison import restore_parts,OLD,ROOT,dump,thresholds,evaluate,paired_ci
from .household import living_alone
from .train import weights
from .refined import CleanPreprocessor,RiskPatternNet,joint_from_logits

PLAN={'name':'primary_risk_optimization_v2',
    'cohort':'Same 4068 eligible participants; no data removed based on outcomes or test metrics',
    'selection':'Only 2021-2022, stratified group 3-fold CV by year-PSU; fold-internal early stopping; refit outer train before evaluation',
    'calibration':'Original 2023 calibration subset; primary and conditional temperatures independently fit',
    'threshold':'Original 2023 threshold subset, Youden and sensitivity90',
    'test':'Original already-inspected 2024; exploratory final evaluation, never used for selection',
    'preprocessing':'Train-only medians, means/scales, optional hinge quartiles; explicit missing indicators and categorical missing levels',
    'objective':'Primary binary BCE + 0.10 conditional positive-pattern CE; one-pass coherent joint output',
    'grid_nn':[{'basis':b,'hidden':h,'seed':42,'weight_decay':.01} for b in ['standard','hinge'] for h in [8,16,32]],
    'grid_lr':[{'basis':b,'C':c} for b in ['standard','hinge'] for c in [.1,1,10]],
    'selection_metric':'Mean fold survey-weighted ROC-AUC (Brier as tie-break); no test-based seed or winner selection',
    'training':{'lr':.003,'max_epochs':180,'patience':20,'batch':'full train batch'},
    'latency':'Benchmark optimized NN and optimized LR through identical validation/feature/output paths; also preserve old runtime comparison'}

def quality_audit(d):
    x=features(d);out={'n':len(d),'duplicates':int(d.duplicated(['survey_year','ID']).sum()),'pregnancy_codes':{str(k):int(v) for k,v in d.HE_prg.value_counts(dropna=False).items()},'features':{},'year_counts':{str(k):int(v) for k,v in d.survey_year.value_counts().items()}}
    for c in FEATURES:
        s=x[c];out['features'][c]={'missing':int(s.isna().sum()),'minimum':float(s.min()),'maximum':float(s.max())}
    out['note']='No invalid categories or duplicate participants observed. Numeric predictors within API measurement ranges. True extreme laboratory targets retained; no test-derived clipping.'
    assert out['duplicates']==0
    assert set(d.HE_prg.unique()) <= {0,8}
    return out

def logistic(pre,train,C=1):
    m=LogisticRegression(C=C,max_iter=3000,random_state=42)
    m.fit(pre.transform(features(train).to_numpy()),train.target,sample_weight=weights(train));return m

def train_net(train,basis,hidden,seed,epochs,valid=None):
    torch.manual_seed(seed);np.random.seed(seed)
    pre=CleanPreprocessor(basis).fit(features(train).to_numpy())
    x=torch.tensor(pre.transform(features(train).to_numpy()),dtype=torch.float32)
    y=torch.tensor(train.target.to_numpy(),dtype=torch.float32);j=torch.tensor(train.joint_target.to_numpy()-1,dtype=torch.long)
    w=torch.tensor(weights(train),dtype=torch.float32)
    lr=logistic(pre,train)
    net=RiskPatternNet(x.shape[1],hidden);net.initialize_risk(lr.coef_[0],lr.intercept_[0])
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01)
    if valid is not None:
        vx=torch.tensor(pre.transform(features(valid).to_numpy()),dtype=torch.float32)
        vy=torch.tensor(valid.target.to_numpy(),dtype=torch.float32);vw=torch.tensor(weights(valid),dtype=torch.float32)
    bestloss=float('inf');stale=0;bestepoch=epochs
    for epoch in range(epochs):
        net.train();opt.zero_grad();z=net(x)
        binary=torch.nn.functional.binary_cross_entropy_with_logits(z[:,0],y,reduction='none')
        pos=y.bool();cond=torch.nn.functional.cross_entropy(z[pos,1:],j[pos],reduction='none')
        loss=(binary*w).sum()/w.sum()+.1*(cond*w[pos]).sum()/w.sum()
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if valid is not None:
            net.eval()
            with torch.no_grad():v=float((torch.nn.functional.binary_cross_entropy_with_logits(net(vx)[:,0],vy,reduction='none')*vw).sum()/vw.sum())
            if v<bestloss-1e-5:bestloss=v;stale=0;bestepoch=epoch+1
            else:stale+=1
            if stale>=20:break
    net.eval();return pre,net,bestepoch

def logits(pre,net,d):
    with torch.inference_mode():return net(torch.tensor(pre.transform(features(d).to_numpy()),dtype=torch.float32)).numpy()

def temperature_binary(z,d):
    fit=minimize_scalar(lambda lt:log_loss(d.target,expit(z/np.exp(lt)),sample_weight=weights(d),labels=[0,1]),bounds=(-2.3,2.3),method='bounded')
    if not fit.success:raise RuntimeError('Calibration failed')
    return float(np.exp(fit.x))

def main(name):
    out=ROOT/'artifacts'/name;out.mkdir(parents=True,exist_ok=False);dump(out/'PLAN.json',PLAN)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    d,flow=load_cohort(ROOT);d=d.copy();d['living_alone']=living_alone(d.cfam)
    dump(out/'data_quality.json',quality_audit(d))
    manifest=json.loads((OLD/'split_manifest.json').read_text());parts=restore_parts(d,manifest)
    dev=d[d.survey_year<=2022].copy()
    splits=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026).split(dev,dev.target,dev.group))
    cv={'nn':[],'lr':[]};oof={}
    for kind,grid in [('lr',PLAN['grid_lr']),('nn',PLAN['grid_nn'])]:
        for i,cfg in enumerate(grid):
            pred=np.full(len(dev),np.nan);folds=[];epochs=[]
            for fold,(tr,te) in enumerate(splits):
                train,val=dev.iloc[tr],dev.iloc[te]
                assert set(train.group).isdisjoint(val.group)
                if kind=='lr':
                    pre=CleanPreprocessor(cfg['basis']).fit(features(train).to_numpy());model=logistic(pre,train,cfg['C'])
                    p=model.predict_proba(pre.transform(features(val).to_numpy()))[:,1]
                else:
                    inner_a,inner_b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=101+fold).split(train,groups=train.group))
                    _,_,best=train_net(train.iloc[inner_a],cfg['basis'],cfg['hidden'],42+fold,180,train.iloc[inner_b])
                    pre,model,_=train_net(train,cfg['basis'],cfg['hidden'],42+fold,best)
                    p=expit(logits(pre,model,val)[:,0]);epochs.append(best)
                pred[te]=p
                folds.append({'auc':float(roc_auc_score(val.target,p,sample_weight=weights(val))),
                    'brier':float(brier_score_loss(val.target,p,sample_weight=weights(val))),'n':len(val)})
            assert np.isfinite(pred).all()
            result={'config':cfg,'folds':folds,'mean_auc':float(np.mean([f['auc'] for f in folds])),
                'mean_brier':float(np.mean([f['brier'] for f in folds])),'epochs':epochs,'oof_auc':float(roc_auc_score(dev.target,pred,sample_weight=weights(dev)))}
            cv[kind].append(result);oof[f'{kind}_{i}']=pred.tolist();dump(out/'cv_progress.json',cv)
            print(kind,str(cfg),'CV AUC',round(result['mean_auc'],5),flush=True)
    selected={k:sorted(v,key=lambda a:(-a['mean_auc'],a['mean_brier']))[0] for k,v in cv.items()}
    # Persist selection before evaluating 2023 calibration or 2024 outcomes.
    dump(out/'SELECTION.json',selected);dump(out/'cv_results.json',cv);dump(out/'development_oof_predictions.json',oof)
    c=selected['nn']['config'];nepoch=max(1,int(np.median(selected['nn']['epochs'])))
    pre,net,_=train_net(dev,c['basis'],c['hidden'],42,nepoch)
    z={k:logits(pre,net,v) for k,v in parts.items() if k not in ['train','validation']}
    rt=temperature_binary(z['calibration'][:,0],parts['calibration'])
    cal=parts['calibration'];pos=cal.target.to_numpy()==1;j=cal.joint_target.to_numpy()[pos]-1;wc=weights(cal)[pos]
    fit=minimize_scalar(lambda lt:float(np.average(-np.log(np.clip(softmax(z['calibration'][pos,1:]/np.exp(lt),axis=1)[np.arange(len(j)),j],1e-12,1)),weights=wc)),bounds=(-2.3,2.3),method='bounded')
    if not fit.success:raise RuntimeError('Pattern calibration failed')
    pt=float(np.exp(fit.x));probs={k:expit(v[:,0]/rt) for k,v in z.items()};cuts=thresholds(parts,probs)
    modeldir=out/'model';modeldir.mkdir();pre.save(modeldir/'preprocessor.npz')
    np.savez(modeldir/'network.npz',**{k:v.detach().numpy() for k,v in net.state_dict().items()})
    config={'name':name,'basis':c['basis'],'hidden':c['hidden'],'epochs':nepoch,'seed':42,'risk_temperature':rt,
        'pattern_temperature':pt,'thresholds':cuts,'train_n':len(dev),'parameter_count':sum(p.numel() for p in net.parameters()),
        'selected_on':'2021-2022 group CV only','architecture':'logistic skip + tanh hidden residual; primary risk and conditional 15-pattern output'}
    dump(modeldir/'config.json',config)
    trained_lr={};results={}
    # Tuned LR and LR with exactly the selected neural feature basis.
    for lrname,cfg in [('tuned_lr',selected['lr']['config']),('matched_basis_lr',{'basis':c['basis'],'C':1})]:
        lp=CleanPreprocessor(cfg['basis']).fit(features(dev).to_numpy());lm=logistic(lp,dev,cfg['C'])
        lz={k:lm.decision_function(lp.transform(features(v).to_numpy())) for k,v in parts.items() if k not in ['train','validation']}
        lt=temperature_binary(lz['calibration'],cal);pr={k:expit(v/lt) for k,v in lz.items()};lc=thresholds(parts,pr)
        ld=out/lrname;ld.mkdir();lp.save(ld/'preprocessor.npz');np.savez(ld/'network.npz',coef=lm.coef_[0],bias=lm.intercept_[0])
        dump(ld/'config.json',{'basis':cfg['basis'],'C':cfg['C'],'temperature':lt,'thresholds':lc,'train_n':len(dev)})
        trained_lr[lrname]=pr;results[lrname]=evaluate(parts,pr,{'thresholds':lc})
    results['refined_nn']=evaluate(parts,probs,{'thresholds':cuts})
    pairs={name:paired_ci(parts['test_2024'],probs['test_2024'],p['test_2024']) for name,p in trained_lr.items()}
    from .comparison import load_frozen_nn
    oldp,oldcfg=load_frozen_nn(parts);results['frozen_v1']=evaluate(parts,oldp,oldcfg)
    report={'plan':PLAN,'selection':selected,'models':results,'paired_nn_minus_lr':pairs,'nn_config':config,
        'caveat':'2024 is a reused exploratory holdout. Tuning uses only pre-2023 data. New models use 2052 development rows vs old model 1680; matched LR uses 2052 too.',
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'metabolic').glob('*.py')},
        'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/raw/knhanes').glob('*.zip')}}
    dump(out/'report.json',report)
    dump(out/'test_predictions.json',{'target':parts['test_2024'].target.tolist(),'refined_nn':probs['test_2024'].tolist(),
        **{k:v['test_2024'].tolist() for k,v in trained_lr.items()}})
    print(json.dumps({'output':str(out),'selection':selected,'test':{k:v['policies']['youden']['weighted'] for k,v in results.items()}},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='optimized_v2');a=p.parse_args()
    if Path(a.run_name).name!=a.run_name:p.error('Invalid output directory name')
    main(a.run_name)
