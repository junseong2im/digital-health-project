"""Fixed 2x2 input/task ablation with five independent LR comparators."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit,softmax
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from .data import load_cohort,features,BITS,COMPONENTS
from .expanded import extended_features,ExpandedPreprocessor,FactorialNet,EXTRA,SOURCE
from .comparison import ROOT,dump,paired_ci,subgroup_masks
from .train import weights,metrics,choose_threshold
from .train_specialized import binary_temp
from .comparison import youden_threshold
from .household import living_alone

PLAN={'design':'2x2: base/expanded inputs x single/multitask NN; no auxiliary labels in single-task initialization, loss or early stopping',
    'extra_features':EXTRA,'seeds':[42,43,44],'NN_hidden':4,'NN_basis':'hinge','LR_C':[.03,.1,.3],
    'cohort':'Same 4068; 3062 development 2021-2023; reused 2024 evaluation n1006',
    'splits':'Exactly specialized_v3 OOF folds and PSU-separated OOF calibration/threshold roles',
    'training':'Fold-internal primary-BCE early stopping for both tasks, then refit outer training fold. LR initialization C=0.1. Multi BCE+.2 marginal BCE+.05 conditional NLL.',
    'primary_nn':'Seed42 prespecified for detailed CI; other seeds stability only, never choose best test seed',
    'LR_comparison':'Five independently trained/tuned logistic outputs per feature set, C selected per endpoint using development OOF only',
    'calibration':'OOF calibration role only; threshold role distinct. Both re-use model-selection development outcomes; exploratory.',
    'evaluation':['overall AUC/AP/Brier','four individual endpoints','macro four-endpoint AUC/AP/Brier','expected abnormality-count MAE','coherence','household strata'],
    'exclusions':['sleep question form changed in 2024','dietary recall variables require nutrition participation/weight handling','personal diagnosis, medication, fasting and laboratory variables never added as predictors'],
    'stop_rule':'Report all planned cells and seeds; no 2024-based feature or architecture selection'}

def matrix(d,expanded):return (extended_features(d) if expanded else features(d)).to_numpy()

def fit_lrs(x,d,C=.1,multitask=True):
    y=np.column_stack([d.target.to_numpy(),BITS[d.joint_target.to_numpy()]]) if multitask else d.target.to_numpy()[:,None]
    out=[]
    for j in range(y.shape[1]):
        model=LogisticRegression(C=C,max_iter=3000,random_state=42).fit(x,y[:,j],sample_weight=weights(d));out.append(model)
    return out

def fit_nn(train,expanded,multi,seed,epochs,valid=None):
    torch.manual_seed(seed);np.random.seed(seed)
    pre=ExpandedPreprocessor(expanded).fit(matrix(train,expanded));x=pre.transform(matrix(train,expanded))
    lr=fit_lrs(x,train,multitask=multi);net=FactorialNet(pre,multi);net.initialize(lr,x)
    tx=torch.tensor(x,dtype=torch.float32);ty=torch.tensor(train.target.to_numpy(),dtype=torch.float32)
    tj=torch.tensor(train.joint_target.to_numpy(),dtype=torch.long);tw=torch.tensor(weights(train),dtype=torch.float32)
    bits=torch.tensor(BITS[train.joint_target.to_numpy()],dtype=torch.float32)
    anchor=net.output.weight[0,:pre.output_dim].detach().clone();opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01)
    if valid is not None:
        vx=torch.tensor(pre.transform(matrix(valid,expanded)),dtype=torch.float32);vy=torch.tensor(valid.target.to_numpy(),dtype=torch.float32);vw=torch.tensor(weights(valid),dtype=torch.float32)
    best=float('inf');bestepoch=epochs;stale=0
    for epoch in range(epochs):
        net.train();opt.zero_grad();z=net(tx)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(z[:,0],ty,reduction='none')*tw).sum()/tw.sum()
        if multi:
            p=net.joint(z);m=p[:,1:]@net.bits
            aux=torch.nn.functional.binary_cross_entropy(m.clamp(1e-6,1-1e-6),bits,reduction='none').mean(dim=1)
            pos=ty.bool();cn=torch.nn.functional.cross_entropy(z[pos,1:]@net.bits.T,tj[pos]-1,reduction='none')
            loss=loss+.2*(aux*tw).sum()/tw.sum()+.05*(cn*tw[pos]).sum()/tw.sum()
        loss=loss+.05*(net.output.weight[0,:pre.output_dim]-anchor).square().sum()
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if valid is not None:
            net.eval()
            with torch.no_grad():value=float((torch.nn.functional.binary_cross_entropy_with_logits(net(vx)[:,0],vy,reduction='none')*vw).sum()/vw.sum())
            if value<best-1e-5:best=value;bestepoch=epoch+1;stale=0
            else:stale+=1
            if stale>=20:break
    net.eval();return pre,net,bestepoch

def logits(pre,model,d,expanded):
    with torch.inference_mode():return model(torch.tensor(pre.transform(matrix(d,expanded)),dtype=torch.float32)).numpy()

def calibrate(z,dev,cal,thr,multi):
    y=dev.target.to_numpy();w=weights(dev);temperature=binary_temp(z[cal,0],y[cal],w[cal]);pattern=1.
    if multi:
        pos=cal[y[cal]==1];j=dev.joint_target.to_numpy()[pos]-1
        fit=minimize_scalar(lambda lt:float(np.average(-np.log(np.clip(softmax(z[pos,1:]@BITS[1:].T/np.exp(lt),axis=1)[np.arange(len(j)),j],1e-12,1)),weights=w[pos])),bounds=(-2.3,2.3),method='bounded')
        if not fit.success:raise RuntimeError('Conditional calibration failed')
        pattern=float(np.exp(fit.x))
    p=expit(z[thr,0]/temperature)
    cuts={'youden':youden_threshold(y[thr],p,w[thr]),'sensitivity90':choose_threshold(y[thr],p,w[thr],.9)}
    return temperature,pattern,cuts

def evaluate(test,p,cut,components=None):
    y=test.target.to_numpy();w=weights(test)
    out={'primary':{k:metrics(y,p,t,w) for k,t in cut.items()},'subgroups':{}}
    for name,mask in subgroup_masks(test).items():
        if mask.any():out['subgroups'][name]={'n':int(mask.sum()),'metrics':metrics(y[mask],p[mask],cut['youden'],w[mask])}
    if components is not None:
        truth=BITS[test.joint_target.to_numpy()]
        out['components']={c:metrics(truth[:,j],components[:,j],.5,w) for j,c in enumerate(COMPONENTS)}
        out['macro']={k:float(np.mean([v[k] for v in out['components'].values()])) for k in ['roc_auc','pr_auc_ap','brier']}
        out['count_mae']=float(np.average(abs(components.sum(axis=1)-truth.sum(axis=1)),weights=w))
        # Necessary union bounds, not a full test of consistency of independent marginals.
        out['union_bound_violation_n']=int(np.sum((components.max(axis=1)>p+1e-6)|(p>np.minimum(1,components.sum(axis=1))+1e-6)))
    return out

def main(name):
    out=ROOT/'artifacts'/name;out.mkdir(parents=True,exist_ok=False);dump(out/'PLAN.json',PLAN)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    d,flow=load_cohort(ROOT);d=d.copy();d['living_alone']=living_alone(d.cfam)
    dev=d[d.survey_year<=2023].copy();test=d[d.survey_year==2024].copy()
    source=json.loads((ROOT/'artifacts/specialized_v3/split_manifest.json').read_text())
    ids=[hashlib.sha256(f'{r.survey_year}:{r.ID}'.encode()).hexdigest() for r in dev.itertuples()]
    assert ids==source['ids'];folds=[(np.array(v['train']),np.array(v['validation'])) for v in source['folds']]
    cal=np.array(source['calibration_oof_rows']);thr=np.array(source['threshold_oof_rows']);dump(out/'split_manifest.json',source)
    audit={'raw_variables':SOURCE,'extra_features':EXTRA,'by_year':{},'cohort_n':len(d)}
    for year,frame in d.groupby('survey_year'):
        x=extended_features(frame);audit['by_year'][str(year)]={c:{'missing':int(x[c].isna().sum()),'n':len(x),'min':float(x[c].min()),'max':float(x[c].max())} for c in EXTRA}
    audit['rules']={'alcohol_nonuse':'BD1=1 & BD1_11=8 -> frequency0; amount8 ->0 only if lifetime non-drinker or last-year no drinking',
        'walking':'BE3_31=1..8 ->0..7 days; 88/99 ->missing','strength':'BE5_1=1..6 ->0..5 categories;5 means >=5 days',
        'sedentary':'BE8_1 0..24 + BE8_2 0..59 /60; total<=24; 88/99 ->missing','others':'Use verified finite code sets; no general sentinel-to-zero replacement'}
    dump(out/'feature_audit.json',audit)
    cv={};predictions={};report={'plan':PLAN,'models':{},'cohort_flow':flow};frozen=[]
    # Every planned model is fitted and frozen before final test metrics are calculated.
    for expanded in [False,True]:
        tag='expanded' if expanded else 'base';lr_candidates=[];lr_oof={}
        for c in PLAN['LR_C']:
            oof=np.full((len(dev),5),np.nan)
            for tr,va in folds:
                train,val=dev.iloc[tr],dev.iloc[va];assert set(train.group).isdisjoint(val.group)
                pre=ExpandedPreprocessor(expanded).fit(matrix(train,expanded));models=fit_lrs(pre.transform(matrix(train,expanded)),train,C=c)
                xv=pre.transform(matrix(val,expanded));oof[va]=np.column_stack([m.decision_function(xv) for m in models])
            target=np.column_stack([dev.target.to_numpy(),BITS[dev.joint_target.to_numpy()]])
            auc=[float(roc_auc_score(target[:,j],expit(oof[:,j]),sample_weight=weights(dev))) for j in range(5)]
            lr_candidates.append({'C':c,'endpoint_oof_auc':auc});lr_oof[c]=oof
        chosen=[max(lr_candidates,key=lambda v:v['endpoint_oof_auc'][j])['C'] for j in range(5)]
        lp=ExpandedPreprocessor(expanded).fit(matrix(dev,expanded));xd=lp.transform(matrix(dev,expanded));models=[]
        target=np.column_stack([dev.target.to_numpy(),BITS[dev.joint_target.to_numpy()]])
        temperatures=[]
        for j,c in enumerate(chosen):
            m=LogisticRegression(C=c,max_iter=3000,random_state=42).fit(xd,target[:,j],sample_weight=weights(dev));models.append(m)
            temperatures.append(binary_temp(lr_oof[c][cal,j],target[cal,j],weights(dev)[cal]))
        pthr=expit(lr_oof[chosen[0]][thr,0]/temperatures[0]);wt=weights(dev)[thr];yt=dev.target.to_numpy()[thr]
        cuts={'youden':youden_threshold(yt,pthr,wt),'sensitivity90':choose_threshold(yt,pthr,wt,.9)}
        folder=out/'models'/(tag+'_LR5');folder.mkdir(parents=True);lp.save(folder)
        np.savez(folder/'network.npz',coef=np.stack([m.coef_[0] for m in models]),bias=np.array([m.intercept_[0] for m in models]))
        dump(folder/'config.json',{'name':tag+'_LR5','chosen_C':chosen,'temperature':temperatures,'thresholds':cuts,'expanded':expanded,'outputs':['any']+COMPONENTS})
        frozen.append((tag+'_LR5','lr',expanded,lp,models,temperatures,cuts))
        cv[tag+'_LR5']=lr_candidates;print(tag+' LR5 fitted',flush=True)
        for multi in [False,True]:
            for seed in PLAN['seeds']:
                modelname=f'{tag}_{"multi" if multi else "single"}_s{seed}';oof=np.full((len(dev),5 if multi else 1),np.nan);bestepochs=[];foldscore=[]
                for k,(tr,va) in enumerate(folds):
                    train,val=dev.iloc[tr],dev.iloc[va]
                    ia,ib=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=100+k).split(train,groups=train.group))
                    _,_,best=fit_nn(train.iloc[ia],expanded,multi,seed+k,180,train.iloc[ib]);bestepochs.append(best)
                    pre,net,_=fit_nn(train,expanded,multi,seed+k,best);z=logits(pre,net,val,expanded);oof[va]=z
                    foldscore.append(float(roc_auc_score(val.target,expit(z[:,0]),sample_weight=weights(val))))
                assert np.isfinite(oof).all()
                t,pt,cuts=calibrate(oof,dev,cal,thr,multi);epochs=max(1,int(np.median(bestepochs)))
                pre,net,_=fit_nn(dev,expanded,multi,seed,epochs)
                folder=out/'models'/modelname;folder.mkdir(parents=True);pre.save(folder)
                np.savez(folder/'network.npz',**{k:v.detach().numpy() for k,v in net.state_dict().items()})
                config={'name':modelname,'expanded':expanded,'multitask':multi,'seed':seed,'temperature':t,'pattern_temperature':pt,'thresholds':cuts,
                    'epochs':epochs,'train_n':len(dev),'parameters':sum(v.numel() for v in net.parameters())}
                dump(folder/'config.json',config);np.save(out/(modelname+'_oof.npy'),oof)
                cv[modelname]={'fold_auc':foldscore,'mean_auc':float(np.mean(foldscore)),'best_epochs':bestepochs,'config':config}
                frozen.append((modelname,'nn',expanded,pre,net,(t,pt),cuts))
                dump(out/'cv_progress.json',cv);print(modelname,'CV',round(np.mean(foldscore),5),'epochs',bestepochs,flush=True)
    dump(out/'FROZEN_SELECTION.json',cv)
    for modelname,kind,expanded,pre,model,temps,cuts in frozen:
        if kind=='lr':
            xt=pre.transform(matrix(test,expanded));p5=expit(np.column_stack([m.decision_function(xt) for m in model])/np.array(temps));p=p5[:,0];components=p5[:,1:]
        else:
            z=logits(pre,model,test,expanded);p=expit(z[:,0]/temps[0]);components=None
            if model.multitask:
                q=softmax(z[:,1:]@BITS[1:].T/temps[1],axis=1);components=(p[:,None]*q)@BITS[1:]
        report['models'][modelname]=evaluate(test,p,cuts,components)
        predictions[modelname]={'primary':p.tolist(),'components':components.tolist() if components is not None else None}
    report['paired_differences']={}
    comparisons={'input_effect_single':('expanded_single_s42','base_single_s42'),'input_effect_multi':('expanded_multi_s42','base_multi_s42'),
        'task_effect_base':('base_multi_s42','base_single_s42'),'task_effect_expanded':('expanded_multi_s42','expanded_single_s42'),
        'expanded_multi_vs_LR':('expanded_multi_s42','expanded_LR5')}
    for key,(a,b) in comparisons.items():report['paired_differences'][key]=paired_ci(test,np.array(predictions[a]['primary']),np.array(predictions[b]['primary']))
    report['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'metabolic').glob('*.py')}
    report['raw_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/raw/knhanes').glob('*.zip')}
    dump(out/'report.json',report);dump(out/'test_predictions.json',predictions)
    print(json.dumps({'artifact':str(out),'seed42':{k:{'auc':v['primary']['youden']['roc_auc'],'ap':v['primary']['youden']['pr_auc_ap'],'macro':v.get('macro')} for k,v in report['models'].items() if 's42' in k or 'LR5' in k}},indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run-name',default='expanded_multitask_v4');args=a.parse_args()
    if Path(args.run_name).name!=args.run_name:a.error('Invalid output directory')
    main(args.run_name)
