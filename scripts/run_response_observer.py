"""Frozen finite pilot: causal response observer, matched continuation, MLP.

Past state inputs are v/r only, timestamps t-14:t. Actions t-14:t-1.
Current v/r at t and future actions t:t+H-1 remain inherited inputs.
"""
from pathlib import Path
import sys,argparse,subprocess,time,copy
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_physical_scene as parent
ex=parent.ex;b=parent.b
from baselines.response_observer import ResponseObserver
from baselines.physical_scene import PhysicalScene,to_fixed
OUT=b.ROOT/'reports/response_observer';CK=b.ROOT/'checkpoints/response_observer'
NAMES=['global','observer','history_mlp']


class HistoryData(b.WindowData):
    def __getitem__(self,i):
        e,t=self.idx[i];r=self.data.records[e]
        z,a,y,k,p=super().__getitem__(i)
        history=torch.from_numpy(r['z'][t-14:t+1,3:5]).float()
        past=torch.from_numpy(r['actions'][t-14:t]).float()
        return z,a,y,history,past


def configure():ex.configure()


def records(names):
    rec=ex.CorrectedRecords(names);parent.attach_placeholder(rec);return rec


def make(name,fold,seed):
    stats=dict(np.load(ex.CK/f'{fold}_seed{seed}'/'independent/normalizers.npz'))
    m=ResponseObserver(name,stats['state_mean'][3:5],stats['state_std'][3:5])
    saved=torch.load(parent.CK/f'{fold}_seed{seed}'/'physical_scene.pt',weights_only=False)['model']
    assert set(saved)=={'raw','throttle_reference'}
    with torch.no_grad():m.raw.copy_(saved['raw']);m.throttle_reference.copy_(saved['throttle_reference'])
    return m,stats


@torch.no_grad()
def evaluate_model(m,ds,mode='normal',indices=None):
    if indices is not None:ds=torch.utils.data.Subset(ds,indices)
    errors=[];biases=[];negative=0;count=0
    for z,a,y,h,p in b.loader(ds,128):
        bias=m.estimate_response(h,p,mode);pred=m.forecast(z,a,bias)
        assert torch.isfinite(pred).all() and torch.isfinite(bias).all()
        errors.append(torch.linalg.vector_norm(pred[:,:,:2]-y[:,:,:2],dim=-1).numpy());biases.append(bias.numpy())
        negative+=int((pred[:,:,3]<0).sum());count+=pred.shape[0]*pred.shape[1]
    return np.concatenate(errors),np.concatenate(biases),dict(negative_speed_fraction=negative/count)


def preflight():
    configure();OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    cfg=b.read_json(parent.OUT/'protocol.json');parent.verify(cfg)
    rec=records(cfg['splits']['temporal']['train']);ds=HistoryData(rec,100,4)
    z,a,y,h,p=next(iter(b.loader(torch.utils.data.Subset(ds,list(range(8))),8)))
    checks={}
    for name in NAMES:
        b.seed_all(0);m,stats=make(name,'temporal',0);pred=m.rollout_history(z,a,h,p)
        loss=b.loss_function(pred,y,stats);loss.backward()
        assert torch.isfinite(pred).all() and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        old=PhysicalScene();old.load_state_dict(torch.load(parent.CK/'temporal_seed0/physical_scene.pt',weights_only=False)['model'])
        assert torch.equal(m.rollout_history(z,a,h,p,'zero'),old.rollout(z,a))
        # Deterministic zero-latent ablation exactly matches the warm-start core.
        checks[name]=dict(parameters=sum(v.numel() for v in m.parameters()),finite_backward=True,zero_response_identity=True)
    # No read of future states/actions by initial response or historical input.
    e,t=ds.idx[0];rr=rec.records[e];state_backup=rr['z'].copy();action_backup=rr['actions'].copy()
    before=tuple(v.clone() for v in ds[0]);rr['z'][t+1:]+=100;rr['actions'][t:]+=100;after=ds[0]
    assert torch.equal(before[3],after[3]) and torch.equal(before[4],after[4]) and torch.equal(before[0],after[0])
    assert not torch.equal(before[1],after[1]);rr['z']=state_backup;rr['actions']=action_backup
    # Observer zero innovation and impulse recurrence can be checked analytically.
    m,_=make('observer','temporal',0);res=torch.zeros(3,14,2);res[:,-1]=torch.tensor([1.,-2.])
    m.innovations=lambda history,actions:res
    gain,tau=m.response_parameters();expected=gain*torch.exp(-.5*(1/22)/tau)*res[:,-1]
    assert torch.allclose(m.estimate_response(h[:3],p[:3]),expected,atol=1e-7)
    res.zero_();assert torch.equal(m.estimate_response(h[:3],p[:3]),torch.zeros(3,2))
    b.write_json(OUT/'preflight.json',dict(status='PASS',checks=checks,future_mutation_history_unchanged=True,observer_impulse_identity=True))
    sources=[Path(__file__),b.ROOT/'src/baselines/response_observer.py',b.ROOT/'src/baselines/physical_scene.py']
    inputs={}
    for fold in cfg['splits']:
        for seed in cfg['seeds']:
            job=f'{fold}_seed{seed}';inputs[job]=dict(physical_checkpoint=b.digest(parent.CK/job/'physical_scene.pt'),normalizers=b.digest(ex.CK/job/'independent/normalizers.npz'),windows=b.digest(ex.CK/job/'evaluation_windows.json'))
    frozen=dict(splits=cfg['splits'],seeds=[0,1,2],variants=NAMES,epochs=40,lr=.003,train_horizon=32,test_horizon=100,stride=4,batch=128,
        status='Retrospective exploratory study on repeatedly reused records, not independent confirmation.',
        inputs='15 v/r observations t-14:t and 14 past actions t-14:t-1; causal r uses up to 5 trailing raw yaw differences. No future observation supplied to encoder. Recorded future actions remain given.',
        models='All variants warm-start same inherited 8-parameter physical model and continue fitting 40 epochs. global disables latent response; observer recursively estimates 2 acceleration mismatches; history_mlp encodes identical history into same 2 physical latent coordinates. Same physical rollout and exponential latent decay.',
        selection='Each variant minimum validation E100; all 27 fits finish before any test evaluation. Adam .003, cosine40, clip1, inherited scaled state loss. No post-test retuning.',
        mechanism_gate='Observer mean E100 <=90% of matched continued global on ALL 3 splits, all 3 paired seeds improve on each split.',
        structure_gate='Observer E100 no worse than history_mlp on ALL 3 splits; this gate tests value of structured estimation, not history in general.',
        performance_gate='Observer temporal E100 <=110% of old full, both source holdouts <= old kinematic. Not an acceptance probability or sufficient novelty claim.',
        diagnostics='zero initial response for observer/MLP; observer last transition only, without refitting. Test interventions are distribution-shift diagnostics, not independently trained controls.',
        parent_protocol_sha256=b.digest(parent.OUT/'protocol.json'),sources={str(p.relative_to(b.ROOT)):b.digest(p) for p in sources},frozen_inputs=inputs,
        old_results={str(p.relative_to(b.ROOT)):b.digest(p) for p in [ex.OUT/'results.json',ex.OUT/'curves.npz',parent.OUT/'results.json',parent.OUT/'curves.npz']})
    b.write_json(OUT/'protocol.json',frozen);print('PREFLIGHT PASS; frozen 27 fits',checks,flush=True)


def verify(cfg):
    parent.verify(b.read_json(parent.OUT/'protocol.json'))
    assert b.digest(parent.OUT/'protocol.json')==cfg['parent_protocol_sha256']
    for n,h in cfg['sources'].items():assert b.digest(b.ROOT/n)==h,n
    for n,h in cfg['old_results'].items():assert b.digest(b.ROOT/n)==h,n
    for job,v in cfg['frozen_inputs'].items():
        assert b.digest(parent.CK/job/'physical_scene.pt')==v['physical_checkpoint']
        assert b.digest(ex.CK/job/'independent/normalizers.npz')==v['normalizers']
        assert b.digest(ex.CK/job/'evaluation_windows.json')==v['windows']


def fit(cfg,fold,seed):
    verify(cfg);configure();d=CK/f'{fold}_seed{seed}';d.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'results.json').exists()
    if (d/'COMPLETE.json').exists():return
    split=cfg['splits'][fold];tr=records(split['train']);va=records(split['validation']);td=HistoryData(tr,32,4);vd=HistoryData(va,100,4)
    for name in NAMES:
        if (d/(name+'_history.json')).exists():assert len(b.read_json(d/(name+'_history.json')))==40;continue
        b.seed_all(seed);m,stats=make(name,fold,seed)
        # Reset batch RNG after architecture construction: identical batches.
        b.seed_all(seed);opt=torch.optim.Adam(m.parameters(),lr=cfg['lr']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40);best=float('inf');hist=[]
        for epoch in range(40):
            m.train();total=0.;n=0;gnmax=0.
            for z,a,y,h,p in b.loader(td,128,True):
                pred=m.rollout_history(z,a,h,p);loss=b.loss_function(pred,y,stats);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));assert np.isfinite(gn);opt.step()
                total+=float(loss.detach())*len(z);n+=len(z);gnmax=max(gnmax,gn)
            m.eval();err,_,_=evaluate_model(m,vd);score=float(err[:,-1].mean())
            hist.append(dict(epoch=epoch+1,train_loss=total/n,val_E100=score,max_preclip_gradient=gnmax))
            saved=dict(model=m.state_dict(),epoch=epoch+1,val_E100=score,parameters=sum(v.numel() for v in m.parameters()),variant=name)
            if score<best:best=score;torch.save(saved,d/(name+'.pt'))
            if epoch==39:torch.save(saved,d/(name+'_last.pt'))
            sch.step();b.write_json(d/(name+'_progress.json'),hist);print(d.name,name,epoch+1,'val',round(score,6),flush=True)
        b.write_json(d/(name+'_history.json'),hist)
    b.write_json(d/'COMPLETE.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),files={name+s:b.digest(d/(name+s)) for name in NAMES for s in ['.pt','_last.pt','_history.json']}))


def completed(cfg):
    for fold in cfg['splits']:
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}';mark=b.read_json(d/'COMPLETE.json');assert mark['protocol_sha256']==b.digest(OUT/'protocol.json')
            for n,h in mark['files'].items():assert b.digest(d/n)==h


def evaluate(cfg):
    verify(cfg);completed(cfg);configure();assert not (OUT/'results.json').exists()
    rows={};arrays={};responses={};diagnostics={};params={};checks={};sub={};segments={}
    oldrows=b.read_json(ex.OUT/'results.json')['results'];oldarrays=np.load(ex.OUT/'curves.npz')
    for fold,split in cfg['splits'].items():
        rec=records(split['test']);ds=HistoryData(rec,100,4);ws=[dict(file=rec.records[e]['name'],t=int(t)) for e,t in ds.idx]
        for seed in cfg['seeds']:
            d=CK/f'{fold}_seed{seed}';assert ws==b.read_json(ex.CK/d.name/'evaluation_windows.json')
            for name in NAMES:
                m,_=make(name,fold,seed);saved=torch.load(d/(name+'.pt'),weights_only=False);m.load_state_dict(saved['model']);m.eval()
                hist=b.read_json(d/(name+'_history.json'));best=min(hist,key=lambda v:v['val_E100']);assert len(hist)==40 and saved['epoch']==best['epoch'] and saved['val_E100']==best['val_E100']
                modes=['normal'] if name=='global' else ['normal','zero']+(['last'] if name=='observer' else [])
                for mode in modes:
                    key=f'{fold}/seed{seed}/{name}'+('' if mode=='normal' else '_'+mode)
                    err,response,diag=evaluate_model(m,ds,mode);arrays[key]=err;rows[key]=b.metrics(err);responses[key]=response;diagnostics[key]=diag
                key=f'{fold}/seed{seed}/{name}';idx=np.unique(np.linspace(0,len(ds)-1,24,dtype=int));err,_,_=evaluate_model(m,ds,indices=idx)
                assert np.allclose(err,arrays[key][idx],atol=2e-5,rtol=2e-4)
                gain,tau=m.response_parameters();params[key]=dict(core=m.physical(),observer_gain=gain.detach().tolist(),response_decay_seconds=tau.detach().tolist(),parameters=saved['parameters'])
                checks[key]=dict(recomputed_windows=len(idx),max_error_difference=float(abs(err-arrays[key][idx]).max()))
                print(key,rows[key],flush=True)
            for name in ['kinematic','guarded_full']:
                key=f'{fold}/seed{seed}/{name}';rows[key]=oldrows[key];arrays[key]=oldarrays[key]
            for name in NAMES+['kinematic','guarded_full']:
                key=f'{fold}/seed{seed}/{name}';err=arrays[key]
                segments[key]={file:float(err[[w['file']==file for w in ws],-1].mean()) for file in sorted({w['file'] for w in ws})}
                for source in sorted({b.PAT.fullmatch(w['file'])[1] for w in ws}):
                    mask=np.array([b.PAT.fullmatch(w['file'])[1]==source for w in ws]);sub[f'{fold}/{source}/seed{seed}/{name}']=b.metrics(err[mask])
            b.write_json(d/'evaluation_windows.json',ws)
    np.savez_compressed(OUT/'curves.npz',**arrays);np.savez_compressed(OUT/'estimated_responses.npz',**responses)
    b.write_json(OUT/'results.json',dict(protocol_sha256=b.digest(OUT/'protocol.json'),results=rows,parameters=params,diagnostics=diagnostics))
    for key,err in arrays.items():assert b.metrics(err)==rows[key]
    b.write_json(OUT/'per_source_results.json',sub);b.write_json(OUT/'per_segment_results.json',segments)
    b.write_json(OUT/'audit.json',dict(status='PASS',new_fits=27,rows=len(rows),checks=checks,frozen_sources_and_weights=True,all_validation_minima=True,metrics_recomputed=True,identical_test_windows=True))
    report(cfg)


def report(cfg):
    rows=b.read_json(OUT/'results.json')['results'];means={};gates={};paired={}
    lines=['# 历史响应估计：有限实验','','27 个新增拟合；全局物理继续训练、结构化观测器、相同历史输入的 MLP 初始响应估计器。所有模型使用同一物理递推。','','| 划分 | 模型 | E25 | E50 | E100 | ADE |','|---|---|---:|---:|---:|---:|']
    for fold in cfg['splits']:
        means[fold]={}
        for name in NAMES+['kinematic','guarded_full','observer_zero','observer_last','history_mlp_zero']:
            metrics=[]
            for metric in ['E25','E50','E100','ADE']:
                vals=[rows[f'{fold}/seed{s}/{name}'][metric] for s in cfg['seeds']];metrics.append(f'{np.mean(vals):.4f} ± {np.std(vals,ddof=1):.4f}')
                if metric=='E100':means[fold][name]=float(np.mean(vals))
            lines.append('| '+' | '.join([fold,name,*metrics])+' |')
        differences=[rows[f'{fold}/seed{s}/observer']['E100']-rows[f'{fold}/seed{s}/global']['E100'] for s in cfg['seeds']]
        paired[fold]=differences
    gates['mechanism']={f:means[f]['observer']<=.9*means[f]['global'] and all(v<0 for v in paired[f]) for f in means}
    gates['structure']={f:means[f]['observer']<=means[f]['history_mlp'] for f in means}
    gates['performance']={f:means[f]['observer']<=(1.1*means[f]['guarded_full'] if f=='temporal' else means[f]['kinematic']) for f in means}
    decision=dict(gates=gates,passed={k:all(v.values()) for k,v in gates.items()},mean_E100=means,observer_minus_global_per_seed=paired)
    b.write_json(OUT/'decision.json',decision)
    lines+=['','## 预设门槛','',str(decision['passed']),'','机制门槛：三个划分均比继续训练的全局模型改善至少10%，每个划分三个种子方向一致。结构门槛：三个划分均不差于相同历史的 MLP。性能门槛：时间划分不超过原完整模型的110%，A/B不差于标定运动学。门槛通过也不等于论文可接收。', '', 'zero/last 为同一权重的测试时干预，不是分别训练的模型。±为种子标准差；两条旧记录反复使用，结果仅属探索性。', '', '模型名称：GPT-6（Codex）。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def jobs(cfg):
    verify(cfg);queue=[(f,s) for f in cfg['splits'] for s in cfg['seeds'] if not (CK/f'{f}_seed{s}'/'COMPLETE.json').exists()];active=[]
    while queue or active:
        while queue and len(active)<3:
            f,s=queue.pop(0);fh=(OUT/f'{f}_seed{s}.log').open('a',encoding='utf-8');p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--fold',f,'--seed',str(s)],cwd=b.ROOT,stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active.append((p,fh,f,s));print('START',f,s,flush=True)
        for item in active[:]:
            p,fh,f,s=item
            if p.poll() is not None:
                fh.close();active.remove(item);print('DONE',f,s,p.returncode,flush=True)
                if p.returncode:
                    for q,h,_,_ in active:q.terminate();q.wait();h.close()
                    raise RuntimeError('Fit failed; inspect logs')
        time.sleep(2)
    evaluate(cfg)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','fit','jobs','report']);ap.add_argument('--fold');ap.add_argument('--seed',type=int);args=ap.parse_args()
    if args.stage=='preflight':preflight()
    else:
        cfg=b.read_json(OUT/'protocol.json')
        if args.stage=='fit':fit(cfg,args.fold,args.seed)
        else:globals()[args.stage](cfg)

