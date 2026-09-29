"""Independent recomputation of diagnostic tables and local geometric identities."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import diagnose_road_branches as diag
ex=diag.ex;b=diag.b
from baselines.consistent_road_geometry import road_geometry, consistent_pose

def main():
    cfg=b.read_json(diag.OUT/'protocol.json');ex.verify(b.read_json(ex.OUT/'protocol.json'))
    for p,h in cfg['sources'].items():assert b.digest(b.ROOT/p)==h
    ex.configure();torch.set_num_threads(2);checks={};overview={}
    for fold,split in cfg['splits'].items():
        for part in cfg['partitions']:
            rec=ex.CorrectedRecords(split[part])
            for seed in cfg['seeds']:
                job=f'{fold}_seed{seed}';dest=diag.OUT/job;d=ex.CK/job;enc=d/'independent'
                mark=b.read_json(dest/'COMPLETE.json');assert mark['protocol_sha256']==b.digest(diag.OUT/'protocol.json')
                for n,h in mark['files'].items():assert b.digest(dest/n)==h
                rows=b.read_json(dest/'summary.json')['rows'];arrays=np.load(dest/'arrays.npz');ws=b.read_json(dest/'windows.json')[part]
                ex.encode(rec,enc);ds=b.WindowData(rec,100,4)
                assert ws==[dict(file=rec.records[e]['name'],t=int(t)) for e,t in ds.idx]
                stats=dict(np.load(enc/'normalizers.npz'));limits=b.read_json(d/'limits.json')['limits'];indices=np.unique(np.linspace(0,len(ds)-1,24,dtype=int))
                for name in diag.MODELS:
                    m=ex.make(name,stats,enc,limits);saved=torch.load(d/(name+'.pt'),weights_only=False);m.load_state_dict(saved['model'])
                    if part=='validation':assert abs(rows[f'{part}/{name}/predicted']['E100']-saved['val_E100'])<1e-6
                    for mode,records in [('predicted',rec),('exact',ex.c.oracle_copy(rec))]:
                        key=f'{part}/{name}/{mode}';err=arrays[key+'/position_error'];assert b.metrics(err)=={k:rows[key][k] for k in b.metrics(err)}
                        subset=torch.utils.data.Subset(b.WindowData(records,100,4),indices)
                        # Original uninstrumented model, independent of diagnostic field wrapper.
                        actual=b.evaluate_model(m,subset,batch=24);delta=float(abs(actual-err[indices]).max())
                        assert np.allclose(actual,err[indices],rtol=2e-4,atol=2e-5),(job,key,delta)
                        checks[job+'/'+key]=dict(windows=len(indices),max_absolute_difference_m=delta)
                        for stage in (['start','midpoint'] if name=='geometry_midpoint' else ['start']):
                            for metric in ['den_guard','den_invalid','g_guard','g_degenerate','preview_outside']:
                                value=arrays[key+'/'+stage+'/'+metric];assert ((value>=0)&(value<=1)).all()
                                assert np.allclose(value*100,np.round(value*100),atol=1e-10)
                                assert abs(float(value.mean())-rows[key]['stages'][stage+'/'+metric])<1e-12
                            assert np.all(arrays[key+'/'+stage+'/den_invalid']<=arrays[key+'/'+stage+'/den_guard'])
                        shape=arrays[f'{part}/covariates/shape_rmse'];curv=arrays[f'{part}/covariates/curvature_rmse'];support=arrays[f'{part}/covariates/action_outside_support']
                        overview[job+'/'+key]=dict(E100=rows[key]['E100'],shape_rmse_mean=float(shape.mean()),curvature_rmse_mean=float(curv.mean()),
                            any_action_outside_train_fraction=float((support>0).mean()),
                            speed_E100=rows[key]['speed_E100'],yaw_rate_E100=rows[key]['yaw_rate_E100'],
                            **rows[key]['stages'])
                print('AUDIT',job,part,flush=True)
    b.write_json(diag.OUT/'audit.json',dict(status='PASS',rows=len(checks),checks=checks,scope='Same auditor, independent uninstrumented checkpoint recomputation, not external replication',script_sha256=b.digest(__file__)))
    b.write_json(diag.OUT/'overview.json',overview)
    local_geometry()

def local_geometry():
    torch.manual_seed(928);n=128;dtype=torch.float64
    qknots=torch.arange(10,dtype=dtype)*.5
    amplitudes=torch.rand(n,1,dtype=dtype)*.15
    points=torch.stack([qknots.expand(n,-1),amplitudes*torch.sin(qknots*1.4)],-1)
    z=torch.zeros(n,5,dtype=dtype);z[:,0]=.1+torch.rand(n,dtype=dtype)*4.2
    z[:,1]=torch.rand(n,dtype=dtype)*.1-.05;z[:,2]=torch.rand(n,dtype=dtype)-.5
    z[:,3]=.2+torch.rand(n,dtype=dtype);z[:,4]=torch.rand(n,dtype=dtype)-.5
    d=ex.CK/'temporal_seed0';enc=d/'independent';stats=dict(np.load(enc/'normalizers.npz'));limits=b.read_json(d/'limits.json')['limits']
    m=ex.make('geometry_midpoint',stats,enc,limits).double()
    # make_model initializes learned final layers at zero; verify before identity test.
    for net in [m.core.dv_net,m.core.dom_net,m.core.res_net]:assert torch.count_nonzero(net.net[-1].weight)==0 and torch.count_nonzero(net.net[-1].bias)==0
    _,theta,g,k=road_geometry(points,z[:,0]);assert (g>.1).all() and (1-z[:,1]*k>.2).all()
    rate=m.field(z,torch.zeros(n,2,dtype=dtype),g,k)
    def readout(state):return consistent_pose(points,state[:,0],state[:,1],state[:,2])[0]
    _,velocity=torch.autograd.functional.jvp(readout,z,rate)
    expected=torch.stack([z[:,3]*torch.cos(theta+z[:,2]),z[:,3]*torch.sin(theta+z[:,2]),z[:,4]],-1)
    identity_error=float(abs(velocity-expected).max());assert identity_error<1e-10
    # Fixed-state partial derivatives, explicitly within the unclamped domain.
    kap=torch.linspace(-2,2,n,dtype=dtype,requires_grad=True);gg=torch.linspace(.5,1.5,n,dtype=dtype,requires_grad=True)
    dd=z[:,1];v=z[:,3];psi=z[:,2];h=1-dd*kap
    qdot=v*psi.cos()/(gg*h);pdot=z[:,4]-kap*v*psi.cos()/h
    dqdk,dqdg=torch.autograd.grad(qdot.sum(),(kap,gg),retain_graph=True)
    dpdk=torch.autograd.grad(pdot.sum(),kap)[0]
    errors=[float((dqdk-v*psi.cos()*dd/(gg*h*h)).abs().max()),float((dqdg+v*psi.cos()/(gg*gg*h)).abs().max()),float((dpdk+v*psi.cos()/(h*h)).abs().max())]
    assert max(errors)<1e-10
    b.write_json(diag.OUT/'local_geometry_checks.json',dict(status='PASS',states=n,dtype='float64',readout_velocity_max_error=identity_error,partial_derivative_max_errors=errors,
        scope='Actual Hermite readout and actual zero-residual GeometricFrenet field in smooth interior, no clamps. Not a global stability theorem or a learned-model timestep convergence test.',script_sha256=b.digest(__file__)))
    print('Local geometry identities PASS',identity_error,errors,flush=True)

if __name__=='__main__':main()
