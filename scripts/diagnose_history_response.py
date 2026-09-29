"""Post-hoc history interventions; no refits or gate changes."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_response_observer as r
b=r.b
MODES=['flat_states','flat_states_actions','permute_old_pairs']


def intervene(h,p,mode,permutation):
    if mode=='flat_states':return h[:,-1:].expand_as(h).clone(),p.clone()
    if mode=='flat_states_actions':return h[:,-1:].expand_as(h).clone(),p[:,-1:].expand_as(p).clone()
    if mode=='permute_old_pairs':return torch.cat([h[:,:-1][:,permutation],h[:,-1:]],1),p[:,permutation]
    raise ValueError(mode)


def main():
    r.configure();cfg=b.read_json(r.OUT/'protocol.json');r.verify(cfg);r.completed(cfg)
    protocol=r.OUT/'history_intervention_protocol.json';output=r.OUT/'history_interventions.json'
    assert not output.exists()
    permutation=np.random.default_rng(19317).permutation(14).tolist()
    spec=dict(status='Post-hoc diagnostic specified after primary test results; no independent confirmation or new training.',
        primary_protocol_sha256=b.digest(r.OUT/'protocol.json'),results_sha256=b.digest(r.OUT/'results.json'),
        source_sha256=b.digest(__file__),models=['observer','history_mlp'],modes=MODES,permutation=permutation,
        interpretation='Current v/r, initial pose, future actions and current history endpoint stay fixed. flat_states retains actual past actions; flat_states_actions repeats last available PAST action, not future action. permute_old_pairs uses one fixed permutation of 14 old state/action pairs and retains current state at end. These create shifted/off-manifold inputs, not fair retrained current-state baselines. No causal sufficiency or parameter identification claims.')
    if protocol.exists():assert b.read_json(protocol)==spec
    else:b.write_json(protocol,spec)
    rows={};arrays={};changes={};audit={};normal=np.load(r.OUT/'curves.npz')
    for fold,split in cfg['splits'].items():
        rec=r.records(split['test']);ds=r.HistoryData(rec,100,4)
        for seed in cfg['seeds']:
            for name in spec['models']:
                m,_=r.make(name,fold,seed);m.load_state_dict(torch.load(r.CK/f'{fold}_seed{seed}'/(name+'.pt'),weights_only=False)['model']);m.eval()
                key=f'{fold}/seed{seed}/{name}'
                for mode in MODES:
                    errors=[];latent_deltas=[];neg=0;count=0
                    with torch.no_grad():
                        for z,a,y,h,p in b.loader(ds,128):
                            hh,pp=intervene(h,p,mode,permutation);assert torch.equal(hh[:,-1],h[:,-1]);assert torch.equal(z[:,3:],hh[:,-1])
                            if mode=='flat_states':assert torch.equal(pp,p)
                            original=m.estimate_response(h,p);changed=m.estimate_response(hh,pp)
                            pred=m.forecast(z,a,changed);assert torch.isfinite(pred).all()
                            errors.append(torch.linalg.vector_norm(pred[:,:,:2]-y[:,:,:2],dim=-1).numpy())
                            latent_deltas.append((changed-original).numpy());neg+=int((pred[:,:,3]<0).sum());count+=pred.shape[0]*pred.shape[1]
                    full=key+'/'+mode;err=np.concatenate(errors);bias=np.concatenate(latent_deltas);arrays[full]=err;rows[full]=b.metrics(err)
                    changes[full]=dict(E100_difference_from_normal_m=float((err[:,-1]-normal[key][:,-1]).mean()),latent_delta_RMSE=np.sqrt(np.mean(bias*bias,axis=0)).tolist(),negative_speed_fraction=neg/count)
                audit[key]=dict(current_history_endpoint_unchanged=True,current_state_unchanged=True,future_actions_unchanged=True)
                print('DONE',key,flush=True)
    np.savez_compressed(r.OUT/'history_intervention_curves.npz',**arrays)
    for key in rows:assert b.metrics(arrays[key])==rows[key]
    b.write_json(output,dict(protocol_sha256=b.digest(protocol),results=rows,changes=changes,audit=audit))
    print('Interventions complete; no fits',flush=True)


if __name__=='__main__':main()
