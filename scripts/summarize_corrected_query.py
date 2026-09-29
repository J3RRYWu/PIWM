"""Summarize every corrected-query comparison without selecting a model family."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_corrected_query as ex
b=ex.b

def main():
    cfg=b.read_json(ex.OUT/'protocol.json');ex.verify(cfg)
    for name in ['audit.json','perception_reaudit.json','checkpoint_spotcheck.json','input_reaudit.json']:assert b.read_json(ex.OUT/name)['status']=='PASS'
    data=b.read_json(ex.OUT/'results.json');curves=np.load(ex.OUT/'curves.npz');sub=b.read_json(ex.OUT/'per_source_results.json')['results'];summary={};paired={};segments={};validation={}
    for fold in list(cfg['splits'])+['traj1_64x64','traj2_64x64']:
        rows=data['results'] if fold in cfg['splits'] else sub
        for name in ex.NAMES:
            entry={}
            for metric in ['E25','E50','E100','ADE']:
                v=[rows[f'{fold}/seed{s}/{name}'][metric] for s in cfg['seeds']];entry[metric]=dict(mean=None if None in v else float(np.mean(v)),sd=None if None in v else float(np.std(v,ddof=1)),seeds=v)
            if fold in cfg['splits']:
                entry['max_seed_invalid_chart_fraction']=max(data['diagnostics'][f'{fold}/seed{s}/{name}']['invalid_chart_fraction'] for s in cfg['seeds'])
                entry['monitor_samples']=sum(data['diagnostics'][f'{fold}/seed{s}/{name}']['samples'] for s in cfg['seeds'])
            summary[fold+'/'+name]=entry
        for other in [n for n in ex.NAMES if n!='guarded_full']:
            primary=np.array(summary[fold+'/guarded_full']['E100']['seeds'],float);control=np.array(summary[fold+'/'+other]['E100']['seeds'],float)
            paired[fold+'/'+other]=dict(delta_primary_minus_control_seeds=(primary-control).tolist(),mean_difference_m=float((primary-control).mean()),primary_lower_seeds=int((primary<control).sum()),relative_reduction_from_control=float(1-primary.mean()/control.mean()))
    for fold in cfg['splits']:
        ws=b.read_json(ex.CK/f'{fold}_seed0/evaluation_windows.json');files=sorted({w['file'] for w in ws});avg={n:np.mean(np.stack([curves[f'{fold}/seed{s}/{n}'] for s in cfg['seeds']]),axis=0) for n in ex.NAMES};segments[fold]={}
        for filename in files:
            mask=np.array([w['file']==filename for w in ws]);segments[fold][filename]={n:dict(E100=float(avg[n][mask,100].mean()),ADE=float(avg[n][mask,1:].mean()),windows=int(mask.sum())) for n in ex.NAMES}
        for seed in cfg['seeds']:
            d=ex.CK/f'{fold}_seed{seed}'
            for name in ex.NAMES:
                h=b.read_json(d/(name+'_history.json'));best=min(h,key=lambda r:r['val_E100']);validation[f'{fold}/seed{seed}/{name}']=dict(best_epoch=best['epoch'],best_val_E100=best['val_E100'],last_val_E100=h[-1]['val_E100'],last_train_loss=h[-1]['train_loss'],maximum_preclip_gradient=max(r['max_preclip_gradient'] for r in h))
    # This control's used inputs/loss are unchanged by road-label correction.
    kinematic_checks={}
    oldtemp=np.load(b.ROOT/'reports/temporal_holdout/curves.npz')
    for s in cfg['seeds']:
        key=f'temporal/seed{s}/kinematic';delta=float(abs(curves[key]-oldtemp[key]).max());assert np.allclose(curves[key],oldtemp[key],atol=2e-5,rtol=2e-4);kinematic_checks[key]=delta
    b.write_json(ex.OUT/'evidence_summary.json',dict(protocol_sha256=b.digest(ex.OUT/'protocol.json'),summary=summary,paired=paired,segments=segments,validation=validation,unchanged_kinematic_reproduction_max_error=kinematic_checks,interpretation='Paired seed and segment descriptions; no independent-session confidence intervals or model-family selection. Label refits jointly change geometry, initialization and learned perception.',script_sha256=b.digest(__file__)))
    print('Evidence summary saved; unchanged kinematic reproduction PASS',flush=True)
if __name__=='__main__':main()
