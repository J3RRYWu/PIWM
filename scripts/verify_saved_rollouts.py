"""Spot-recompute saved errors from checkpoints; do not fit or select models."""
from pathlib import Path
import sys,importlib,argparse
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('suite',choices=['geometry','conditioned']);args=ap.parse_args()
    ex=importlib.import_module('run_geometric_frenet' if args.suite=='geometry' else 'run_conditioned_dynamics')
    torch.set_num_threads(4);ex.base.DEVICE='cpu';cfg=ex.base.read_json(ex.OUT/'protocol.json');ex.verify(cfg)
    curves=np.load(ex.OUT/'curves.npz');checks={}
    for f,split in cfg['splits'].items():
        records=ex.base.Records(split['test'])
        for seed in cfg['seeds']:
            d=ex.CK/f'{f}_seed{seed}';old=ex.OLDCK/d.name;ex.encode(records,old)
            stats=dict(np.load((d if args.suite=='conditioned' else old)/'normalizers.npz'))
            limits=ex.base.read_json(d/'limits.json')['limits'];ds=ex.base.WindowData(records,100,4)
            # Spread checks across the source instead of choosing only early windows.
            indices=np.unique(np.linspace(0,len(ds)-1,24,dtype=int));subset=torch.utils.data.Subset(ds,indices)
            for name in ex.VARIANTS:
                m=ex.make(name,stats,old,limits);m.load_state_dict(torch.load(d/(name+'.pt'),weights_only=False)['model'])
                errors=ex.base.evaluate_model(m,subset,batch=24);key=f'{f}/seed{seed}/{name}';saved=curves[key][indices]
                delta=float(np.max(np.abs(errors-saved)));assert np.isfinite(errors).all()
                assert np.allclose(errors,saved,rtol=2e-4,atol=2e-5),(key,delta)
                checks[key]=dict(windows=len(indices),max_absolute_recomputed_difference=delta)
                print(key,delta,flush=True)
    ex.base.write_json(ex.OUT/'checkpoint_spotcheck.json',dict(status='PASS',checks=checks,scope='24 evenly spaced windows per checkpoint; not a full independent rerun',script_sha256=ex.base.digest(__file__)))
if __name__=='__main__':main()
