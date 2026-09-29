"""Independent validation audit of all original and conditioned road encoders."""
from pathlib import Path
import sys,json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
from models.conditioned_road_encoder import ConditionedRoadEncoder
from train_conditioned_perception import verify,OUT,CK

def main():
    torch.set_num_threads(4);cfg=b.read_json(OUT/'protocol.json');verify(cfg);result={}
    for fold,split in cfg['splits'].items():
        tr=b.Records(split['train']);va=b.Records(split['validation'])
        td=b.PerceptionData(tr,2);vd=b.PerceptionData(va,2)
        means=np.stack([td[i][1].numpy() for i in range(len(td))]).mean(0)
        for seed in cfg['seeds']:
            job=f'{fold}_seed{seed}';d=CK/job
            saved=torch.load(d/'perception.pt',map_location=b.DEVICE,weights_only=False)
            assert np.array_equal(means,saved['mean'])
            oldscale=np.load(b.CK/job/'normalizers.npz')['perception_scale']
            assert np.array_equal(oldscale,saved['scale'])
            new=ConditionedRoadEncoder(saved['mean'],saved['scale']).to(b.DEVICE);new.load_state_dict(saved['model'])
            old=b.RoadContextEncoder(predict_shape=True).to(b.DEVICE)
            old.load_state_dict(torch.load(b.CK/job/'perception.pt',map_location=b.DEVICE,weights_only=False)['model'])
            assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())
            row={}
            for label,m in [('original',old),('conditioned',new)]:
                m.eval();ps=[];ys=[];fs=[]
                with torch.no_grad():
                    for x,y in b.loader(vd,64):
                        x=x.to(b.DEVICE);k,sh=m.forward_both(x)
                        ps.append(torch.cat([k,sh.flatten(1)],-1).cpu().numpy());ys.append(y.numpy());fs.append(m.features(x).cpu().numpy())
                p=np.concatenate(ps);y=np.concatenate(ys);f=np.concatenate(fs)
                assert np.isfinite(p).all() and np.isfinite(f).all()
                value=float(np.mean(((p-y)/oldscale)**2))
                if label=='conditioned':assert np.isclose(value,saved['val_loss'],rtol=1e-5)
                row[label]=dict(validation_loss=value,windows=len(p),curvature_output_std=float(p[:,:10].std(0).mean()),
                               feature_std=float(f.std(0).mean()),zero_feature_fraction=float((f==0).mean()),
                               all_zero_feature_rows=int((f==0).all(1).sum()),checkpoint_sha256=b.digest((d if label=='conditioned' else b.CK/job)/'perception.pt'))
            result[job]=row;print(job,row,flush=True)
    b.write_json(OUT/'independent_validation_audit.json',dict(status='PASS',scope='validation only; no test loading',
        checks=['train-only mean recomputed','training scales reproduced','identical parameter counts','all finite validation outputs','saved validation metric independently recomputed'],results=result,script_sha256=b.digest(__file__)))
if __name__=='__main__':main()
