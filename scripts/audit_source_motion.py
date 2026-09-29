"""Describe retained sources and rollout-window coverage without model selection."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_holdout as b
cfg=b.read_json(b.OUT/'protocol.json');out={}
for f,split in cfg['splits'].items():
    data=b.Records(split['test']);r=data.records
    signals=np.concatenate([np.column_stack([x['z'][:,3:5],x['actions']]) for x in r])
    summaries={name:dict(zip(['min','p05','median','p95','max'],map(float,np.quantile(signals[:,i],[0,.05,.5,.95,1])))) for i,name in enumerate(['speed_m_s','causal_yaw_rate_rad_s','steering_command','throttle_command'])}
    distances=[]
    for x in r:
        progress=np.unwrap(x['z'][:,0]*2*np.pi/data.L)*data.L/(2*np.pi)
        for t in range(14,len(x['z'])-100,4):distances.append(float(progress[t+100]-progress[t]))
    out[f]=dict(source=split['test_source'],segments=len(r),retained_frames=sum(len(x['z']) for x in r),windows=len(distances),motion=summaries,
                projected_progress_median=float(np.median(distances)),projected_progress_p95=float(np.quantile(distances,.95)),
                fraction_end_progress_outside_0_to_4_5=float(np.mean((np.asarray(distances)<0)|(np.asarray(distances)>4.5))))
b.write_json(b.ROOT/'reports/data_provenance/motion_and_preview.json',dict(scope='descriptive post hoc audit; no additional independent data',results=out,script_sha256=b.digest(__file__)))
print(out)
