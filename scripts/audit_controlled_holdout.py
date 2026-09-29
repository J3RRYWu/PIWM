"""Validate saved experiment provenance and recompute metrics independently.

Run after run_controlled_holdout.py completes. Reads checkpoints/arrays only;
never trains or selects models using test results.
"""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'controlled_holdout';CK=ROOT/'checkpoints'/'controlled_holdout'
def j(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def audit():
    protocol=j(OUT/'protocol.json');saved=j(OUT/'test_results.json');checks=[]
    assert saved['protocol_sha256']==sha(OUT/'protocol.json')
    for rel,expected in j(OUT/'source_snapshot.json').items():
        assert sha(ROOT/rel)==expected,rel
    checks.append('Execution source matches frozen snapshot')
    for name,meta in protocol['files'].items():
        assert sha(ROOT.parent/'Data_Donkeycar_frenet'/name)==meta['sha256']
    for name,expected in j(OUT/'raw_data_fingerprints.json').items():
        assert sha(ROOT.parent/'Data_Donkeycar'/name)==expected
    checks.append('Prepared and raw data fingerprints unchanged')
    for fold,split in protocol['splits'].items():
        assert not set(split['train'])&set(split['validation'])
        assert not (set(split['train'])|set(split['validation']))&set(split['test'])
        windows=None
        for seed in protocol['seeds']:
            d=CK/f'{fold}_seed{seed}';done=j(d/'TRAINING_COMPLETE.json')
            assert done['protocol_sha256']==saved['protocol_sha256']
            for name,expected in done['checkpoint_sha256'].items():
                assert sha(d/(name+'.pt'))==expected
                ck=torch.load(d/(name+'.pt'),map_location='cpu',weights_only=False)
                hist=j(d/(name+'_history.json'))
                field='val_loss' if name=='perception' else 'val_E100'
                best=min((x for x in hist if x[field] is not None),key=lambda x:x[field])
                assert ck['epoch']==best['epoch'],(fold,seed,name)
                assert abs(ck[field]-best[field])<1e-8
            ws=j(d/'test_diagnostics.json')['windows']
            assert all(w['file'] in split['test'] for w in ws)
            if windows is not None:assert windows==ws
            windows=ws
    checks.append('Disjoint partitions; all selected checkpoints minimize validation metric; identical windows across seeds')
    arrays=np.load(OUT/'test_curves.npz');assert set(arrays.files)==set(saved['results'])
    for key,result in saved['results'].items():
        e=arrays[key];assert e.shape==(result['n'],101),key
        assert np.allclose(e[:,0],0,atol=1e-7),key
        failed=int((~np.isfinite(e).all(1)).sum())
        assert failed==result['nonfinite_windows']
        if failed:
            assert all(result[m] is None for m in ('E25','E50','E100','ADE'))
            continue
        values={'E25':e[:,25].mean(),'E50':e[:,50].mean(),'E100':e[:,100].mean(),'ADE':e[:,1:].mean()}
        for metric,value in values.items():assert abs(float(value)-result[metric])<1e-7,(key,metric)
    checks.append('Nonfinite windows explicitly counted (never discarded); exact zero initial error; every finite metric independently recomputed')
    result={'status':'PASS','checks':checks,'rows':len(saved['results']),'protocol_sha256':saved['protocol_sha256']}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
