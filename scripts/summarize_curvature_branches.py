"""Report-only serialization adapter; frozen fitting/evaluation sources unchanged."""
import numpy as np
import run_curvature_branches as run
original=run.b.write_json

def write_json(path,obj):
    def native(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [native(x) for x in v]
        return v
    original(path,native(obj))

if __name__=='__main__':
    cfg=run.b.read_json(run.OUT/'protocol.json');run.verify(cfg)
    assert run.b.read_json(run.OUT/'audit.json')['status']=='PASS'
    run.b.write_json=write_json;run.report(cfg)
    write_json(run.OUT/'report_generation.json',dict(status='PASS',script_sha256=run.b.digest(__file__),
        note='Frozen report() computed a NumPy integer segment count that json could not serialize. This report-only adapter converts NumPy scalars to native scalars. No fit, metric, checkpoint, protocol or frozen source changed.'))
    print((run.OUT/'RESULTS.md').read_text(encoding='utf-8'))
