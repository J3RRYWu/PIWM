"""Run remaining independent fold/seed jobs, then open final tests once."""
from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/controlled_holdout';CK=ROOT/'checkpoints/controlled_holdout'

def main():
    cfg=json.loads((OUT/'protocol.json').read_text());queue=[]
    for fold in cfg['splits']:
        for seed in cfg['seeds']:
            if not (CK/f'{fold}_seed{seed}'/'TRAINING_COMPLETE.json').exists():queue.append((fold,seed))
    active=[]
    while queue or active:
        while queue and len(active)<3:
            fold,seed=queue.pop(0);log=(OUT/f'{fold}_seed{seed}.log').open('a',encoding='utf-8')
            cmd=[sys.executable,'-u','scripts/run_controlled_holdout.py','train','--outer',fold,'--seed',str(seed)]
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            active.append((p,log,fold,seed));print('START',fold,seed,'pid',p.pid,flush=True)
        for item in active[:]:
            p,log,fold,seed=item
            if p.poll() is not None:
                log.close();active.remove(item);print('DONE',fold,seed,'exit',p.returncode,flush=True)
                if p.returncode:
                    for q,fh,_,_ in active:q.terminate();fh.close()
                    raise SystemExit(p.returncode)
        time.sleep(2)
    subprocess.run([sys.executable,'scripts/run_controlled_holdout.py','evaluate'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/run_controlled_holdout.py','report'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/audit_controlled_holdout.py'],cwd=ROOT,check=True)

if __name__=='__main__':main()
