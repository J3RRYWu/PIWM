"""Run a matrix of independent training jobs in parallel, locally or on a rented box.

WHY
The 9-baseline rerun (2026-08-11) took 5.5 h as a strictly serial bash loop while
7 of 8 cores idled. The 5-fold matrix is ~65 jobs; serial that is 13-15 h. Every
job is independent -- nothing consumes another's checkpoint -- so this is
embarrassingly parallel and the only real constraint is leaving the machine usable.

STAYING USABLE WHILE IT RUNS  (this is the point, not an afterthought)
  * --workers defaults to HALF the logical CPUs' worth of threads, so interactive
    work keeps the rest.
  * every child runs at BELOW_NORMAL priority (Windows) / nice +10 (POSIX), so a
    game or a browser preempts training instead of fighting it.
  * --threads pins each job's BLAS/OMP thread count. Without it torch grabs every
    core per process and N jobs thrash each other into being slower than serial.
  * Ctrl-C once = stop launching new jobs, let the running ones finish cleanly.
    Or `touch reports/matrix/STOP` from anywhere. Re-running resumes: any job
    whose output checkpoint already exists is skipped.

    <py311> scripts/run_matrix.py --preset 5fold --dry-run     # see the plan first
    <py311> scripts/run_matrix.py --preset 5fold               # run it
    <py311> scripts/run_matrix.py --preset 5fold --workers 2   # gentler

Numerical note: train here, but EVALUATE on one machine. Different BLAS builds
shift results in the last digits and a 100-step rollout amplifies that; the tables
are only self-consistent if one evaluator measured every checkpoint.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # piwm/
STOP_FILE = ROOT / "reports" / "matrix" / "STOP"
LOG_DIR = ROOT / "reports" / "matrix" / "logs"

DELTAS = [0.0, 0.05, 0.10]
VARIANTS = ["dvbf", "goku", "v2p"]


def _py():
    """The project interpreter, on either OS -- see piwm/README.md."""
    win = ROOT / ".venv" / "Scripts" / "python.exe"
    nix = ROOT / ".venv" / "bin" / "python"
    return str(win if win.exists() else nix)


def _dtag(d):
    return "" if d == 0 else f"_d{int(round(d * 100))}"


def jobs_5fold(nfolds=5, deltas=DELTAS, variants=VARIANTS, epochs_bl=60, snap=5):
    """The matrix HANDOFF §4 step 3 specifies: ~65 independent trainings.

    One kappa encoder AND one shape encoder per fold (not per delta): the encoder
    is trained on images+map curvature, which the delta label noise does not touch.
    Crucially the encoder for fold f must be trained on fold f, or it has seen the
    dynamics' validation episodes -- that is what `--fold` on both scripts buys.
    """
    js = []
    for f in range(nfolds):
        ff = ["--fold", str(f), "--nfolds", str(nfolds)]
        # --- road perception: one per fold, scratch backbone (HANDOFF §5: best) ---
        for target, out in [("kappa", f"checkpoints/cv/kappa_f{f}.tar"),
                            ("shape", f"checkpoints/cv/shape_f{f}.tar")]:
            js.append(dict(
                name=f"{target}_f{f}", out=out,
                argv=["src/train/train_kappa_perception.py", "--target", target,
                      "--mode", "finetune", "--backbone", "scratch",
                      "--save", out] + ff))
        # --- Frenet dynamics: one per (fold, delta) ---
        for d in deltas:
            out = f"checkpoints/cv/dyn_f{f}{_dtag(d)}.tar"
            js.append(dict(
                name=f"dyn_f{f}{_dtag(d)}", out=out,
                argv=["src/train/train_frenet.py", "--K", "16", "--save", out,
                      "--delta", str(d)] + ff))
        # --- baselines: one per (fold, delta, variant) ---
        # batch 128/64 is NOT tunable here: HANDOFF §8 -- larger batches train these
        # latency-bound baselines WEAKER and silently inflate our margin.
        for d in deltas:
            for v in variants:
                suf = f"_f{f}{_dtag(d)}"
                js.append(dict(
                    name=f"{v}{suf}", out=f"checkpoints/{v}_lane_donkey{suf}/best.tar",
                    argv=["src/train/train_baselines_donkey.py", "--variant", v,
                          "--K", "32", "--suffix", suf, "--batch", "128",
                          "--batch_v2p", "64", "--epochs", str(epochs_bl),
                          "--delta", str(d), "--snapshot-every", str(snap)] + ff))
    return js


def jobs_seeds(seeds=(0, 1, 2, 3, 4), variants=VARIANTS, epochs_bl=60):
    """Extend the single-split repeat study to more seeds (what produced
    reports/repeats/main_table_3seeds.md). Split stays the legacy hold-out, so
    these stay paired with everything measured so far."""
    js = []
    for s in seeds:
        js.append(dict(name=f"dyn_s{s}", out=f"checkpoints/_repeat/dyn_s{s}.tar",
                       argv=["src/train/train_frenet.py", "--K", "16",
                             "--save", f"checkpoints/_repeat/dyn_s{s}.tar", "--seed", str(s)]))
        for target in ("kappa", "shape"):
            out = f"checkpoints/_repeat/{target}_s{s}.tar"
            js.append(dict(name=f"{target}_s{s}", out=out,
                           argv=["src/train/train_kappa_perception.py", "--target", target,
                                 "--mode", "finetune", "--backbone", "scratch",
                                 "--save", out, "--seed", str(s)]))
        for v in variants:
            js.append(dict(name=f"{v}_s{s}", out=f"checkpoints/{v}_lane_donkey_s{s}/best.tar",
                           argv=["src/train/train_baselines_donkey.py", "--variant", v,
                                 "--K", "32", "--suffix", f"_s{s}", "--batch", "128",
                                 "--batch_v2p", "64", "--epochs", str(epochs_bl),
                                 "--seed", str(s)]))
    return js


def _spawn(job, threads, logf):
    """Launch one job de-prioritised and thread-capped."""
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
               TORCH_NUM_THREADS=str(threads))
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x00004000          # BELOW_NORMAL_PRIORITY_CLASS
    else:
        kw["preexec_fn"] = lambda: os.nice(10)
    return subprocess.Popen([_py(), "-u"] + job["argv"], cwd=str(ROOT), env=env,
                            stdout=logf, stderr=subprocess.STDOUT, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["5fold", "seeds"], default="5fold")
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=2,
                    help="BLAS/OMP threads per job; these models are latency-bound "
                         "so more than ~2 buys very little")
    ap.add_argument("--workers", type=int, default=0,
                    help="concurrent jobs; 0 = use half the machine (leaves it usable)")
    ap.add_argument("--epochs-bl", type=int, default=60, dest="epochs_bl")
    ap.add_argument("--snapshot-every", type=int, default=5, dest="snap",
                    help="baseline epoch snapshots, so the checkpoint-selection criterion "
                         "(HANDOFF §0.3) can be changed WITHOUT retraining the matrix. "
                         "~78 MB for the whole 5-fold run; 0 disables.")
    ap.add_argument("--force", action="store_true", help="rerun jobs whose output exists")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ncpu = os.cpu_count() or 4
    workers = a.workers or max(1, (ncpu // 2) // max(1, a.threads))

    jobs = (jobs_5fold(a.nfolds, epochs_bl=a.epochs_bl, snap=a.snap) if a.preset == "5fold"
            else jobs_seeds(tuple(a.seeds), epochs_bl=a.epochs_bl))
    todo = [j for j in jobs if a.force or not (ROOT / j["out"]).exists()]
    done_already = len(jobs) - len(todo)

    print(f"preset={a.preset}  {len(jobs)} jobs, {done_already} already done, "
          f"{len(todo)} to run")
    print(f"machine: {ncpu} logical CPUs -> {workers} workers x {a.threads} threads "
          f"= {workers * a.threads}/{ncpu} occupied, below-normal priority")
    print(f"stop anytime: Ctrl-C once, or create {STOP_FILE}")
    if a.dry_run:
        for j in todo:
            print(f"  {j['name']:<22} -> {j['out']}")
        return
    if not todo:
        print("nothing to do."); return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    stopping = {"flag": False}
    def _on_sigint(sig, frm):
        if stopping["flag"]:
            print("\n! second Ctrl-C -- leaving children running; kill them manually")
            sys.exit(1)
        stopping["flag"] = True
        print("\n! stopping: no new jobs will start, running ones will finish "
              "(Ctrl-C again to bail out)")
    signal.signal(signal.SIGINT, _on_sigint)

    t0 = time.time()
    running, queue, results = [], list(todo), []
    while queue or running:
        while (queue and len(running) < workers
               and not stopping["flag"] and not STOP_FILE.exists()):
            job = queue.pop(0)
            logf = open(LOG_DIR / f"{job['name']}.log", "w", encoding="utf-8")
            p = _spawn(job, a.threads, logf)
            running.append((job, p, logf, time.time()))
            print(f"[{time.time()-t0:7.0f}s] start  {job['name']} "
                  f"({len(running)}/{workers} busy, {len(queue)} queued)", flush=True)
        for entry in running[:]:
            job, p, logf, ts = entry
            if p.poll() is None:
                continue
            running.remove(entry); logf.close()
            ok = (p.returncode == 0) and (ROOT / job["out"]).exists()
            results.append(dict(name=job["name"], rc=p.returncode, ok=ok,
                                secs=round(time.time() - ts, 1)))
            print(f"[{time.time()-t0:7.0f}s] {'done ' if ok else 'FAIL '} {job['name']} "
                  f"({(time.time()-ts)/60:.1f} min, rc={p.returncode})"
                  f"{'' if ok else '  -> ' + str(LOG_DIR / (job['name'] + '.log'))}",
                  flush=True)
        if running or queue:
            time.sleep(2)
        if (stopping["flag"] or STOP_FILE.exists()) and not running:
            break

    bad = [r for r in results if not r["ok"]]
    print(f"\nfinished {len(results)}/{len(todo)} in {(time.time()-t0)/60:.1f} min"
          f"  ({len(bad)} failed)")
    for r in bad:
        print(f"  FAILED {r['name']}  rc={r['rc']}  log: {LOG_DIR / (r['name'] + '.log')}")
    if queue:
        print(f"  {len(queue)} still queued -- rerun the same command to resume")
    summ = ROOT / "reports" / "matrix" / f"{a.preset}_summary.json"
    summ.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"saved -> {summ}")


if __name__ == "__main__":
    main()
