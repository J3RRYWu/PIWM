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


def _py(venv=".venv"):
    """The project interpreter, on either OS -- see piwm/README.md.

    `venv=".venv-sindy"` selects the CPU-torch + pysindy environment; pysindy
    segfaults in the same process as the CUDA build, so SINDYc must run there.

    Falls back to the interpreter running this script, so the driver works on a
    rented box where the environment is the ambient one and there is no .venv.
    """
    for p in (ROOT / venv / "Scripts" / "python.exe", ROOT / venv / "bin" / "python"):
        if p.exists():
            return str(p)
    return sys.executable


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


def jobs_goku_obs(nfolds=5, epochs_bl=60, snap=5):
    """GokuNet WITH its observation pathway, one per fold, delta=0.

    HANDOFF §0.3: the conference version treats GokuNet and Vid2Param BOTH as
    intrinsic (observation-consuming), but this repo kept only Vid2Param's -- which
    is the sole reason Vid2Param is the strongest baseline. `goku_obs` restores it
    at theta_dim=8, matching Vid2Param's parameter count exactly, so the 5-fold main
    table can compare against a baseline the conference version would recognise.
    """
    return [dict(
        name=f"goku_obs_f{f}", out=f"checkpoints/goku_obs_lane_donkey_f{f}/best.tar",
        argv=["src/train/train_baselines_donkey.py", "--variant", "goku_obs",
              "--K", "32", "--suffix", f"_f{f}", "--batch", "128",
              "--batch_v2p", "64", "--epochs", str(epochs_bl),
              "--snapshot-every", str(snap), "--fold", str(f), "--nfolds", str(nfolds)])
        for f in range(nfolds)]


def jobs_encoder_variants(nfolds=5, lambdas=(1e4,), epochs=25, vae_epochs=30):
    """The conference version's encoder design space, plus the two sequence benchmarks.

    Five families over each fold:
      extrinsic_vae -> extrinsic   the two-stage conference recipe. Stage 1 sees NO
                                   physical target; stage 2 reads a frozen latent.
                                   These are the only jobs here with a dependency.
      intrinsic                    single encoder, latent split physical/visual
      lstm, transformer            the non-physical sequence benchmarks the journal
                                   draft dropped and the conference version had

    `lambdas` sweeps the intrinsic interpretability weight, because the reference
    value (1000) was chosen for a 2-D state and this target is a 10-D curvature
    profile whose reconstruction term is four orders of magnitude larger; one
    smoke-test epoch had recon 41590 against interp 0.82, so the balance has to be
    found rather than assumed.
    """
    js = []
    for f in range(nfolds):
        ff = ["--fold", str(f), "--nfolds", str(nfolds)]
        vae = f"checkpoints/cv/enc_vae_f{f}.tar"
        js.append(dict(
            name=f"vae_f{f}", out=vae,
            argv=["src/train/train_kappa_perception.py", "--arch", "extrinsic_vae",
                  "--epochs", str(vae_epochs), "--save", vae] + ff))
        js.append(dict(
            name=f"extrinsic_f{f}", out=f"checkpoints/cv/enc_extrinsic_f{f}.tar",
            needs=[f"vae_f{f}"],          # stage 2 cannot start before stage 1 exists
            argv=["src/train/train_kappa_perception.py", "--arch", "extrinsic",
                  "--vae-ckpt", vae, "--epochs", str(epochs),
                  "--save", f"checkpoints/cv/enc_extrinsic_f{f}.tar"] + ff))
        for lam in lambdas:
            tag = f"l{int(lam):g}" if len(lambdas) > 1 else ""
            out = f"checkpoints/cv/enc_intrinsic{tag}_f{f}.tar"
            js.append(dict(
                name=f"intrinsic{tag}_f{f}", out=out,
                argv=["src/train/train_kappa_perception.py", "--arch", "intrinsic",
                      "--lambda-interp", str(lam), "--epochs", str(epochs),
                      "--save", out] + ff))
        for arch in ("lstm", "transformer"):
            out = f"checkpoints/cv/enc_{arch}_f{f}.tar"
            js.append(dict(
                name=f"{arch}_f{f}", out=out,
                argv=["src/train/train_kappa_perception.py", "--arch", arch,
                      "--epochs", str(epochs), "--save", out] + ff))
    return js


def jobs_kappa_ablation(nfolds=5):
    """Per-fold encoders for the kappa-source ablation (paper Table 2).

    That table is still measured on the single split, from the same checkpoint family
    the main table turned out not to reproduce, so it cannot sit beside 5-fold numbers.
    The `scratch` row already exists per fold as checkpoints/cv/kappa_f{f}.tar; this
    adds the other three encoder variants.

    NOTE the v6-backbone rows inherit a shared encoder trained on the legacy split,
    which saw episodes these folds hold out. That leak favours THEM, so if they still
    lose to scratch the ablation's conclusion is conservative -- but it must be stated.
    """
    v6 = "checkpoints/piwm_lane_v6_donkey/ae.tar"
    variants = [("finetune", ["--mode", "finetune", "--backbone", v6]),
                ("frozen",   ["--mode", "frozen",   "--backbone", v6]),
                ("vqformer", ["--mode", "finetune", "--backbone", "scratch",
                              "--arch", "vqformer", "--epochs", "100"])]
    return [dict(
        name=f"kappa_{tag}_f{f}", out=f"checkpoints/cv/kappa_{tag}_f{f}.tar",
        argv=["src/train/train_kappa_perception.py", "--target", "kappa",
              "--save", f"checkpoints/cv/kappa_{tag}_f{f}.tar",
              "--fold", str(f), "--nfolds", str(nfolds)] + extra)
        for f in range(nfolds) for tag, extra in variants]


def jobs_sindyc(nfolds=5, deltas=(0.0,)):
    """The FOURTH baseline, which the k-fold sweep so far has been missing.

    The paper compares against DVBF, GokuNet, Vid2Param and SindyC, but the 5-fold
    run covered only the first three -- SindyC's curve was still the legacy-split
    cache. It runs in .venv-sindy (pysindy segfaults beside the CUDA torch build) and
    each fit pickles ~133 MB, so this is deliberately kept to delta=0 and few workers.
    """
    return [dict(
        name=f"sindyc_f{f}", out=f"checkpoints/sindyc_lane_donkey_f{f}/model.pkl",
        venv=".venv-sindy",
        argv=["src/train/train_baselines_donkey.py", "--variant", "sindyc",
              "--K", "32", "--suffix", f"_f{f}",
              "--fold", str(f), "--nfolds", str(nfolds)])
        for f in range(nfolds) for d in deltas]


def jobs_gauss(nfolds=5, delta=0.10):
    """Matched-magnitude Gaussian label noise, one per fold, as the control for the
    delta result: the 5-fold run found label noise makes our model monotonically
    better, and this says whether that needs the conference's biased-uniform weak
    supervision or is ordinary regularisation. Pairs with dyn_f{f}_d10."""
    return [dict(
        name=f"dyn_gauss_f{f}", out=f"checkpoints/cv/dyn_gauss_f{f}.tar",
        argv=["src/train/train_frenet.py", "--K", "16",
              "--save", f"checkpoints/cv/dyn_gauss_f{f}.tar",
              "--delta", str(delta), "--noise_kind", "gauss",
              "--fold", str(f), "--nfolds", str(nfolds)])
        for f in range(nfolds)]


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


def _spawn(job, threads, logf, nice=True):
    """Launch one job thread-capped, and de-prioritised unless told otherwise.

    De-prioritising is right on a workstation someone is using and pointless on a
    box rented to run exactly this, hence the switch.
    """
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
               TORCH_NUM_THREADS=str(threads))
    kw = {}
    if nice:
        if os.name == "nt":
            kw["creationflags"] = 0x00004000      # BELOW_NORMAL_PRIORITY_CLASS
        else:
            kw["preexec_fn"] = lambda: os.nice(10)
    return subprocess.Popen([_py(job.get("venv", ".venv")), "-u"] + job["argv"],
                            cwd=str(ROOT), env=env,
                            stdout=logf, stderr=subprocess.STDOUT, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset",
                    choices=["5fold", "seeds", "goku_obs", "gauss", "sindyc",
                             "kappa_ablation", "encoder_variants"],
                    default="5fold")
    ap.add_argument("--no-nice", dest="no_nice", action="store_true",
                    help="run children at normal priority. Use on a dedicated or "
                         "rented machine; leave off on a workstation you are using.")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1e4],
                    help="encoder_variants: intrinsic interpretability weights to sweep")
    ap.add_argument("--enc-epochs", dest="enc_epochs", type=int, default=25)
    ap.add_argument("--vae-epochs", dest="vae_epochs", type=int, default=30)
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

    if a.preset == "5fold":
        jobs = jobs_5fold(a.nfolds, epochs_bl=a.epochs_bl, snap=a.snap)
    elif a.preset == "goku_obs":
        jobs = jobs_goku_obs(a.nfolds, epochs_bl=a.epochs_bl, snap=a.snap)
    elif a.preset == "gauss":
        jobs = jobs_gauss(a.nfolds)
    elif a.preset == "sindyc":
        jobs = jobs_sindyc(a.nfolds)
    elif a.preset == "kappa_ablation":
        jobs = jobs_kappa_ablation(a.nfolds)
    elif a.preset == "encoder_variants":
        jobs = jobs_encoder_variants(a.nfolds, lambdas=tuple(a.lambdas),
                                     epochs=a.enc_epochs, vae_epochs=a.vae_epochs)
    else:
        jobs = jobs_seeds(tuple(a.seeds), epochs_bl=a.epochs_bl)
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
    # a job may declare `needs`; those names must have finished OK before it starts.
    # Anything already on disk counts as satisfied, so a resumed run does not stall.
    done_ok = {j["name"] for j in jobs if (ROOT / j["out"]).exists()}
    failed = set()

    def _ready(job):
        return all(n in done_ok for n in job.get("needs", ()))

    def _blocked(job):
        return any(n in failed for n in job.get("needs", ()))

    while queue or running:
        while (len(running) < workers and not stopping["flag"]
               and not STOP_FILE.exists()):
            nxt = next((j for j in queue if _ready(j)), None)
            if nxt is None:
                for j in [j for j in queue if _blocked(j)]:
                    queue.remove(j)
                    print(f"[{time.time()-t0:7.0f}s] SKIP   {j['name']} "
                          f"(depends on a failed job)", flush=True)
                    results.append(dict(name=j["name"], rc=None, ok=False, secs=0))
                break
            queue.remove(nxt)
            job = nxt
            logf = open(LOG_DIR / f"{job['name']}.log", "w", encoding="utf-8")
            p = _spawn(job, a.threads, logf, nice=not a.no_nice)
            running.append((job, p, logf, time.time()))
            print(f"[{time.time()-t0:7.0f}s] start  {job['name']} "
                  f"({len(running)}/{workers} busy, {len(queue)} queued)", flush=True)
        for entry in running[:]:
            job, p, logf, ts = entry
            if p.poll() is None:
                continue
            running.remove(entry); logf.close()
            have = (ROOT / job["out"]).exists()
            ok = (p.returncode == 0) and have
            # A non-zero exit WITH the artifact present is neither clean success nor
            # plain failure. cuDNN's RNN teardown on Windows crashes the interpreter
            # (0xC0000409) after training has finished and the checkpoint is written,
            # so calling it a failure is wrong; but a job that died mid-run can also
            # leave an early-epoch checkpoint behind, so calling it success is wrong
            # too. Report it as its own state and let a human look.
            dirty = have and p.returncode != 0
            (done_ok if have else failed).add(job["name"])
            results.append(dict(name=job["name"], rc=p.returncode, ok=ok,
                                dirty=dirty, secs=round(time.time() - ts, 1)))
            tag = "done " if ok else ("DIRTY" if dirty else "FAIL ")
            print(f"[{time.time()-t0:7.0f}s] {tag} {job['name']} "
                  f"({(time.time()-ts)/60:.1f} min, rc={p.returncode})"
                  f"{'' if ok else '  -> ' + str(LOG_DIR / (job['name'] + '.log'))}"
                  f"{'  [artifact written; check the log tail]' if dirty else ''}",
                  flush=True)
        if running or queue:
            time.sleep(2)
        if (stopping["flag"] or STOP_FILE.exists()) and not running:
            break

    bad = [r for r in results if not r["ok"] and not r.get("dirty")]
    dirty = [r for r in results if r.get("dirty")]
    print(f"\nfinished {len(results)}/{len(todo)} in {(time.time()-t0)/60:.1f} min"
          f"  ({len(bad)} failed, {len(dirty)} dirty)")
    for r in bad:
        print(f"  FAILED {r['name']}  rc={r['rc']}  log: {LOG_DIR / (r['name'] + '.log')}")
    for r in dirty:
        print(f"  DIRTY  {r['name']}  rc={r['rc']} but the checkpoint exists -- "
              f"confirm the log reached the end: {LOG_DIR / (r['name'] + '.log')}")
    if queue:
        print(f"  {len(queue)} still queued -- rerun the same command to resume")
    summ = ROOT / "reports" / "matrix" / f"{a.preset}_summary.json"
    summ.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"saved -> {summ}")


if __name__ == "__main__":
    main()
