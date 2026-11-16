"""pycma's own stop() at every checkpoint under candidate tolerances, the exact verdict including the pc term of tolx."""
import sys, os, re, glob, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/ws/scratch/pylib")
from es_state import load  # noqa, the torch free loader
# The values of BipedCMAESCfg and QuadrupedCMAESCfg, plans/copt_training_refinement.md section 3.6
CAND = {"copt_moral": dict(tolx=0.03, tolfunhist=1.0, tolfun=1.0),
        "quadruped_copt": dict(tolx=0.05, tolfunhist=0.3, tolfun=0.3)}
for run in sys.argv[1:]:
    opts = CAND[run.split("/")[0]]
    print(f"\n==== {run} {opts}")
    paths = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
    for p in paths:
        copt = load(p).get("copt") or {}
        if not copt.get("copt_started"):
            continue
        es = pickle.loads(copt["design_generator"]["es"])
        if es.countiter < 1:
            continue
        es.opts.update(opts)
        stop = dict(es.stop(check_in_same_iteration=True))
        it = int(re.search(r"(\d+)\.pt$", p).group(1))
        print(f"  ckpt {it:6d} cma_it {es.countiter:3d} stop {stop}")
