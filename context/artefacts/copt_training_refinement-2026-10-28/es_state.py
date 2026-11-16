"""Read the pickled pycma strategy out of each checkpoint without torch, and evaluate its stop criteria."""
import sys, os, re, glob, pickle, zipfile, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/ws/scratch/pylib")
import cma  # noqa

class _Stub:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return None
    def __setstate__(self, s): pass

class TorchFree(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch") or module.startswith("collections") and name == "OrderedDict":
            if name == "OrderedDict":
                import collections; return collections.OrderedDict
            return _Stub()
        return super().find_class(module, name)
    def persistent_load(self, pid):
        return None

def load(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        with z.open(name) as f:
            return TorchFree(f).load()

SPAN = {"thigh_length_scale": 0.5, "shank_length_scale": 0.5, "thigh_bend_offset": 0.2, "knee_bend_offset": 0.5}
def main():
  for run in sys.argv[1:]:
      print(f"\n==== {run}")
      paths = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
      for p in paths:
          c = load(p)
          copt = c.get("copt") or {}
          if not copt.get("copt_started"):
              continue
          es = pickle.loads(copt["design_generator"]["es"])
          if es.countiter < 1:
              continue
          sx = es.sigma * es.sigma_vec.scaling * np.sqrt(es.dC)
          spc = np.abs(es.sigma * es.sigma_vec.scaling * es.pc)
          hist = np.array(es.fit.hist)
          cur = np.ptp(es.fit.fit)
          it = int(re.search(r"(\d+)\.pt$", p).group(1))
          print(f"ckpt {it:6d} cma_it {es.countiter:3d} sigma {es.sigma:.3f} "
                f"max sigma*sqrt(C_ii) {sx.max():.4f} max |sigma*pc| {spc.max():.4f} "
                f"cur_range {cur:.3f} hist({len(hist)}) range {np.ptp(hist):.3f} "
                f"terminated {copt['design_generator'].get('terminated')} stop {dict(es.stop())}")

if __name__ == '__main__':
    main()