"""Host checks 1 and 3 of plans/copt_training_refinement.md section 8, without the simulator.

Loads usd_generator.py and config/cmaes_cfg.py by file path, stubbing torch for Isaac Lab's
configclass and the USD conversion of the generator, which need the simulator.
"""

import importlib.util
import os
import pickle
import sys
import tempfile
import types

REPO = "/ws/tron1-rl-isaaclab-cozum"
sys.path.insert(0, REPO)  # design_generator
sys.path.insert(0, "/ws/scratch/pylib")  # cma, numpy, pandas, yaml absent so pickle only

# --- Isaac Lab configclass with torch stubbed --------------------------------------------
sys.modules["torch"] = types.ModuleType("torch"); sys.modules["torch"].Tensor = type("Tensor", (), {})
isaaclab = types.ModuleType("isaaclab"); isaaclab.__path__ = []
utils = types.ModuleType("isaaclab.utils"); utils.__path__ = ["/ws/IsaacLab/source/isaaclab/isaaclab/utils"]
array = types.ModuleType("isaaclab.utils.array"); array.TENSOR_TYPE_CONVERSIONS = {}; array.TENSOR_TYPES = {}
sys.modules.update({"isaaclab": isaaclab, "isaaclab.utils": utils, "isaaclab.utils.array": array})
from isaaclab.utils.configclass import configclass  # noqa: E402
utils.configclass = configclass


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


U = load("usd_generator", f"{REPO}/co_optimisation/co_optimisation/runners/usd_generator.py")
C = load("cmaes_cfg", f"{REPO}/co_optimisation/co_optimisation/config/cmaes_cfg.py")

# The USD conversion needs the simulator, the MJCF path stands in for it.
U.DesignGeneratorBase._generate_individual_usd = lambda self, xml_path, idx: xml_path

failures = []


def check(cond, text):
    print(("PASS " if cond else "FAIL ") + text)
    if not cond:
        failures.append(text)


out = tempfile.mkdtemp(dir="/ws/scratch")

# 1. tolerances reach the strategy only when set
for family, cfg_cls in (("biped", C.BipedCMAESCfg), ("quadruped", C.QuadrupedCMAESCfg)):
    plain = U.CMAESDesignGenerator(family=family, num_individuals=80, output_dir=out)
    o = plain._es.opts
    check((o["tolx"], o["tolfun"], o["tolfunhist"]) == (1e-11, 1e-11, 1e-12),
          f"{family}: no tolerances keeps pycma's defaults {o['tolx'], o['tolfun'], o['tolfunhist']}")
    cfg = cfg_cls()
    gen = U.CMAESDesignGenerator(family=family, num_individuals=80, output_dir=out,
                                 tolx=cfg.tolx, tolfun=cfg.tolfun, tolfunhist=cfg.tolfunhist,
                                 final_population="mean")
    o = gen._es.opts
    check((o["tolx"], o["tolfun"], o["tolfunhist"]) == (cfg.tolx, cfg.tolfun, cfg.tolfunhist),
          f"{family}: configured tolerances reach the strategy {o['tolx'], o['tolfun'], o['tolfunhist']}")
    # A pickled strategy keeps its own options through get_state and load_state
    state = gen.get_state()
    other = U.CMAESDesignGenerator(family=family, num_individuals=80, output_dir=out)
    other.load_state(state)
    check(other._es.opts["tolx"] == cfg.tolx, f"{family}: a restored strategy keeps the options it was made with")

# 2. final_population validation
try:
    U.CMAESDesignGenerator(family="biped", num_individuals=4, output_dir=out, final_population="best")
    check(False, "an unknown final_population raises")
except ValueError:
    check(True, "an unknown final_population raises")

# 3. the terminated branch under "last" and "mean"
for mode in ("last", "mean"):
    gen = U.GrowingDesignDistCMAESDesignGenerator(
        family="quadruped", num_individuals=8, output_dir=os.path.join(out, mode), final_population=mode)
    gen.builder_kwargs = {"shank_foot_gap": 0.001}
    gen.sample_batch()
    gen.generate_population(0)
    gen._terminated = True
    pop = gen.generate_population(1)
    if mode == "last":
        check(pop is None, "terminated under 'last' returns None, as before")
    else:
        values, _ = gen.optimised_design()
        params = pop.get_parameters()
        check(len(params) == 8 and all({k: p[k] for k in values} == values for p in params),
              "terminated under 'mean' carries the distribution mean in all 8 slots")
        check(len(pop.get_link_length_params()) == 8 and len(pop.get_usd_files()) == 8
              and len(set(pop.get_xml_files())) == 1, "the mean population keeps its size from one emitted document")
        files = sorted(os.listdir(os.path.join(out, mode, "gen_0001")))
        check(files == ["individual_0000.xml"], f"one document emitted under gen_0001, found {files}")
        check(all(p["feasible"] for p in params), "the distribution mean emitted feasibly")

# 4. the configuration classes
b, q = C.BipedCMAESCfg(), C.QuadrupedCMAESCfg()
check((b.tolx, b.tolfun, b.tolfunhist, b.finetune_iterations, b.shank_foot_gap) == (0.03, 1.0, 1.0, 400, None),
      "BipedCMAESCfg carries the biped values")
check((q.tolx, q.tolfun, q.tolfunhist, q.finetune_iterations, q.shank_foot_gap) == (0.05, 0.3, 0.3, 400, 0.001),
      "QuadrupedCMAESCfg carries the quadruped values")
base = C.CMAESCfg()
check((base.tolx, base.finetune_iterations, base.num_individuals, base.ea_update_interval, base.ea_late_start)
      == (None, None, 80, 400, 6000), "CMAESCfg carries the launch values and no termination")
check(sorted(base.param_ranges) == sorted(U.CMAES_PARAM_KEYS), "param_ranges names the four searched parameters")
legacy = C.without_termination(q)
check((legacy.tolx, legacy.tolfun, legacy.tolfunhist, legacy.finetune_iterations) == (None,) * 4
      and legacy.shank_foot_gap == 0.001, "without_termination clears the four fields and keeps the rest")
check(q.tolx == 0.05 and q.finetune_iterations == 400, "without_termination leaves its argument untouched")
restored = pickle.loads(pickle.dumps(q))
check(restored.to_dict() == q.to_dict() and type(restored).__name__ == "QuadrupedCMAESCfg",
      "the pickle restores an equal QuadrupedCMAESCfg")
check(q.param_ranges is not C.QuadrupedCMAESCfg().param_ranges, "param_ranges is not shared between instances")
check(C.CMAES_CFG_BY_FAMILY["biped"] is C.BipedCMAESCfg and C.CMAES_CFG_BY_FAMILY["quadruped"] is C.QuadrupedCMAESCfg,
      "CMAES_CFG_BY_FAMILY maps both families")
# The ranges the configuration names are the ones the generator would otherwise default to.
check(U._typed_ranges(dict(base.param_ranges)) == U._typed_ranges(None), "param_ranges reproduce DEFAULT_PARAM_RANGES")

print("\n" + ("ALL PASSED" if not failures else f"{len(failures)} FAILED"))
sys.exit(1 if failures else 0)
