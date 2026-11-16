"""Validate arm 14, the hip yaw fixed in the URDF, before any of it is applied.

Section 21.14 of ../../../plans/kscale_gait_shaping.md proposes one URDF of two changed
attributes and nine edits in five files, and every claim it rests upon is checkable without a
simulator.  This program checks them.

It reads the PROPOSED dictionaries out of the plan file itself rather than taking a
transcription of them, in the manner validate_scale.py established, so the plan and this
check cannot drift apart.  It reads the LIVE dictionaries out of the configuration files the
same way, for the same reason.  The articulation is never constructed, isaaclab.sim requiring
the USD runtime, so every structural claim is settled from the URDF and every behavioural one
from the symmetry module, which imports torch alone.

Run inside the GPU container, which has torch and numpy:

  docker exec isaac-lab-base-shreyas-gpu1 /workspace/isaaclab/_isaac_sim/python.sh \
      /workspace/isaaclab/biped/context/artefacts/\
kscale_flat-2026-09-28_to_09-29-clip-ablation/scripts/validate_arm14.py
"""
import ast
import os
import re
import sys
import xml.etree.ElementTree as ET

BIPED = "/workspace/isaaclab/biped"
if not os.path.isdir(BIPED):
    BIPED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

URDF = os.path.join(BIPED, "environments/environments/assets/urdf/solefoot/kscale/kscale.urdf")
ASSET_CFG = os.path.join(BIPED, "environments/environments/assets/config/kscale_identified_cfg.py")
ENV_CFG = os.path.join(
    BIPED, "environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py")
PLAN = os.path.join(BIPED, "plans/kscale_gait_shaping.md")
BRS_URDF = os.path.join(
    BIPED, "environments/environments/assets/urdf/solefoot/SD_BRS1/SD_BRS1_Assembly2.urdf")
BRS_DUMP = "/workspace/isaaclab/logs/rsl_rl/sd_brs1_flat/2026-08-17_04-26-44/data/42/dump.npy"

HIP_YAW = ("left_hip_yaw_03", "right_hip_yaw_03")

_fails, _passes, _skips = [], 0, []


def skip(label, why):
    _skips.append(label)
    print(f"  skip {label} -- {why}")


def check(label, ok, detail=""):
    global _passes
    if ok:
        _passes += 1
        print(f"  ok   {label}")
    else:
        _fails.append(label)
        print(f"  FAIL {label}{(' -- ' + detail) if detail else ''}")


def dict_after(text, key, start=0):
    """The python dict literal following `key` in text, by brace matching."""
    i = text.index(key, start)
    i = text.index("{", i)
    depth, j = 0, i
    while True:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    body = text[i:j + 1]
    # strip comments, which ast cannot read inside a literal
    body = re.sub(r"#[^\n]*", "", body)
    return ast.literal_eval(body)


def name_dict_after(text, key, start=0):
    """Like :func:`dict_after` but for a dict whose VALUES are identifiers rather than

    literals, which is what an actuator group table is. Returns {key: identifier name}.
    """
    i = text.index(key, start)
    i = text.index("{", i)
    depth, j = 0, i
    while True:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    body = re.sub(r"#[^\n]*", "", text[i:j + 1])
    node = ast.parse(body, mode="eval").body
    out = {}
    for k, v in zip(node.keys, node.values):
        out[ast.literal_eval(k)] = v.id if isinstance(v, ast.Name) else ast.dump(v)
    return out


def section(text, heading, nxt):
    return text[text.index(heading):text.index(nxt)]


def resolve(patterns, names):
    """(matched names, keys matching nothing, names matched more than once)."""
    hits = {n: [] for n in names}
    dead = []
    for k in patterns:
        m = [n for n in names if re.fullmatch(k, n)]
        if not m:
            dead.append(k)
        for n in m:
            hits[n].append(k)
    return ({n for n, k in hits.items() if k}, dead,
            {n for n, k in hits.items() if len(k) > 1})


# ---------------------------------------------------------------------------------------
print("A. The URDF, and what a fixed joint costs the articulation")

root = ET.parse(URDF).getroot()
links = [l.get("name") for l in root.findall("link")]
joints = {j.get("name"): j.get("type") for j in root.findall("joint")}
revolute = sorted(n for n, t in joints.items() if t == "revolute")
survivors = sorted(n for n in revolute if n not in HIP_YAW)

check("kscale.urdf carries 13 links", len(links) == 13, f"{len(links)}")
check("kscale.urdf carries 12 revolute joints and no fixed joint",
      len(revolute) == 12 and "fixed" not in joints.values(), f"{len(revolute)}")
check("both hip yaws are revolute today", all(joints.get(n) == "revolute" for n in HIP_YAW))
check("freezing them leaves 10 driven joints", len(survivors) == 10, f"{len(survivors)}")

# the children of the two hip yaws are the bodies an unconditional merge would delete
children = {j.get("name"): j.find("child").get("link") for j in root.findall("joint")}
parents = {j.get("name"): j.find("parent").get("link") for j in root.findall("joint")}
yaw_children = sorted(children[n] for n in HIP_YAW)
check("the hip yaw children are kd_d_301l_6061 and kd_d_301r_6061",
      yaw_children == ["kd_d_301l_6061", "kd_d_301r_6061"], str(yaw_children))
check("neither hip yaw child is a foot link",
      not any("foot" in c for c in yaw_children))
check("the hip yaw parents survive as the merge targets",
      sorted(parents[n] for n in HIP_YAW) == ["kd_d_201r_6061", "rs03"])

print("B. merge_fixed_joints is False, which is why no body is lost")
asset_src = open(ASSET_CFG).read()
check("kscale_identified_cfg sets merge_fixed_joints=False",
      re.search(r"merge_fixed_joints\s*=\s*False", asset_src) is not None)
brs_src = open(os.path.join(BIPED, "environments/environments/assets/config/"
                                   "sd_brs1_identified_cfg.py")).read()
check("sd_brs1_identified_cfg sets merge_fixed_joints=False too",
      re.search(r"merge_fixed_joints\s*=\s*False", brs_src) is not None)

broot = ET.parse(BRS_URDF).getroot()
bjoints = {j.get("name"): j.get("type") for j in broot.findall("joint")}
blinks = [l.get("name") for l in broot.findall("link")]
bfixed = sorted(n for n, t in bjoints.items() if t == "fixed")
check("SD_BRS1 declares exactly two fixed joints, both hip yaws",
      bfixed == ["HipYawL", "HipYawR"], str(bfixed))
check("SD_BRS1 carries 13 links and 10 revolute joints",
      len(blinks) == 13 and sum(1 for t in bjoints.values() if t == "revolute") == 10,
      f"{len(blinks)} links")

try:
    import numpy as np
    if not os.path.exists(BRS_DUMP):
        raise FileNotFoundError(BRS_DUMP)
    d = np.load(BRS_DUMP, allow_pickle=True).item()
    jn = list(d["joint_names"])
    bodies = np.asarray(d["robot_mass"]).shape[-1]
    check("SD_BRS1 dump reports 10 joint_names, no hip yaw among them",
          len(jn) == 10 and not any("Yaw" in n for n in jn), str(jn))
    check("SD_BRS1 dump reports 13 BODIES, so the fixed joint child survives",
          bodies == 13, f"{bodies}")
except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:   # noqa: BLE001
    # The log tree is mounted in the GPU containers and numpy lives there too, so these two
    # checks are skipped on the host rather than failed.
    skip("the two SD_BRS1 dump checks", f"{exc}. Run in a GPU container for them.")

print("C. The symmetry augmentation needs no change")
# The module imports torch and nothing else, so it loads in the GPU containers and not in the
# dev container. Section C is skipped rather than failed where torch is absent, which lets the
# same program run on the host, where /ws/djinn is visible and section G can check it.
sym = None
sys.path.insert(0, os.path.join(BIPED, "environments"))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "symkscale", os.path.join(
            BIPED, "environments/environments/tasks/locomotion/mdp/symmetry/kscale.py"))
    sym = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sym)
except ModuleNotFoundError as exc:                         # noqa: BLE001
    # exec_module binds the partially initialised module before raising, so sym must be
    # cleared explicitly or the guard below passes on a module with no attributes.
    sym = None
    skip("the whole of section C", f"{exc}. Run in a GPU container for these eight checks.")

if sym is not None:
    bp = sym._BODY_PARTNERS
    check("_BODY_PARTNERS covers exactly the URDF's 13 links",
          set(bp) == set(links), str(set(links) ^ set(bp)))
    check("_BODY_PARTNERS is an involution",
          all(bp[bp[k]] == k for k in bp))
    check("no hip yaw CHILD is dropped from the body map, so it needs no edit",
          all(c in bp for c in yaw_children))

    jperm, jsign, perm = sym._build_joint_map(survivors, "cpu")
    check("the 10 joint map is an involution", all(perm[p] == i for i, p in enumerate(perm)))
    flipped = {survivors[i] for i, s in enumerate(jsign.tolist()) if s < 0}
    expect = {n for n in survivors
              if any(k in n for k in ("hip_pitch", "hip_roll", "foot_roll"))}
    check("the flip set is exactly the hip pitch, hip roll and ankle roll pairs",
          flipped == expect, str(flipped ^ expect))
    check("'hip_yaw' stays in _FLIP_JOINT_KINDS and is inert",
          "hip_yaw" in sym._FLIP_JOINT_KINDS
          and not any("hip_yaw" in n for n in survivors))
    bperm = sym._build_body_map(links)
    check("the 13 body map is an involution", all(bperm[p] == i for i, p in enumerate(bperm)))
    try:
        sym._build_joint_map(sorted(set(survivors) | {"left_hip_yaw_03"}), "cpu")
        check("freezing ONE hip yaw raises rather than degrading", False, "no raise")
    except Exception:                                          # noqa: BLE001
        check("freezing ONE hip yaw raises rather than degrading", True)

print("D. The proposed dictionaries, read out of the plan")
plan = open(PLAN).read()
arm14 = section(plan, "### 21.14 Arm 14,", "### 21.15 Arm 15,")

prop_init = dict_after(arm14, "init_state_hip_yaw_fixed = ArticulationCfg.InitialStateCfg(")
prop_scale = dict_after(arm14, "scale={")
prop_clip = dict_after(arm14, "clip={")

live_init = dict_after(asset_src, "joint_pos={")
env_src = open(ENV_CFG).read()
live_scale = dict_after(env_src, "scale={")
live_clip = dict_after(env_src, "clip={")

check("proposed init_state names exactly the 10 survivors",
      set(prop_init) == set(survivors), str(set(prop_init) ^ set(survivors)))
check("every surviving init_state value is byte identical to the live one",
      all(prop_init[k] == live_init[k] for k in prop_init),
      str({k: (prop_init[k], live_init.get(k)) for k in prop_init
           if prop_init[k] != live_init.get(k)}))
check("proposed init_state drops both hip yaw keys",
      not any(n in prop_init for n in HIP_YAW))
check("the plan's spawn height is unchanged at 0.74492",
      "pos=(0.0, 0.0, 0.74492)" in arm14 and "pos=(0.0, 0.0, 0.74492)" in asset_src)

matched, dead, dup = resolve(prop_scale, survivors)
check("proposed scale covers all 10 survivors", matched == set(survivors),
      str(set(survivors) - matched))
check("proposed scale has no key matching nothing", not dead, str(dead))
check("proposed scale matches no joint twice", not dup, str(dup))
check("proposed scale drops the hip yaw key",
      not any(re.fullmatch(k, HIP_YAW[0]) for k in prop_scale))
check("every surviving scale value equals the live one",
      all(prop_scale[k] == live_scale[k] for k in prop_scale if k in live_scale))

matched, dead, dup = resolve(prop_clip, survivors)
check("proposed clip covers all 10 survivors", matched == set(survivors),
      str(set(survivors) - matched))
check("proposed clip has no key matching nothing", not dead, str(dead))
check("proposed clip matches no joint twice", not dup, str(dup))
check("proposed clip drops the hip yaw key",
      not any(re.fullmatch(k, HIP_YAW[0]) for k in prop_clip))
check("every surviving clip bound equals the live arm 10 bound",
      all(tuple(prop_clip[k]) == tuple(live_clip[k]) for k in prop_clip if k in live_clip),
      str({k: (prop_clip[k], live_clip.get(k)) for k in prop_clip
           if k in live_clip and tuple(prop_clip[k]) != tuple(live_clip[k])}))

print("E. The actuator groups and the self collision pairs")
groups = name_dict_after(asset_src, "actuators = {")
check("the live configuration carries 6 actuator groups including hip_yaw",
      len(groups) == 6 and "hip_yaw" in groups, str(sorted(groups)))
prop_groups = name_dict_after(arm14, "actuators_hip_yaw_fixed = {")
check("the proposed configuration carries 5 groups and no hip_yaw",
      len(prop_groups) == 5 and "hip_yaw" not in prop_groups, str(sorted(prop_groups)))
check("the 5 surviving groups name the same configuration objects as the live ones",
      all(prop_groups[k] == groups[k] for k in prop_groups),
      str({k: (prop_groups[k], groups.get(k)) for k in prop_groups
           if prop_groups[k] != groups.get(k)}))
# each surviving group's expression must still resolve
exprs = re.findall(r'joint_names_expr=\["([^"]+)"\]', asset_src)
alive = [e for e in exprs if not re.fullmatch(e, HIP_YAW[0])]
for e in alive:
    check(f"group expression {e!r} still resolves",
          any(re.fullmatch(e, n) for n in survivors))

pairs = ast.literal_eval(
    re.search(r"SELF_COLLISION_FILTERED_PAIRS = (\[.*?\])", asset_src, re.S).group(1))
flat = {n for p in pairs for n in p}
check("every self collision pair link survives the freeze", flat <= set(links),
      str(flat - set(links)))
check("no self collision pair names a hip yaw child",
      not (flat & set(yaw_children)))

print("F. The pose is still closed, and the reward and event terms that must go")
# context/KScale.md section 16.2 states the identity on the SIGNED hip pitch, and the sign
# matters because the left and right hip pitch axes are genuinely anti-parallel, which is why
# mdp/symmetry/kscale.py:76 carries hip_pitch in its flip set. Taking the right leg's
# convention, in which the hip pitch nominal is negative, the three sum to zero. Both legs are
# checked with the left hip pitch negated onto that convention.
for side, sgn in (("right", 1.0), ("left", -1.0)):
    hp = sgn * prop_init[f"{side}_hip_pitch_04"]
    kn = prop_init[f"{side}_knee_04"]
    ap = prop_init[f"{side}_foot_pitch_02"]
    check(f"sagittal closure holds at the surviving {side} nominal pose",
          abs(hp + kn + ap) < 2e-5, f"{hp + kn + ap:+.6f}")
check("both hip yaw nominals are ZERO in the live pose, so the freeze moves no link",
      all(abs(live_init[n]) < 1e-12 for n in HIP_YAW))

named = re.findall(r"(\w+)\s*=\s*Rew(?:Term|ardTerm)\(", env_src)
for term in ("rew_keep_hip_yaw_zero_in_air", "pen_hip_yaw_deviation"):
    check(f"{term} exists today and the plan removes it",
          term in env_src and f"self.rewards.{term} = None" in arm14)
check("reset_hip_yaw_joints exists today and the plan removes it",
      "reset_hip_yaw_joints = EventTerm" in env_src
      and "self.events.reset_hip_yaw_joints = None" in arm14)
check("the gain randomisation group names the hip yaw and the plan drops the entry",
      "hip_joint_stiffness_and_damping" in env_src
      and 'if "hip_yaw" not in n' in arm14)
# Every hip_yaw bearing line of the environment configuration must belong to a construct the
# plan addresses, which is checked by attributing each to the nearest enclosing four space
# assignment rather than by counting. A new construct appearing here fails this check, which
# is the point of it.
EXPECTED_CONSTRUCTS = {
    "joint_pos",                          # the action term's scale and clip
    "hip_joint_stiffness_and_damping",    # the gain randomisation group
    "reset_hip_yaw_joints",               # the reset event
    "rew_keep_hip_yaw_zero_in_air",       # the in air kernel
    "pen_hip_yaw_deviation",              # the deviation penalty
}
# only the part ABOVE the arm 14 block, that block being the arm rather than a
# construct the arm must address
twelve_joint_src = env_src.split("# ARM 14 of plans/kscale_gait_shaping.md")[0]
env_lines = twelve_joint_src.split("\n")
found, orphans = set(), []
for i, line in enumerate(env_lines):
    if "hip_yaw" not in line:
        continue
    owner = None
    for j in range(i, -1, -1):
        m = re.match(r"^    (\w+)\s*[:=]", env_lines[j])
        if m:
            owner = m.group(1)
            break
    if owner in EXPECTED_CONSTRUCTS:
        found.add(owner)
    elif line.lstrip().startswith("#"):
        found.add("comment")
    else:
        orphans.append((i + 1, line.strip()[:60]))
check("every hip_yaw bearing line belongs to a construct the plan addresses",
      not orphans, str(orphans))
check("all five hip_yaw bearing constructs are present and addressed",
      EXPECTED_CONSTRUCTS <= found, str(EXPECTED_CONSTRUCTS - found))

print("G. The APPLIED tree, checked against the plan it came from")

NEW_URDF = os.path.join(
    BIPED, "environments/environments/assets/urdf/solefoot/kscale/kscale_hip_yaw_fixed.urdf")
TASK_MOD = os.path.join(
    BIPED, "environments/environments/tasks/locomotion/robots/kscale_hip_yaw_fixed_env_cfg.py")
REG = os.path.join(BIPED, "environments/environments/tasks/locomotion/robots/__init__.py")
DJINN = "/ws/djinn" if os.path.exists("/ws/djinn") else os.path.join(BIPED, "..", "djinn")

check("the hip yaw fixed URDF exists", os.path.exists(NEW_URDF))
if os.path.exists(NEW_URDF):
    orig_lines = open(URDF).read().split("\n")
    new_lines = open(NEW_URDF).read().split("\n")
    check("it has the same line count as kscale.urdf",
          len(orig_lines) == len(new_lines), f"{len(orig_lines)} vs {len(new_lines)}")
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(orig_lines, new_lines), 1) if a != b]
    check("it differs from kscale.urdf on exactly 2 lines", len(diff) == 2,
          str([i for i, _, _ in diff]))
    check("both differing lines are a hip yaw joint type, revolute becoming fixed",
          all('type="revolute"' in a and 'type="fixed"' in b and "hip_yaw" in a
              for _, a, b in diff))
    nroot = ET.parse(NEW_URDF).getroot()
    nlinks = [l.get("name") for l in nroot.findall("link")]
    njoints = {j.get("name"): j.get("type") for j in nroot.findall("joint")}
    check("it carries the SAME 13 links as kscale.urdf", nlinks == links,
          f"{len(nlinks)}")
    check("it carries 10 revolute joints",
          sum(1 for v in njoints.values() if v == "revolute") == 10)
    check("it carries exactly the two hip yaws as fixed",
          sorted(n for n, v in njoints.items() if v == "fixed") == sorted(HIP_YAW))

# the applied articulation configuration, read back out of the source
check("KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED is defined",
      "KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED = ArticulationCfg(" in asset_src)
check("it is CONSTRUCTED and not derived by replace",
      "KSCALE_IDENTIFIED_CFG.replace(" not in asset_src.split(
          "KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED")[-1])
check("KSCALE_IDENTIFIED_CFG above it is untouched",
      "KSCALE_IDENTIFIED_CFG = ArticulationCfg(" in asset_src
      and asset_src.index("KSCALE_IDENTIFIED_CFG = ArticulationCfg(")
          < asset_src.index("KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED = ArticulationCfg("))
applied_init = dict_after(asset_src, "init_state_hip_yaw_fixed = ArticulationCfg.InitialStateCfg(")
check("the applied init_state equals the plan's", applied_init == prop_init,
      str({k: (applied_init.get(k), prop_init.get(k)) for k in set(applied_init) | set(prop_init)
           if applied_init.get(k) != prop_init.get(k)}))
applied_groups = name_dict_after(asset_src, "actuators_hip_yaw_fixed = {")
check("the applied actuator table equals the plan's", applied_groups == prop_groups,
      str(applied_groups))
tail = asset_src.split("KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED = ArticulationCfg(")[-1]
check("the applied spawn sets merge_fixed_joints=False",
      re.search(r"merge_fixed_joints\s*=\s*False", tail) is not None)
check("the applied spawn names the new URDF",
      "hip_yaw_fixed_urdf_path" in tail)
check("KSCALE_HIP_YAW_ACTUATOR_CFG is retained and unreferenced by the new table",
      "KSCALE_HIP_YAW_ACTUATOR_CFG = IdentifiedActuatorCfg(" in asset_src
      and "KSCALE_HIP_YAW_ACTUATOR_CFG" not in asset_src.split(
          "actuators_hip_yaw_fixed = {")[-1].split("}")[0])

# the applied action term and environment
check("ActionsCfgHipYawFixed is defined", "class ActionsCfgHipYawFixed:" in env_src)
check("KscaleEnvHipYawFixedCfg is defined", "class KscaleEnvHipYawFixedCfg(KscaleEnvCfg):" in env_src)
applied_scale = dict_after(env_src, "scale={", env_src.index("class ActionsCfgHipYawFixed:"))
applied_clip = dict_after(env_src, "clip={", env_src.index("class ActionsCfgHipYawFixed:"))
check("the applied scale equals the plan's", applied_scale == prop_scale, str(applied_scale))
check("the applied clip equals the plan's",
      {k: tuple(v) for k, v in applied_clip.items()} == {k: tuple(v) for k, v in prop_clip.items()},
      str(applied_clip))
envtail = env_src.split("class KscaleEnvHipYawFixedCfg(KscaleEnvCfg):")[-1]
for line in ("self.rewards.rew_keep_hip_yaw_zero_in_air = None",
             "self.rewards.pen_hip_yaw_deviation = None",
             "self.events.reset_hip_yaw_joints = None",
             'if "hip_yaw" not in n'):
    check(f"the applied environment carries {line!r}", line in envtail)
check("the applied environment calls super().__post_init__()",
      "super().__post_init__()" in envtail)
check("ActionsCfg and KscaleEnvCfg above are untouched",
      env_src.index("class ActionsCfg:") < env_src.index("class ActionsCfgHipYawFixed:")
      and env_src.index("class KscaleEnvCfg(") < env_src.index("class KscaleEnvHipYawFixedCfg("))

# the task module and the registrations
check("the task module exists", os.path.exists(TASK_MOD))
if os.path.exists(TASK_MOD):
    task_src = open(TASK_MOD).read()
    for cls in ("KscaleHipYawFixedBaseEnvCfg", "KscaleHipYawFixedBaseEnvCfg_PLAY",
                "KscaleHipYawFixedBlindFlatEnvCfg", "KscaleHipYawFixedBlindFlatEnvCfg_PLAY"):
        check(f"{cls} is defined", f"class {cls}" in task_src)
    check("the task module imports the new articulation",
          "KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED" in task_src)
    check("the task module declares no HIM class",
          not [c for c in re.findall(r"^class (\w+)", task_src, re.M) if "HIM" in c])
reg_src = open(REG).read()
for task in ("Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-v0",
             "Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-Play-v0"):
    check(f"{task} is registered", f'id="{task}"' in reg_src)
check("the registration imports the new module",
      "kscale_hip_yaw_fixed_env_cfg," in reg_src)
check("both new tasks share limx_kscale_runner_cfg",
      reg_src.split("kscale hip yaw fixed")[-1].count(
          '"rsl_rl_cfg_entry_point": limx_kscale_runner_cfg') == 2)
check("the twelve joint registrations are untouched",
      'id="Isaac-Limx-Kscale-Blind-Flat-v0"' in reg_src
      and 'id="Isaac-Limx-Kscale-HIM-Blind-Rough-Play-v0"' in reg_src)

if os.path.exists(DJINN):
    dj = open(DJINN).read()
    check("djinn dispatches the mode in both chains",
          dj.count('elif [[ "$3" == "kscale-hip-yaw-fixed" ]]; then') == 2)
    check("djinn names the training and the play task",
          'task="Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-v0"' in dj
          and 'task="Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-Play-v0"' in dj)
    check("djinn lists the mode in both discoverability echoes",
          dj.count("kscale-hip-yaw-fixed") == 4)
else:
    skip("the djinn dispatch", f"not visible from here, expected at {DJINN}."
                               " Run this program on the host to check it.")

print()
print(f"{_passes} checks passed, {len(_fails)} failed, {len(_skips)} skipped")
for f in _fails:
    print(f"  FAILED: {f}")
for s in _skips:
    print(f"  SKIPPED: {s}")
sys.exit(1 if _fails else 0)
