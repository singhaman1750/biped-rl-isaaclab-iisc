# Co-optimisation Observation and Network Configuration Cleanup

Status, IMPLEMENTED on 2026-09-10, every proposal in sections 5.1 to 5.8 having landed. The static validation of section 9 passed, the runtime steps of that section remain unrun, no simulator being reachable from the implementing session. Section 11 records the outcome together with the four divergences from the plan as written.

Correction, 2026-09-17. The 2026-09-10 implementation was carried out entirely as uncommitted working tree state, no commit having been made between it and `a79a603` of 2026-09-07, and was destroyed in full by the `changelog_creator.py` incident of the morning of 2026-09-17, documented in `../../plans/repo_restoration.md`. The outer workspace documents this section's closing paragraph names, `/ws/ARCHITECTURE.md`, `/ws/CO_OPTIMISATION.md`, and `/ws/context/copt.md`, survived that incident intact, since they live in a separate, untouched git repository, and their surviving text was used as corroborating evidence during re-implementation. Sections 5.1 to 5.8 were re-executed against the reset tree the same day, together with the four divergences of section 11, and the static validation of section 9 was re-run and passed. The runtime steps of section 9 remain unrun, per the same limitation recorded below.

## 1. Scope and purpose

The co-optimisation training path presently hard codes the identity of every tensor that enters and leaves its estimator, so that changing which observation the estimator consumes, or which quantity it is asked to predict, requires editing three files and, in the general case, defining a new module class and a new algorithm class beside it. This document specifies the removal of that coupling, so that the actor input, the estimator input and the estimator regression target are each declared once in the runner configuration and read from there by a single `CoptActorCritic`, a single `CoptPPO` and a single `CoptOnPolicyRunner`.

Four network configurations motivate the work, and they differ from one another only in the composition of two lists. Each takes the current observation together with the estimator output as the actor input, and they differ in what the estimator reads and what it predicts. The first reads the observation history and predicts the morphology. The second reads the observation history together with the privileged dynamic state and predicts the morphology. The third reads the observation history together with the morphology and predicts the privileged dynamic state. The fourth reads the observation history and predicts the morphology together with the base linear velocity. Under the architecture specified here each of the four is one configuration class of five lines, and no module or algorithm code distinguishes them. Only the first is registered as a task by this document, the remaining three being written as configuration classes and left unregistered until the first has produced data that justifies running them.

The document also removes the co-optimisation configurations that preceded these four. The architecture in use before this change is not retained beside them, its migrated form being made equivalent to the third experiment and superseded by it, and the encoder decoder learned model variants are deleted outright rather than retired in place, leaving `CoptEstimator` as the only artefact of that line kept in the tree. What remains after this work is one actor-critic class, one algorithm class, one runner and four configurations.

Backwards compatibility is understood here in the sense the requester has fixed, that existing checkpoints need not remain loadable, that the co-optimisation configurations this document deletes are deliberately not preserved, and that no other experiment may be disturbed. The experiments that must survive untouched are therefore the plain TRON1 SoleFoot task and every task registered against it, the HIMLoco tasks, and the plain and HIM quadruped tasks. Section 7 discharges each in turn. No HIMLoco code is edited, and the integration of HIMLoco into this same runner, which will additionally require a mechanism for declaring the loss, is anticipated in the design of section 4 but not implemented here. Every edit falls inside this repository save one, the workspace launcher `/ws/djinn` carrying two co-optimisation modes that section 5.8 removes, which this document records rather than splits into a second plan.

## 2. What the present implementation does

`CoptActorCritic` at `co_optimisation/co_optimisation/modules/copt_actor_critic.py:44` derives every network dimension by name. The actor width is the sum over `obs_groups["policy"]` plus the width of `obs["predictedPrivilegedObs"]` (`copt_actor_critic.py:63-71`), the estimator input width is the width of `obs["morphologyObs"]` plus the flattened width of `obs["obsHistory"]` (`copt_actor_critic.py:98-110`), and the estimator output width is again that of `obs["predictedPrivilegedObs"]` (`copt_actor_critic.py:103`). The forward path repeats the same names, `_get_estimator_input` concatenating `obs["morphologyObs"]` with the flattened history (`copt_actor_critic.py:132-136`), and `_update_distribution` concatenating the estimator output onto the already normalised actor observation (`copt_actor_critic.py:138-148`).

The regression target is hard coded a second time in the algorithm, `CoptPPO.update` computing the mean squared error against `obs_batch["predictedPrivilegedObs"]` at `co_optimisation/co_optimisation/algorithms/copt_ppo.py:252-254`, with the two learned model variants naming `predictedPrivilegedObs` at `copt_ppo.py:500` and `predictedMorphologyObs` at `copt_ppo.py:719`. A change of target therefore requires an edit inside a two hundred line method that is otherwise a verbatim copy of the base implementation, and the copy exists solely to carry that one line.

The environment configuration repeats the same observation terms across five group classes. `ObservationsCfg` at `environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:143` and `CoptObservationsCfg` at `limx_base_env_cfg.py:360` each define a policy group, a critic group and a commands group, and the co-optimisation class adds a morphology group at `:417`, a duplicate morphology group at `:447`, a privileged dynamics group at `:477` and a history group at `:600`. The policy terms are repeated four times across the file and the critic terms three times, so that a change to any observation term must be made in every copy or the configurations silently diverge.

## 3. Findings from the investigation

The following were established against the sources and each constrains the design. They are stated before the design because three of them contradict the request as literally worded.

### 3.1 A synthetic observation name cannot be declared in the runner configuration

`OnPolicyRunner.__init__` passes the configured mapping through `resolve_obs_groups` at `/ws/rsl_rl/rsl_rl/runners/on_policy_runner.py:45`, and that function raises `ValueError` for any group named in any set that is not a key of the environment observation dictionary (`/ws/rsl_rl/rsl_rl/utils/utils.py:269-275`). The check is applied to every set, not only to `policy` and `critic`, so the additional sets this design introduces are permitted provided they name real environment groups, while an entry `encoderOut` inside `obs_groups["policy"]` would abort the run before the policy is constructed. The estimator output must therefore be added to the mapping inside `CoptActorCritic`, after resolution has taken place, and never in the configuration.

### 3.2 The set names must be distinguished from the tensor name

The request names the runner configuration sets `policy`, `critic`, `encoderIn` and `encoderOut`, and separately requires the loss to read `obs_group['gtEncoderOut']` and the injected tensor to be named `encoderOut`. Two of those cannot both hold, since a set named `encoderOut` listing the environment groups that constitute the regression target would be read for one purpose while the string `encoderOut` inside `obs_groups["policy"]` would be read for another. This document adopts `gtEncoderOut` as the set name, matching the loss requirement, and reserves `encoderOut` for the injected tensor alone. The attribute is `obs_groups` in the plural, following `/ws/rsl_rl/rsl_rl/modules/actor_critic.py:42`.

### 3.3 The estimator output width is that of the target, not of the input

The estimator output is the same shape as the concatenation of `gtEncoderOut`, since that is what the mean squared error compares it against, and it bears no relation to the width of `encoderIn`. In the first experiment the input is the flattened twenty five step history, of order a thousand columns for the SoleFoot term set, and the output is the morphology, of order ten. The two can never be interchanged, so the placeholder that section 4 introduces cannot be seeded from the estimator input. It need not be seeded from anything, existing only during construction and being overwritten on every forward pass before it is read, as section 6 traces.

### 3.4 The normaliser must be sized for the concatenated actor input

`ActorCritic` sizes its empirical normaliser to the actor width computed from `obs_groups["policy"]` (`actor_critic.py:43-46`, `:64`), and the present co-optimisation code normalises the actor observation before concatenating the raw estimator output onto it (`copt_actor_critic.py:139-142`), so the estimator output is not normalised at all. The requirement that it be normalised with the remainder of the actor input is met by presenting a placeholder of the estimator output width to the parent constructor, so that the parent sizes both the actor and the normaliser for the full concatenation, and by injecting the real estimator output into the observation dictionary before `get_actor_obs` is called at every forward pass.

### 3.5 The normaliser update path must be overridden

`PPO.process_env_step` calls `self.policy.update_normalization(obs)` with the raw environment observation at `/ws/rsl_rl/rsl_rl/algorithms/ppo.py:160`, and the inherited implementation calls `get_actor_obs` on it (`actor_critic.py:178-184`). Once `encoderOut` is a member of `obs_groups["policy"]` that call raises `KeyError` unless the estimator output is injected first. This is a hard requirement and not an optional refinement, since the run aborts on the first step without it. The requirement is the same under either of the two designs section 4.1 weighs, so it does not discriminate between them.

### 3.6 Groups and terms set to None are skipped by the observation manager

The observation manager skips any group whose configuration is `None` at `/ws/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:496` and any term whose configuration is `None` at `:536`. A single merged observation class may therefore carry every group and every term that any of the tasks needs, with those belonging to only one task defaulting to `None`, and the calling environment configuration enabling exactly what it requires. Nothing is computed for a disabled group, nothing appears in the observation dictionary for it, and the width of every enabled group is unchanged. This is the mechanism that makes one `ObservationsCfg` possible without altering any task.

### 3.7 The two policy groups and the two critic groups are otherwise identical

`ObservationsCfg.PolicyCfg` (`limx_base_env_cfg.py:147-204`) and `CoptObservationsCfg.PolicyCfg` (`:364-414`) carry the same seven terms with the same noise, clip and scale on every one, and differ only in that the former sets `history_length = 10` with `flatten_history_dim = True` while the latter sets neither. `ObservationsCfg.CriticCfg` (`:266-346`) and `CoptObservationsCfg.CriticCfg` (`:498-597`) carry the same terms in the same order with the same post initialisation, and differ only in that the latter appends `link_lengths` after `heights`. The merge is therefore lossless, the two differences being carried by a group attribute and by one optional term.

### 3.8 The duplicate morphology group is numerically identical to the original

`MorphologyCfg` (`limx_base_env_cfg.py:417-444`) and `PredictedMorphologyCfg` (`:447-474`) declare the same three terms with the same parameters, and differ only in `enable_corruption`. That attribute has effect solely through `if not group_cfg.enable_corruption: term_cfg.noise = None` at `observation_manager.py:547-548`, and none of the three terms declares a noise model, so the attribute is inert and the two groups produce the same tensor. `predictedMorphologyObs` is therefore deleted, its sole reader being `CoptLearnedModelV2PPO`, which section 5.7 deletes with it.

### 3.9 Moving the history flattening into the configuration permutes the estimator input

With `flatten_history_dim` set, the manager flattens each term's buffer separately at `observation_manager.py:424` and concatenates the flattened terms at `:433`, giving a term major layout. The present code instead concatenates the terms first, the group being declared with `flatten_history_dim = False` at `limx_base_env_cfg.py:652`, and flattens the resulting three dimensional tensor at `copt_actor_critic.py:133`, giving a time major layout. The two layouts hold the same numbers in a different order. For a freshly initialised first linear layer the difference is a relabelling of columns and is statistically immaterial, and since existing checkpoints need not remain loadable the difference has no further consequence. It is recorded because it is the one respect in which a co-optimisation run started after this change is not bit identical to one started before it.

### 3.10 The rollout storage tolerates a superset destination but not a superset source

`RolloutStorage` allocates one buffer per key of the observation dictionary presented at construction (`/ws/rsl_rl/rsl_rl/storage/rollout_storage.py:49-51`) and writes each transition with `copy_` at `:84`. Injecting `encoderOut` into the dictionary the runner holds would place a key in the source that the destination lacks, so the injection must be performed on a shallow copy local to the forward pass. `TensorDict.copy` is a non recursive clone that shares the underlying tensors, so the copy costs a dictionary allocation and no data movement. This property should be confirmed once against the installed `tensordict` at implementation time, since the package is present only inside the container and could not be exercised here.

### 3.11 Replacing the actor after construction discards the parent's noise initialisation

Where `state_dependent_std` is set, the parent initialises the standard deviation half of the final layer in place, zeroing its weights and setting its bias from `init_noise_std`, at `actor_critic.py:81-88`. The present implementation replaces `self.actor` wholesale after that initialisation has run (`copt_actor_critic.py:89-95`), so the initialisation is silently discarded. The attribute is False in every configuration in the tree, so nothing observed today depends on it, but the defect bears directly on the choice section 4.1 makes, since one of the two candidate designs must replace the actor and the other need not.

### 3.12 Defects observed in passing

`CoptActorCritic` constructs `self.log_std` at `copt_actor_critic.py:113` while the parent has already constructed `self.std` under the default `noise_std_type` of `scalar` (`actor_critic.py:92-93`), leaving an unused parameter registered with the optimiser. `CoptPPO.compute_returns_design_wise` at `copt_ppo.py:96` implements the per individual advantage normalisation the class docstring advertises, but the runner calls the inherited `compute_returns` at `copt_on_policy_runner.py:402`, so the normalisation is never applied. The export block at `scripts/rsl_rl/play.py:909-914` reads `ppo_runner.alg.actor_critic`, an attribute this version of `rsl_rl` does not define, the policy being held as `alg.policy`. Section 8 records what is repaired and what is deliberately left standing.

## 4. Design

The design rests on one idea, that the estimator output is an actor observation like any other, and that the only thing distinguishing it is that the policy produces it rather than the environment. Once it is presented to the inherited machinery as an observation, every mechanism that already exists for observations applies to it unchanged, the actor width is computed by the parent, the normaliser is sized by the parent, the concatenation is performed by `get_actor_obs`, and the co-optimisation module retains no arithmetic of its own.

### 4.1 Why the estimator output is named rather than assumed

Two designs deliver the required behaviour and the choice between them decides how the module scales. The first, adopted here, appends a reserved name to the module's own copy of `obs_groups["policy"]` and injects a tensor under that name into a shallow copy of the observation before each forward pass. The second declares nothing and instead assumes that where an estimator exists its output is concatenated onto the actor observation before normalisation, computing the actor width as the parent's plus the estimator's and rebuilding the actor and the normaliser at that width.

The second is shorter to write and worse in three respects. It must replace `self.actor` after the parent has constructed and initialised it, which discards the noise initialisation of finding 3.11 and rebuilds a network only to throw the first one away. It leaves the position of the estimator output in the concatenation implicit in the order of a `torch.cat` call rather than declared in a list, so a reader must open the module to learn what the actor sees. And it does not compose, since a second estimator, which the HIMLoco integration will bring, requires a second manual addition to the width and a second hand placed `torch.cat`, whereas under the first design it requires one further reserved name appended to the same list and nothing else. The first design also still needs the normaliser override of finding 3.5, so the second buys no simplification there.

The reserved name is appended by the module rather than written in the configuration only because `resolve_obs_groups` would reject it, per finding 3.1. That is a limitation of the validation in the vendored library and not a property of the design, and were the validation to admit names the policy supplies, the same code would work with the name declared in the configuration and the append removed.

### 4.2 The observation sets

Four sets are declared in the runner configuration and every one of them names environment groups only.

| Set | Meaning | Consumed by |
|---|---|---|
| `policy` | Environment groups forming the actor input, the estimator output being appended by the module | `ActorCritic.get_actor_obs` |
| `critic` | Environment groups forming the critic input | `ActorCritic.get_critic_obs` |
| `encoderIn` | Environment groups forming the estimator input | `CoptActorCritic._get_estimator_input` |
| `gtEncoderOut` | Environment groups forming the estimator regression target, whose total width sets the estimator output width | `CoptActorCritic.__init__` and `CoptPPO.update` |

Every group named in `encoderIn` and in `gtEncoderOut` is asserted to be two dimensional, which is what obliges the history to be flattened by the observation configuration rather than by the module, and which is what permits the encoder decoder estimator to be retired from use in favour of a plain multilayer perceptron.

The four experiments are then distinguished entirely by the last two rows of that table.

| Experiment | `encoderIn` | `gtEncoderOut` |
|---|---|---|
| 1 | `["historyObs"]` | `["morphologyObs"]` |
| 2 | `["historyObs", "privilegedDynamicsObs"]` | `["morphologyObs"]` |
| 3 | `["historyObs", "morphologyObs"]` | `["privilegedDynamicsObs"]` |
| 4 | `["historyObs"]` | `["morphologyObs", "estimatorGT"]` |

The design anticipates the HIMLoco integration named in the request without implementing it. That architecture differs from these four in that its estimator is supervised by a contrastive objective against a next step target rather than by a mean squared error against a concurrent one, so the extension it will require is a declaration of the loss family beside the set names, not a further set. Placing the target width behind `gtEncoderOut` rather than behind a fixed name is what leaves that extension to a configuration attribute.

## 5. The edits

### 5.1 `environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py`

Replace `ObservationsCfg` at line 143 and `CoptObservationsCfg` at line 360 with one merged class. The policy, critic and commands groups are taken from the present `ObservationsCfg` unchanged, so that `SFEnvCfg` and every task registered against it computes exactly what it computes today. The co-optimisation groups are added with a default of `None`, and the one critic term that differs is added as an optional term after `heights`, preserving the concatenation order of the present `CoptObservationsCfg.CriticCfg`.

A module level factory supplies the link length term, so that no `ObservationTermCfg` instance is shared between two groups. Sharing would be unsafe, the manager mutating the term in place when it casts a tuple scale to a tensor at `observation_manager.py:575` and when it clears the noise at `:548`.

```python
def _sf_link_lengths_obs_term() -> ObsTerm:
    """A fresh link length observation term for the four scalable links.

    Returned by a factory rather than held as a module constant, since the
    observation manager mutates term configurations in place.
    """
    return ObsTerm(
        func=mdp.robot_link_lengths,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parent_body_names": [
                "hip_R_thigh_Link", "hip_L_thigh_Link", "knee_R_Link", "knee_L_Link",
            ],
            "child_body_names": [
                "knee_R_Link", "knee_L_Link", "ankle_R_actuator_Link", "ankle_L_actuator_Link",
            ],
        },
        clip=(0.0, 100.0),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the SF MDP, plain and co-optimised alike.

    The groups a plain task needs are enabled by default and carry exactly the
    terms and post initialisation they carried before the merge. The groups only
    a co-optimisation task needs default to None, which the observation manager
    skips at observation_manager.py:496, and are enabled by SFCoptEnvCfg.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ... the seven terms of the present ObservationsCfg.PolicyCfg, unchanged ...
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        # ... the terms of the present ObservationsCfg.CriticCfg, unchanged, ending with heights ...
        # Enabled by SFCoptEnvCfg alone. Declared last so that the concatenation
        # order reproduces CoptObservationsCfg.CriticCfg, where link_lengths
        # followed heights.
        link_lengths: ObsTerm | None = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    @configclass
    class MorphologyCfg(ObsGroup):
        """The design parameters, an estimator target or an estimator input."""
        link_lengths = _sf_link_lengths_obs_term()
        robot_mass = ObsTerm(func=mdp.robot_mass, clip=(0.0, 100.0))
        robot_inertia = ObsTerm(func=mdp.robot_inertia)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedDynamicsCfg(ObsGroup):
        """The dynamic state, formerly predictedPrivilegedObs."""
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_.*")},
        )
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_.*")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HistoryObsCfg(ObsGroup):
        """The rolling actor state history, flattened by the manager.

        Flattened here rather than in the module so that CoptActorCritic may
        assert every estimator input flat and size its first layer from the
        configuration alone. The layout is term major and differs by a column
        permutation from the time major layout the deleted obsHistory group
        produced, per section 3.9.
        """
        # ... the eight terms of the present CoptObservationsCfg.HistoryObsCfg ...
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 25
            self.flatten_history_dim = True

    @configclass
    class EstimatorGTCfg(ObsGroup):
        """The base linear velocity, an estimator target in experiment four."""
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            noise=GaussianNoise(mean=0.0, std=0.00),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    morphologyObs: MorphologyCfg | None = None
    privilegedDynamicsObs: PrivilegedDynamicsCfg | None = None
    historyObs: HistoryObsCfg | None = None
    estimatorGT: EstimatorGTCfg | None = None
```

`SFEnvCfg` at `limx_base_env_cfg.py:1403` keeps `observations: ObservationsCfg = ObservationsCfg()` and needs no other change, every co-optimisation group being disabled by default. `SFCoptEnvCfg` at `:1465` binds the same class and enables what it needs in `__post_init__`, which also records the choice in the run's dumped `params/env.yaml`.

```python
@configclass
class SFCoptEnvCfg(ManagerBasedRLEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    # ... the remaining fields unchanged ...

    def __post_init__(self):
        # ... the present body unchanged ...

        # Co-optimisation observation groups, disabled in ObservationsCfg so that
        # SFEnvCfg is unaffected.
        self.observations.morphologyObs = ObservationsCfg.MorphologyCfg()
        self.observations.privilegedDynamicsObs = ObservationsCfg.PrivilegedDynamicsCfg()
        self.observations.historyObs = ObservationsCfg.HistoryObsCfg()
        self.observations.estimatorGT = ObservationsCfg.EstimatorGTCfg()
        self.observations.critic.link_lengths = _sf_link_lengths_obs_term()
        # The co-optimisation actor state is a single step, not the ten step
        # history the plain task uses, reproducing CoptObservationsCfg.PolicyCfg.
        self.observations.policy.history_length = 0

```

`HIMObservationsCfg` at `:670` and `SFHIMEnvCfg` at `:1434` are untouched, the request excluding HIMLoco from this pass.

### 5.2 `co_optimisation/co_optimisation/modules/copt_actor_critic.py`

`CoptActorCritic` is rewritten and the file then contains nothing else. `CoptLearnedModelActorCritic` at `:166` and `CoptLearnedModelV2ActorCritic` at `:266` are deleted with the rest of the encoder decoder line, per section 5.7, so the question of what they would inherit from the rewritten parent does not arise.

```python
class CoptActorCritic(ActorCritic):
    """Actor-critic whose estimator is configured entirely by observation sets.

    The estimator input is the concatenation of the groups named by
    ``obs_groups["encoderIn"]`` and its output width is that of the groups named
    by ``obs_groups["gtEncoderOut"]``, which are also the regression target
    CoptPPO reads through :meth:`get_estimator_target`. The estimator output is
    presented to the inherited machinery as one further actor observation under
    the reserved key ``encoderOut``, so that the actor width, the empirical
    normaliser and the concatenation are all handled by :class:`ActorCritic`
    without arithmetic of our own, and so that the estimator output is
    normalised together with the rest of the actor input.
    """

    is_recurrent = False
    ENCODER_OUTPUT_KEY = "encoderOut"

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        encoder_cfg: dict,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        # Defaulted to the log parameterisation, which the previous
        # implementation imposed unconditionally at copt_actor_critic.py:113 so
        # that the standard deviation cannot become negative. Obtaining it from
        # the parent rather than after the fact also removes the unused self.std
        # parameter the previous implementation left registered.
        noise_std_type: str = "log",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ):
        for required in ("encoderIn", "gtEncoderOut"):
            assert required in obs_groups, (
                f"CoptActorCritic requires the '{required}' observation set. "
                f"Found sets: {sorted(obs_groups)}"
            )

        num_encoder_in = self._flat_width(obs, obs_groups["encoderIn"], "encoderIn")
        num_encoder_out = self._flat_width(obs, obs_groups["gtEncoderOut"], "gtEncoderOut")

        # Present the estimator output to the parent as one further actor
        # observation, so that the parent sizes the actor and the normaliser for
        # the full concatenation and no network is built only to be replaced.
        # Both the mapping and the observation dictionary are copied, the former
        # because the runner holds the resolved mapping and the latter because
        # the rollout storage is sized from the dictionary the runner holds and
        # must not gain a key the rollout never writes, per section 3.10.
        obs_groups = {name: list(groups) for name, groups in obs_groups.items()}
        obs_groups["policy"] = obs_groups["policy"] + [self.ENCODER_OUTPUT_KEY]
        reference = obs[obs_groups["critic"][0]]
        obs = obs.copy()
        obs[self.ENCODER_OUTPUT_KEY] = reference.new_zeros(
            reference.shape[0], num_encoder_out
        )

        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization,
            critic_obs_normalization,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            noise_std_type,
            state_dependent_std,
            **kwargs,
        )

        self.estimator = MLP(
            num_encoder_in,
            num_encoder_out,
            encoder_cfg.get("hidden_dims", [128, 64, 16]),
            encoder_cfg.get("activation", activation),
        )
        print(f"Estimator MLP: {self.estimator}")

    @staticmethod
    def _flat_width(obs: TensorDict, groups: list[str], set_name: str) -> int:
        width = 0
        for group in groups:
            assert len(obs[group].shape) == 2, (
                f"Observation group '{group}' in set '{set_name}' has shape "
                f"{tuple(obs[group].shape)}. Every estimator observation must be "
                "flat, so a history group must set flatten_history_dim."
            )
            width += obs[group].shape[-1]
        return width

    def _get_estimator_input(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [obs[group] for group in self.obs_groups["encoderIn"]], dim=-1
        )

    def get_estimator_target(self, obs: TensorDict) -> torch.Tensor:
        """The regression target CoptPPO minimises the estimator against."""
        return torch.cat(
            [obs[group] for group in self.obs_groups["gtEncoderOut"]], dim=-1
        )

    def _with_encoder_output(
        self, obs: TensorDict, encoder_output: torch.Tensor
    ) -> TensorDict:
        obs = obs.copy()
        obs[self.ENCODER_OUTPUT_KEY] = encoder_output
        return obs

    def _actor_input(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.estimator(self._get_estimator_input(obs))
        actor_obs = self.get_actor_obs(self._with_encoder_output(obs, encoder_output))
        return self.actor_obs_normalizer(actor_obs), encoder_output

    def act(
        self, obs: TensorDict, **kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_obs, encoder_output = self._actor_input(obs)
        self._update_distribution(actor_obs)
        return self.distribution.sample(), encoder_output

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs, _ = self._actor_input(obs)
        if self.state_dependent_std:
            return self.actor(actor_obs)[..., 0, :]
        return self.actor(actor_obs)

    def update_normalization(self, obs: TensorDict) -> None:
        # The estimator output is part of the actor input and must therefore
        # contribute to the actor statistics. Without this the inherited
        # implementation raises KeyError on 'encoderOut', since PPO passes the
        # raw environment observation at ppo.py:160.
        if self.actor_obs_normalization:
            with torch.no_grad():
                obs = self._with_encoder_output(
                    obs, self.estimator(self._get_estimator_input(obs))
                )
        super().update_normalization(obs)
```

Three consequences of the shape of that class deserve note. The overridden `_update_distribution` of the present implementation disappears, the inherited one at `actor_critic.py:124-146` accepting the flat tensor and already handling both standard deviation parameterisations. The overridden `get_actions_log_prob` at `copt_actor_critic.py:154` disappears, being identical to the inherited one at `actor_critic.py:175`. And `evaluate` was never overridden and remains inherited, the critic set containing no synthetic name.

### 5.3 `co_optimisation/co_optimisation/algorithms/copt_ppo.py`

One line changes in `CoptPPO.update`, at `copt_ppo.py:252-254`.

```python
            # Estimator loss against the configured ground truth set, so that a
            # change of target is a change of obs_groups["gtEncoderOut"] alone.
            model_estimation_loss = torch.nn.functional.mse_loss(
                predicted_model_info,
                self.policy.get_estimator_target(obs_batch).detach(),
            )
```

`CoptLearnedModelPPO` at `copt_ppo.py:351` and `CoptLearnedModelV2PPO` at `:594` are deleted, per section 5.7, which removes the file's two remaining readers of the old target names and with them some four hundred and fifty lines that were a verbatim copy of the base update method save for one line each. `CoptPPO` is left as the only class in the file.

### 5.4 `environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`

The base class gains the mapping it presently leaves at `MISSING` (`/ws/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:159`). This is behaviourally inert, `resolve_obs_groups` already inferring the same mapping and warning twice while doing so (`/ws/rsl_rl/rsl_rl/utils/utils.py:242-250` and `:277-289`), and it makes the base usable as a parent for the co-optimisation classes without their inheriting a sentinel.

```python
@configclass
class SF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # ... unchanged ...
    # States explicitly what resolve_obs_groups already infers for every task
    # registered against this configuration, removing two deprecation warnings
    # and giving the co-optimisation subclasses a concrete parent mapping.
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
    }


@configclass
class SFCoptBaseRunnerCfg(SF_TRON1AFlatPPORunnerCfg):
    """Settings common to the four co-optimisation experiments.

    The policy and algorithm class names are assigned by the training script at
    train.py:232-233 from the COPT policy type and are therefore not restated
    here. The observation sets are left to the leaves, being the only thing
    that distinguishes the experiments.
    """

    max_iterations: int = 45000
    # hidden_dims is now the estimator's hidden layer sequence in full. The
    # previous implementation dropped the final entry at copt_actor_critic.py:104,
    # a convention inherited from the encoder decoder latent width, so the four
    # entry list of the parent is given here as three and the topology is
    # unchanged.
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=19,
        hidden_dims=[1024, 512, 256],
        activation="elu",
        orthogonal_init=False,
    )


@configclass
class SFCoptMorphologyRunnerCfg(SFCoptBaseRunnerCfg):
    """Experiment 1, morphology inferred from proprioceptive history alone."""

    experiment_name: str = "copt_moral"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
        "encoderIn": ["historyObs"],
        "gtEncoderOut": ["morphologyObs"],
    }


@configclass
class SFCoptMorphologyFromDynamicsRunnerCfg(SFCoptBaseRunnerCfg):
    """Experiment 2, morphology inferred from history and privileged dynamics."""

    experiment_name: str = "copt_morphology_from_dynamics"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
        "encoderIn": ["historyObs", "privilegedDynamicsObs"],
        "gtEncoderOut": ["morphologyObs"],
    }


@configclass
class SFCoptDynamicsFromMorphologyRunnerCfg(SFCoptBaseRunnerCfg):
    """Experiment 3, dynamics inferred from history and the true morphology."""

    experiment_name: str = "copt_dynamics_from_morphology"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
        "encoderIn": ["historyObs", "morphologyObs"],
        "gtEncoderOut": ["privilegedDynamicsObs"],
    }


@configclass
class SFCoptMorphologyAndVelocityRunnerCfg(SFCoptBaseRunnerCfg):
    """Experiment 4, morphology and base linear velocity inferred together."""

    experiment_name: str = "copt_moral_base_vel"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
        "encoderIn": ["historyObs"],
        "gtEncoderOut": ["morphologyObs", "estimatorGT"],
    }
```

`SFCoptPPORunnerCfg` at `limx_rsl_rl_ppo_cfg.py:139` and `SFCoptLearnedModelPPORunnerCfg` at `:150` are deleted, together with the `DecoderCfg` import they alone require at `:18`. The first is the configuration of the architecture in use before this change, whose estimator read the morphology together with the history and predicted the dynamics, and whose actor saw the morphology directly as well. Migrated to the new sets it would differ from `SFCoptDynamicsFromMorphologyRunnerCfg` in the policy set alone, and the requester has settled that difference by making the two equivalent, so the third experiment stands in its place and the actor's direct morphology channel goes with it, the estimator's prediction being what the actor now reads. Nothing of the previous configuration survives, which is the intent, since a second configuration differing from an experiment by one list entry would invite exactly the confusion this document exists to remove.

### 5.5 `environments/environments/tasks/locomotion/robots/__init__.py`

Nothing is added to `limx_solefoot_env_cfg.py`. The scenario classes descending from `SFCoptBaseEnvCfg` at `limx_solefoot_env_cfg.py:140` already supply the flat, rough and play leaves the co-optimisation experiments need, and the learned model siblings an earlier draft of this document proposed are unnecessary now that the classes requiring an unflattened history are deleted.

The registry loses seven co-optimisation entries and gains three. The three entries for the previous architecture at `robots/__init__.py:387-415`, the four learned model entries at `:420-458`, and the two runner configuration instances that serve them at `:44` and `:46` are deleted. In their place one runner configuration instance is created for `SFCoptMorphologyRunnerCfg` and bound to `SFCoptBlindFlatEnvCfg`, `SFCoptBlindRoughEnvCfg` and the latter's play variant, following the pattern the deleted entries used.

```python
limx_sf_copt_moral_runner_cfg = SFCoptMorphologyRunnerCfg()

gym.register(
    id="Isaac-Limx-SF-Copt-MoRAL-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_copt_moral_runner_cfg,
    },
)
# and the Rough and Rough-Play siblings, identically shaped
```

The three remaining experiment configurations of section 5.4 are written but left unregistered, per section 1, so that they cost nothing until the first has produced data. Registering one later is the four line addition above and requires no further change anywhere, which is the property the whole design exists to deliver.

### 5.6 `environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py` and `agents/quadruped_rsl_rl_ppo_cfg.py`

The quadruped co-optimisation task shares `CoptActorCritic` with the biped and must therefore be migrated in the same pass or it will fail its constructor assertions. The migration is the same shape as the biped's and smaller, since the quadruped observation classes are not merged in this pass. The history group at `cfg/quadruped/base_env_cfg.py:690` is renamed `historyObs` and set to flatten, no unflattened reader remaining once the learned model classes are gone, `predictedPrivilegedObs` at `:689` is renamed `privilegedDynamicsObs`, `predictedMorphologyObs` at `:688` is deleted on the equality of section 3.8, and `PFQuadrupedCoptPPORunnerCfg` at `agents/quadruped_rsl_rl_ppo_cfg.py:68` gains the two new sets. The quadruped's learned model apparatus goes with the biped's, `PFQuadrupedCoptLearnedModelPPORunnerCfg` at `:85` with its `DecoderCfg` import at `:5`, the instance at `robots/__init__.py:54`, and the registration of `Isaac-Quadruped-Copt-Learned-Rough-v0` at `:698` all being deleted.

The quadruped's sets are given below in the form that preserves its present architecture, which is the shape this document deletes on the biped side. That asymmetry is deliberate and is put to validation in section 10, the instruction to remove every trace of the previous setup being read as governing the SoleFoot configuration whose experiments this document plans, not as authorising a silent redesign of a registered task belonging to another robot's programme.

```python
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "morphologyObs"],
        "critic": ["critic"],
        "encoderIn": ["morphologyObs", "historyObs"],
        "gtEncoderOut": ["privilegedDynamicsObs"],
    }
```

Merging the quadruped observation classes as section 5.1 merges the biped's is deliberately not attempted here. It is a mechanical repetition of the same argument against a second and larger file, and folding it into this change would enlarge the surface over which section 9 must validate without answering any question the four experiments pose.

### 5.7 Retirements

`CoptEstimator` in `co_optimisation/co_optimisation/modules/copt_estimator.py` is retained, unreferenced, as the sole record of the encoder decoder formulation, so that a later experiment may take it up without reconstructing it from the history. Every other member of that line is deleted, being `CoptLearnedModelActorCritic` at `copt_actor_critic.py:166`, `CoptLearnedModelV2ActorCritic` at `:266`, `CoptLearnedModelPPO` at `copt_ppo.py:351` and `CoptLearnedModelV2PPO` at `:594`, together with their exports from `modules/__init__.py:3-4` and `algorithms/__init__.py:2-3`, and the imports of them at `copt_on_policy_runner.py:30-31`.

With no class taking a decoder, the runner's decoder apparatus is dead and is removed with it, being `self.decoder_cfg` at `copt_on_policy_runner.py:135` and the conditional branch it guards at `:507-520`, which collapses to the single unconditional construction the `else` arm already performs. `DecoderCfg` at `utils/wrappers/rsl_rl/rl_mlp_cfg.py:33` is retained alongside `CoptEstimator`, being that class's configuration and meaningless apart from it, and its two importers are removed by sections 5.4 and 5.6.

`predictedMorphologyObs` is deleted from both observation configurations on the equality of section 3.8. `CoptActorCritic.init_weights` at `copt_actor_critic.py:122-130` is unreferenced anywhere in the tree and is deleted with the rewrite.

### 5.8 The entry points and the workspace launcher

Two policy types disappear with the classes they select. `COPT-LEARNED` and `COPT-LEARNED-2` are removed from the guard at `scripts/rsl_rl/train.py:197` and from the branch it protects at `:225-230`, leaving the `else` arm that assigns `CoptActorCritic` and `CoptPPO` as the only co-optimisation path. The corresponding branch at `scripts/rsl_rl/play.py:860-880` is deleted, and the help string at `scripts/rsl_rl/cli_args.py:64` is reduced to the three types that remain.

The workspace launcher `/ws/djinn` carries the same two modes and is the one edit this document makes outside the repository. Its train arm loses the `copt-learned` clause at `djinn:134-136` and the `copt-learned-2` clause at `:137-139`, its play arm loses the `copt-learned` clause at `:213-215`, and the two usage strings at `:173-175` and `:243-245` drop both names. The surviving `copt` clauses at `:131-133` and `:210-212` name `Isaac-Limx-SF-Copt-Rough-v0` and `Isaac-Limx-SF-Copt-Rough-Play-v0`, which section 5.5 deletes, so both are repointed at the MoRAL identifiers that replace them.

```bash
                elif [[ "$3" == "copt" ]]; then
                    task="Isaac-Limx-SF-Copt-MoRAL-Rough-v0"
                    policy_type="COPT"
```

The `quadruped-copt` clauses at `djinn:151-153` and `:227-229` name identifiers section 5.6 retains and are left untouched.

## 6. The data path, end to end

The pipeline the request asks to be validated holds under this design, and the following is the trace that establishes it.

During a rollout, `CoptOnPolicyRunner.learn` calls `self.alg.act(obs)` at `copt_on_policy_runner.py:313`, which calls `self.policy.act(obs)` and unpacks the pair at `copt_ppo.py:117`. The estimator output is computed once inside `act`, concatenated into the actor input through the inherited `get_actor_obs`, and returned so that it is available to the caller, though `CoptPPO.act` discards it, the loss being recomputed from stored observations during the update. The transition records the environment observation unmodified at `copt_ppo.py:126`, so the storage receives exactly the keys it was sized for, per section 3.10. At no point does a forward pass read the placeholder written during construction, every path through the module writing the estimator output into its own copy of the observation before `get_actor_obs` is reached.

`PPO.process_env_step` then calls `update_normalization` with the next observation at `ppo.py:160`, which the override of section 5.2 augments with a fresh estimator output before delegating. This costs one estimator forward pass per environment step when actor normalisation is enabled, and nothing when it is not, which is the case for every co-optimisation configuration in the tree today.

During the update, `CoptPPO.update` draws minibatches at `copt_ppo.py:139-147`, applies symmetry augmentation where configured at `:158-170`, and recomputes the pair with `self.policy.act(obs_batch, ...)` at `:173-175`. The estimator output returned there is the prediction, and the target is read from the same batch through `get_estimator_target`, so prediction and target are drawn from the same transition and the gradient reaches actor, critic and estimator through the single optimiser step at `:298-310`. The symmetry branch at `:271-285` calls `act_inference`, which recomputes the estimator on the augmented batch, so the mirrored action means it compares are produced by the same network the surrogate loss trains.

Two points in that trace warrant a check at implementation. The augmentation function is given the environment observation only, `encoderOut` being injected after augmentation inside `act`, so no symmetry map for it is required, and none of the four experiments enables symmetry in any case. And the estimator target is detached in the loss, so no gradient flows from the estimator objective into the observation pipeline.

## 7. What must not be disturbed, and why it is not

The plain TRON1 SoleFoot task is the first obligation and the merged observation class discharges it by construction. Every group and every term the co-optimisation task adds defaults to `None` and is skipped by the manager, per section 3.6, and the enabled groups carry the terms, the ordering and the post initialisation of the present `ObservationsCfg` unchanged, per section 3.7. `SFEnvCfg` therefore produces the same observation dictionary, the same widths and the same values as before, so the twenty or so tasks registered against it at `robots/__init__.py:112-382` are untouched. The addition of an explicit `obs_groups` to `SF_TRON1AFlatPPORunnerCfg` changes nothing, stating what `resolve_obs_groups` already infers.

The HIMLoco tasks are the second and are untouched outright, `HIMObservationsCfg` and `SFHIMEnvCfg` being left as they stand and no file under `himloco/` being edited. They share `SF_TRON1AFlatPPORunnerCfg` through `limx_sf_him_blind_flat_runner_cfg` at `robots/__init__.py:40`, and the explicit mapping added to it resolves for their environment exactly as the inferred one does, both a `policy` and a `critic` group being present.

The quadruped tasks are the third. The plain and HIM quadruped tasks touch none of the edited symbols. The quadruped co-optimisation task is migrated in section 5.6 and its architecture is preserved in every respect but one, the estimator input being permuted by the move of the history flattening into the configuration, per section 3.9. Since existing checkpoints need not remain loadable, the consequence is confined to a fresh run, where the permutation relabels the columns of a randomly initialised first layer and changes nothing measurable. Its learned model variant is deleted rather than preserved, on the same instruction that deletes the biped's.

The co-optimisation configurations this document removes are, by contrast, not preserved in any sense and that is the point of removing them. Seven registered task identifiers cease to exist, being the three of the previous architecture and the four of the learned model line, and any script, note or dashboard naming one of them will fail to resolve it rather than silently resolving it to something else, which is the failure mode to prefer. What is lost is the ability to reproduce those runs from the tree at this commit, and what is kept against that loss is `CoptEstimator`, the formulation itself, together with the record in `../../context/copt.md` sections 9 and 10 of what the deleted classes did and why.

The one thing that does change for every co-optimisation run, old configuration and new alike, is the normalisation of the estimator output, which was previously excluded and is now included. That is the change the requester asked for and it is recorded here so that a reader comparing a run started after this work against one started before it knows that this, and the history permutation, are the two differences.

## 8. Defects recorded

Repaired by this work. The estimator output was excluded from the actor normalisation, the previous implementation normalising before concatenating at `copt_actor_critic.py:139-142`, and is now included by construction. The normaliser was applied to the three dimensional history inside `_get_estimator_input` at `copt_actor_critic.py:133-134`, where an enabled empirical normaliser sized for the flat actor input would have failed, and that misuse disappears with the flat history. An unused `std` parameter was registered with the optimiser alongside `log_std`, per section 3.12, and is removed by declaring the log parameterisation through the parent. The noise initialisation for a state dependent standard deviation was discarded by the actor rebuild, per section 3.11, and is preserved now that no rebuild occurs, though the attribute is False everywhere in the tree so nothing observed depended upon it.

Left standing deliberately, so that another implementation may opt in. `CoptPPO.compute_returns_design_wise` at `copt_ppo.py:96` remains uncalled, the runner using the inherited `compute_returns` at `copt_on_policy_runner.py:402`, and wiring it would alter the advantage scale of every co-optimisation run and invalidate comparison against the runs recorded in `../../context/copt_ppo_nonstationarity.md`. `EncoderCfg.num_output_dim` remains unread, the estimator output width now deriving from `gtEncoderOut`, and the attribute is retained rather than deleted because the HIMLoco configurations share the class. The broken export block at `scripts/rsl_rl/play.py:909-914`, which reads an `alg.actor_critic` attribute this version of `rsl_rl` does not define, is not repaired here, being unrelated to the observation architecture and reachable only when policy export is requested. The quadruped observation configuration is not merged, per section 5.6.

## 9. Validation

The following is the order in which the work should be checked, each step being cheap and each failing loudly.

Compile every edited module with `python3 -m py_compile`, which catches the configuration class errors that would otherwise appear only after a simulator start.

Instantiate the observation configurations without a simulator and compare their term registers against the present classes, asserting that `ObservationsCfg()` exposes exactly `policy`, `critic` and `commands` as non `None` and that its critic link length term is `None`, and that `SFCoptEnvCfg().observations` additionally exposes `morphologyObs`, `privilegedDynamicsObs`, `historyObs`, `estimatorGT` and the critic link lengths. This establishes sections 3.6 and 3.7 mechanically rather than by reading.

Grep the tree for every name section 5.7 deletes and for the seven retired task identifiers, confirming that no reference survives in the repository or in `/ws/djinn`. A deletion of this size fails most often by leaving an import behind, and the check costs one command.

Launch `Isaac-Limx-SF-Blind-Flat-v0` for a handful of iterations and confirm from the printed observation set resolution at `/ws/rsl_rl/rsl_rl/utils/utils.py:296-302`, and from the printed actor and critic multilayer perceptrons at `actor_critic.py:59` and `:70`, that the widths equal those a run of the same task before the change reports. This is the single most informative check, the plain task being the one the merge could most easily disturb.

Launch `Isaac-Limx-SF-Copt-MoRAL-Rough-v0` for a handful of iterations and confirm four things from the startup output, that the resolved sets list the intended groups, that the actor input width equals the sum of the policy groups plus the total width of `gtEncoderOut`, that the estimator input width equals the total width of `encoderIn`, and that `model_estimation_loss` appears in the logged scalars and is finite after the first update. This is the only new identifier registered, per section 5.5.

Instantiate the three unregistered experiment configurations and assert that their four sets name groups the co-optimisation environment supplies, which is the whole of what can be checked without registering them and is enough to catch a mistyped group name before it is discovered months later.

Run the co-optimisation play path for one of the new identifiers, which exercises `act_inference` and therefore the injection path outside the training loop.

Launch the quadruped co-optimisation identifier for a handful of iterations, which is the only check that section 5.6 has not broken a task this document does not otherwise exercise.

Invoke `djinn start train copt` and `djinn start play copt` far enough to see the resolved task identifier, confirming that section 5.8 has repointed both arms and that neither now names a deleted identifier.

## 10. Decisions for the validation phase

Five points were resolved by judgement rather than by evidence and should be confirmed before implementation.

The set name for the regression target is taken as `gtEncoderOut` rather than `encoderOut`, for the reason of section 3.2. If `encoderOut` is preferred as the set name then the injected tensor requires a different reserved name.

The estimator hidden dimensions are taken from `EncoderCfg.hidden_dims` in full, rather than from all but the last entry as the present implementation does at `copt_actor_critic.py:104`. The configuration in section 5.4 accordingly gives three entries where the parent gives four, reproducing the present topology. The alternative is to keep the truncation and leave the parent's list unchanged, which preserves the arithmetic but retains a convention that no longer has a referent once the encoder decoder is retired.

The quadruped observation classes are migrated but not merged, per section 5.6. Merging them in the same pass is a mechanical repetition of section 5.1 and a larger validation surface.

The quadruped co-optimisation task keeps the observation set shape this document deletes on the biped side, its actor continuing to read the true morphology alongside the estimator's prediction, for the reason section 5.6 gives. The alternative is to give it the shape of one of the four experiments, which would be a redesign of a registered task belonging to another robot's programme and is not something this document should decide silently. If that redesign is wanted, the set to give it should be named here.

The task identifiers are derived from the experiment names rather than dictated, `copt_moral` giving `Isaac-Limx-SF-Copt-MoRAL-{Flat,Rough}[-Play]-v0` throughout sections 5.5 and 5.8. The four experiment names themselves are settled and are `copt_moral`, `copt_morphology_from_dynamics`, `copt_dynamics_from_morphology` and `copt_moral_base_vel`.

## 11. Outcome

The work landed on 2026-09-10 across fourteen source files of this repository and one of the workspace, `/ws/djinn`, together with seven documents. The source files are the three modules and two package exports of `co_optimisation`, its runner, the two agent configuration files, the two environment templates, the task registry, and the three entry point scripts. The documents are `/ws/ARCHITECTURE.md`, `/ws/CO_OPTIMISATION.md`, `/ws/context/copt.md` and its index, `ARCHITECTURE.md` and `plans/README.md` of this repository, and this plan. The net effect on the source is 595 lines added against 1202 removed. Every proposal of sections 5.1 to 5.8 was applied as written save for the four divergences below, and every step of section 9 that does not require a simulator was run and passed.

The static validation is worth recording in its particulars, since it establishes the central claim of section 3.7 mechanically rather than by inspection. The merged `ObservationsCfg` was parsed and its term registers compared against those of the two classes it replaces, taken from `git show HEAD`. The policy group's terms and post initialisation match the plain class exactly. The critic group's terms match the plain class exactly once `link_lengths` is excluded, and match the co-optimisation class exactly once it is included, confirming that the optional term occupies the position the co-optimisation critic gave it. The morphology, dynamics, history and commands groups each match their co-optimisation counterpart term for term. The one intended difference is the history group's `flatten_history_dim`, False before and True after, which is the permutation section 3.9 predicts and the only respect in which the observation is not identical. Separately, every observation set declared by every runner configuration was checked to name only groups its environment supplies, and every module compiled.

Four divergences from the plan as written are recorded here.

The first concerns the quadruped estimator. Section 5.6 provided for the quadruped's observation sets but overlooked that `PFQuadrupedCoptPPORunnerCfg` inherits its `EncoderCfg` from a parent whose `hidden_dims` carries four entries, so that under the new convention of reading that list in full the quadruped estimator would have gained a layer it never had. The configuration now restates the list as three entries, exactly as the SoleFoot base does, and the quadruped estimator topology is preserved.

The second concerns the quadruped history group. Section 5.6 proposed renaming `obsHistory` to `historyObs` and setting it to flatten, and the flatten edit was first applied to the wrong group, the HIM history group sharing the surrounding text. It was caught by an audit that printed every `flatten_history_dim` in the file against its owning group and class, and both groups now carry the value they should, the co-optimisation history flattened and the HIM history not. The audit is worth repeating on any file where two configurations declare groups of the same name.

The third concerns `CoptEstimator`. Its class docstring described a decoder regressing `predictedPrivilegedObs`, a group this work renames, and it now names `privilegedDynamicsObs` and states plainly that the class is retained unreferenced as a record rather than used. Nothing in the tree imports it but `modules/__init__.py`.

The fourth is a small addition. Section 5.2 did not say what became of the module's imports once the encoder decoder classes left, and `torch.nn` and `numpy` proved to be needed by none of the surviving code. Both were removed.

Two things the plan promised are not yet done, and both require a simulator. The runtime steps of section 9 remain unrun, being the launches of the plain SoleFoot task, the MoRAL co-optimisation task, the co-optimisation play path, the quadruped co-optimisation task, and the two launcher arms. Until those are run, the claim that the plain task is unaffected rests on the structural comparison above, which is strong evidence and not proof, since it establishes that the configuration is unchanged and not that the observation the manager builds from it is. The first launch of `Isaac-Limx-SF-Blind-Flat-v0` should therefore compare its printed actor and critic widths against a run predating this change before anything else is concluded.
