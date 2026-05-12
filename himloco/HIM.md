# HIM Loco Documentation

## Hybrid Internal Model Learning

The Hybrid Internal Model (HIM) is a learned internal state estimator for legged locomotion policies. It combines supervised velocity estimation with prototype-based contrastive representation learning to produce a compact latent representation of the robot's proprioceptive history. This latent, together with a predicted base velocity, augments the actor's observation at every step, enabling the policy to implicitly infer unobservable states (e.g., terrain contact, slip, external forces) from raw sensor history alone — without privileged teacher observations at deployment time.

---

### 1. Contrastive Learning

#### What is Contrastive Learning?

Contrastive learning is a family of self-supervised representation learning methods that train an encoder to produce embeddings where semantically similar inputs are close together and dissimilar inputs are far apart in the embedding space — without requiring explicit class labels.

Classical formulations (e.g., SimCLR, MoCo) define positive pairs as different augmented views of the same data point and push them together while pushing negative pairs apart. The key challenge often faced when working with contrastive learning methods is **representation collapse**. Represenation Collapse leads to the degenerate solution space where all inputs to the representation predictor map to the same point.

#### Application in HIM: SwAV-Style Prototype Contrastive Learning

HIM uses a prototype-based variant inspired by **SwAV** (Swapping Assignments between Views, Caron et al. 2020). Instead of directly comparing pairs of embeddings, both views are soft-assigned to a shared set of learnable prototype vectors. The loss then enforces **cross-prediction consistency**: the prototype assignment of one view should be predictable from the representation of the other view.

In the HIM context, the two "views" are:
- **Historical view (`z_s`)**: the encoder's latent from a window of past proprioceptive observations.
- **Current view (`z_t`)**: the target network's latent from the next single-step observation.

The intuition is that the robot's internal state at time `t` should be mutually predictable from both its history up to `t` and its current observation at `t+1`.

#### Mathematical Background

**Notation:**
| Symbol | Description |
|---|---|
| `B` | Batch size |
| `K` | Number of prototype vectors |
| `D` | Latent embedding dimension |
| `z_s ∈ R^(B×D)` | L2-normalized encoder embeddings (from history) |
| `z_t ∈ R^(B×D)` | L2-normalized target network embeddings (from current obs) |
| `C ∈ R^(K×D)` | Prototype matrix (L2-normalized rows) |
| `τ` | Temperature hyperparameter |
| `ε` | Sinkhorn regularization coefficient |

##### Step 1 — Prototype Score Computation

Cosine similarity scores between each embedding and all prototypes:
```
score_s = z_s @ C^T    ∈ R^(B×K)
score_t = z_t @ C^T    ∈ R^(B×K)
```
Because both `z_s`, `z_t`, and `C` are L2-normalized, this is equivalent to cosine similarity, bounded in `[-1, 1]`.

##### Step 2 — Soft Prototype Assignment via Entropic Optimal Transport (Sinkhorn-Knopp)

###### Background: Optimal Transport

**Classical Optimal Transport (Kantorovich formulation)**

Given two discrete probability distributions `μ ∈ Δ^B` (over B samples) and `ν ∈ Δ^K` (over K prototypes), and a cost matrix `C ∈ R^(B×K)`, the optimal transport problem seeks the joint distribution (transport plan) `T ∈ R^(B×K)` that minimizes total transport cost. `T` is called a transport plan because each entry `T_ij` specifies how much of the mass at source point `i` is "shipped" to target point `j`. `T` is essentially a plan for moving probability mass from the source distribution `μ` to the target distribution `ν`, which is why its marginals must recover `μ` and `ν` exactly.

```
OT(μ, ν) = min_{T ≥ 0}  Σ_{ij} T_ij * C_ij
             s.t.  Σ_j T_ij = μ_i   for all i  (row marginals = source distribution)
                   Σ_i T_ij = ν_j   for all j  (column marginals = target distribution)
```
This is a linear program (Kantorovich relaxation of the Monge problem). Its solution is a permutation matrix when both distributions are uniform. In practice, solving this exactly is expensive: O(n^3) with the Hungarian algorithm.

**Entropic Regularization**

Entropic optimal transport (EOT) adds a negative entropy term to regularize the problem, making it strictly convex and yielding a smooth, unique solution:
```
EOT_ε(μ, ν) = min_{T ≥ 0}  Σ_{ij} T_ij * C_ij  -  ε * H(T)
               s.t.  row marginals = μ,  col marginals = ν
```
where `H(T) = -Σ_{ij} T_ij * ln(T_ij)` is the entropy of the transport plan and `ε > 0` is the regularization strength. Higher `ε` → smoother, more uniform plans. Lower `ε` → closer to the original sparse OT solution.

The unique solution to EOT has a closed form called a **scaled Gibbs kernel**:
```
T*_ij = u_i * K_ij * v_j
```
where the Gibbs kernel is `K_ij = exp(-C_ij / ε)`, and `u ∈ R^B`, `v ∈ R^K` are positive scaling vectors (Lagrange multipliers) that enforce the marginal constraints. The **marginal constraints** require that the row sums of T equal the source distribution (`Σ_j T_ij = μ_i` for all i) and the column sums equal the target distribution (`Σ_i T_ij = ν_j` for all j). In other words, the marginals are the distributions you are transporting *from* and *to*, and the constraint ensures the transport plan is a valid coupling of those two distributions.

**Deriving the Closed-Form Solution**

*Step 1 — The Lagrangian.*

To handle the equality constraints, introduce Lagrange multipliers `λ_i` (one per row, i.e. per source point) and `ρ_j` (one per column, i.e. per target point) — see [Appendix A](#appendix-a-lagrange-multipliers) for a self-contained introduction to Lagrange multipliers. Rewriting the entropy term explicitly as `+ε * Σ_{ij} T_ij * ln(T_ij)` (which equals `-ε * H(T)`), the Lagrangian is:
```
L(T, λ, ρ) = Σ_{ij} T_ij * C_ij  +  ε * Σ_{ij} T_ij * ln(T_ij)
             − Σ_i  λ_i * ( Σ_j T_ij − μ_i )
             − Σ_j  ρ_j * ( Σ_i T_ij − ν_j )
```

The three lines are:
1. the original EOT objective
2. penalised row-marginal constraints
3. penalised column-marginal constraints
At optimality the penalty terms vanish (constraints hold with equality), so minimising L over T gives the same solution as the original constrained problem.

*Step 2 — Stationarity condition.*

The entropy term makes L strictly convex in T for any ε > 0, and the interior constraint `T_ij > 0` is automatically satisfied at the minimum (the entropy diverges as any `T_ij → 0`). We can therefore find the unconstrained minimum of L by differentiating with respect to each entry `T_ij` and setting to zero:
```
∂L / ∂T_ij  =  C_ij  +  ε * (ln T_ij + 1)  −  λ_i  −  ρ_j  =  0
```

The `+1` comes from differentiating `T_ij * ln(T_ij)`. Rearranging:
```
ε * ln T*_ij  =  λ_i + ρ_j − C_ij − ε
ln T*_ij      =  (λ_i + ρ_j − C_ij) / ε  −  1
```

*Step 3 — Exponentiating to the closed form.*

Exponentiating both sides:
```
T*_ij  =  exp( (λ_i + ρ_j − C_ij) / ε  −  1 )
        =  exp( (λ_i − ε/2) / ε )  *  exp( −C_ij / ε )  *  exp( (ρ_j − ε/2) / ε )
```

The constant `−1` can be split arbitrarily between `λ_i` and `ρ_j`; the exact split doesn't matter because `u` and `v` are determined by the marginal constraints. Defining:
```
u_i  =  exp( (λ_i − ε/2) / ε )        (absorbs the row Lagrange multiplier)
v_j  =  exp( (ρ_j − ε/2) / ε )        (absorbs the column Lagrange multiplier)
K_ij =  exp( −C_ij / ε )              (Gibbs kernel — fixed, depends only on cost)
```
the optimal transport plan factors as:
```
T*_ij  =  u_i * K_ij * v_j
```

This is the **scaled Gibbs kernel** form. The key insight is that the optimal T* is fully determined by two vectors `u` and `v` of length B and K respectively — the problem has reduced from finding a B×K matrix to finding two vectors.

*Step 4 — Finding u and v from the marginal constraints.*

Substituting `T*_ij = u_i * K_ij * v_j` into the row-marginal constraint `Σ_j T*_ij = μ_i`:
```
Σ_j  u_i * K_ij * v_j  =  μ_i
u_i * (K v)_i           =  μ_i          (matrix–vector product K v, element i)
u_i                      =  μ_i / (K v)_i
```
In vector form: `u = μ ⊘ (K v)`, where `⊘` denotes element-wise division.

Substituting into the column-marginal constraint `Σ_i T*_ij = ν_j`:
```
Σ_i  u_i * K_ij * v_j  =  ν_j
v_j * (K^T u)_j         =  ν_j
v_j                      =  ν_j / (K^T u)_j
```
In vector form: `v = ν ⊘ (K^T u)`.

These two equations are **coupled** — `u` depends on `v` and vice versa. There is no closed-form solution for them in general, but the system has a unique solution (guaranteed by strict convexity), which can be found by the fixed-point iteration described next.

**Sinkhorn Algorithm (Sinkhorn-Knopp)**

Starting from the Gibbs kernel `K = exp(-C / ε)`, the Sinkhorn algorithm iterates:
```
u^(t+1) = μ / (K  * v^(t))      # Update row scaling (enforce row marginals)
v^(t+1) = ν / (K^T * u^(t+1))   # Update column scaling (enforce col marginals)
```

In practice, this is implemented via in-place alternating normalization of the transport matrix, without explicitly tracking u and v:
```
Q = K                           # Initialize from Gibbs kernel
for each iteration:
    Q = diag(μ / row_sums(Q)) * Q    # Normalize rows to marginal μ
    Q = Q * diag(ν / col_sums(Q))    # Normalize columns to marginal ν
```
This converges geometrically (linear convergence rate), and even 3–10 iterations typically suffice in practice.

###### Application in HIM: EOT with Uniform Marginals

In HIM, the "cost" is the **negative prototype score** (higher cosine similarity = lower cost):
```
C_ij = -score_ij = -(z_i · c_j)
```

The desired marginals are **uniform**: `μ_i = 1/B` (each sample contributes equally) and `ν_j = 1/K` (each prototype receives equal total weight). This gives the EOT problem:
```
Q* = argmin_{Q ∈ U(1/B, 1/K)}  Σ_{ij} Q_ij * (-score_ij / ε)  +  Σ_{ij} Q_ij * ln(Q_ij)
```
where `U(1/B, 1/K)` is the transport polytope with uniform marginals. The uniform marginal constraint on prototypes (`ν_j = 1/K`) is the key mechanism that **prevents representation collapse**. The constraint ensures every prototype must receive equal total assignment weight, so the model cannot trivially collapse to a single prototype.

The Gibbs kernel for this problem is `K_ij = exp(score_ij / ε)`, and the Sinkhorn algorithm (in the direct alternating normalization form) finds Q*.

###### Implementation Walkthrough

The actual implementation in `sinkhorn()` (line 132) corresponds to:
```python
# Step 0: Compute Gibbs kernel and transpose to K×B for row-major iteration
Q = exp(score / ε).T        # Shape: [K × B]
K, B = Q.shape

# Step 1: Global normalization (numerical stability)
# Divides all entries by the grand total. Scales Q to be a joint distribution
# summing to 1 over the K×B grid. This does NOT change the final normalized
# result (Sinkhorn is scale-invariant), but prevents fp32 overflow for large scores.
Q /= Q.sum()

# Steps 2-N: Alternating Sinkhorn normalizations
for each iteration:
    # Enforce prototype marginal ν_k = 1/K (row marginals in K×B layout)
    Q /= row_sums(Q)          # Q rows now sum to 1
    Q /= K                    # Q rows now sum to 1/K

    # Enforce sample marginal μ_b = 1/B (column marginals in K×B layout)
    Q /= col_sums(Q)          # Q columns now sum to 1
    Q /= B                    # Q columns now sum to 1/B

# Step N+1: Rescale and transpose back to B×K
# Multiply by B so each sample's row sums to 1 (valid probability distribution over K prototypes)
return (Q * B).T              # Shape: [B × K]
```

> **Important difference from the simplified pseudocode**: The global normalization `Q /= Q.sum()` (line 135) occurs before the iterative loop and is absent from simplified descriptions of the algorithm. It is not part of the theoretical Sinkhorn iteration — it is a practical initialization step for numerical stability. Because the Sinkhorn algorithm normalizes rows and columns alternately, multiplying the initial matrix by any positive scalar does not affect the converged result. However, without this step, `exp(score / 0.05)` can produce very large values (scores near 1.0 give `exp(20) ≈ 5×10^8`), risking fp32 overflow during the iterations.

##### Step 3 — Predicted Probability Distribution

Each embedding also produces a predicted probability over prototypes via softmax with temperature `τ`:
```
p_s = softmax(score_s / τ)
p_t = softmax(score_t / τ)
```
Temperature `τ` (default `3.0`) sharpens or softens the distribution. Higher `τ` → softer distributions, preventing overconfident predictions early in training.

##### Step 4 — Swap Loss

The swap loss is a symmetric cross-entropy between the hard assignment of one view and the predicted distribution of the other:
```
L_swap = -0.5 * E[ q_s · log(p_t) + q_t · log(p_s) ]
       = -0.5 * mean_over_batch[ Σ_k q_s_k * log(p_t_k) + Σ_k q_t_k * log(p_s_k) ]
```
This enforces:
- The encoder history embedding `z_s` should predict the prototype assignment `q_t` of the current observation.
- The target embedding `z_t` should predict the prototype assignment `q_s` of the history.

The "swap" is the cross-prediction: each view predicts the assignment of the **other** view. This is more stable than direct embedding comparison because the Sinkhorn assignments provide a well-distributed, collapse-free target signal.

---

### 2. Hybrid Model Learning Architecture

#### Overview

The HIM system is composed of four main components that extend the standard RSL-RL PPO pipeline:

```
HIMOnPolicyRunner
    └── HIMPPO (algorithm)
            └── HIMActorCritic (policy)
                    ├── HIMEstimator (encoder + target + prototypes)
                    │       ├── encoder:  [T * D_obs] → [3 + D_latent]
                    │       ├── target:   [D_obs]     → [D_latent]
                    │       └── proto:    Embedding[K × D_latent]
                    ├── actor:    [D_obs + 3 + D_latent] → [num_actions]
                    └── critic:   (inherited from ActorCritic)
    └── HIMRolloutStorage (stores current + next observations)
```

#### Networks

**Encoder (`HIMEstimator.encoder`)**

An MLP that processes the flattened observation history:
- Input: `T × D_obs` flattened to `[T * D_obs]` (where `T` = `temporal_steps`, `D_obs` = `num_one_step_obs`)
- Hidden layers: configurable (default `[128, 64]`)
- Output: `D_latent + 3` (latent embedding + 3D velocity prediction)
- The output is split: first 3 values → `pred_vel`, remaining `D_latent` values → `z_s`
- `z_s` is L2-normalized before use

**Target Network (`HIMEstimator.target`)**

A separate, smaller MLP that processes only the current (next) single-step observation:
- Input: `D_obs` (single step)
- Hidden layers: configurable (default `[128, 64]`)
- Output: `D_latent`
- Output `z_t` is L2-normalized before use
- This network is **trained jointly** with the encoder (not an EMA target — both receive gradients)

**Prototype Layer (`HIMEstimator.proto`)**

An `nn.Embedding` of shape `[K × D_latent]` (default `K=32`):
- Rows are the learnable prototype vectors
- Prototype weights are **re-normalized to unit length** at every update step (no-gradient op) before score computation
- Trained jointly via the swap loss

**Actor (`HIMActorCritic.actor`)**

An MLP taking the concatenation of:
- Normalized current policy observations: `D_obs`
- Predicted velocity: `3`
- Latent embedding: `D_latent`

Input dimension: `D_obs + 3 + D_latent`

**Critic** Inherited from `rsl_rl.modules.ActorCritic`.

#### Learning Setup

| Component | Optimizer | Loss |
|---|---|---|
| Estimator (encoder + target + proto) | Adam (separate) | `L_estimation + L_swap` |
| Actor + Critic | Adam (PPO optimizer) | PPO surrogate + value loss + entropy |

The estimator has its **own optimizer** (`HIMEstimator.optimizer`) that is updated independently inside `HIMPPO.update()`, before the PPO gradient step. Its learning rate is synchronized with the PPO learning rate (adaptive schedule or fixed).

During **actor inference**, the encoder runs in `no_grad` mode, producing detached `vel` and `latent` tensors as additional actor inputs.

---

### 3. Loss Functions

#### 3.1 Estimation Loss

**Purpose:** Supervised regression of the robot's base velocity from proprioceptive history.

**Formulation:**
```
L_estimation = MSE(pred_vel, vel_GT)
             = (1/B) * Σ_i || pred_vel_i - vel_GT_i ||^2
```

Where:
- `pred_vel` = first 3 outputs of the encoder, representing the predicted 3D base velocity
- `vel_GT` = ground-truth base velocity from `obs["estimatorGT"]` (provided by the simulation environment, detached — not backpropagated through)

**Implementation:**
```python
# him_estimator.py, lines 99, 120
pred_vel, z_s = z_s[..., :3], z_s[..., 3:]
estimation_loss = F.mse_loss(pred_vel, vel)
```

**File:** `himloco/himloco/modules/him_estimator.py`, lines 99 and 120

**Note:** `obs["estimatorGT"]` must be registered as an observation key in the environment and agent configuration. The comment on line 94 of `him_estimator.py` explicitly states this requirement.

---

#### 3.2 Swap Loss (Prototype Contrastive Loss)

**Purpose:** Self-supervised contrastive learning to align the encoder's history-based latent representation with the target network's single-step latent representation, using shared prototype assignments as the learning signal.

**Formulation:**
```
L_swap = -0.5 * mean[ q_s · log(softmax(score_s / τ)) + q_t · log(softmax(score_t / τ)) ]
```

Expanded:
```
score_s = z_s @ C^T                             (encoder scores against prototypes)
score_t = z_t @ C^T                             (target scores against prototypes)
q_s = Sinkhorn(score_s)                         (no grad — assignment target)
q_t = Sinkhorn(score_t)                         (no grad — assignment target)
log_p_s = log_softmax(score_s / τ)              (predicted log-distribution from encoder)
log_p_t = log_softmax(score_t / τ)              (predicted log-distribution from target)
L_swap = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
```

**Implementation:**
```python
# him_estimator.py, lines 101-119
z_s = F.normalize(z_s, dim=-1, p=2)            # L2 normalize encoder latent
z_t = F.normalize(z_t, dim=-1, p=2)            # L2 normalize target latent

with torch.no_grad():                           # Normalize prototype weights in-place
    w = self.proto.weight.data.clone()
    w = F.normalize(w, dim=-1, p=2)
    self.proto.weight.copy_(w)

score_s = z_s @ self.proto.weight.T             # [B x K]
score_t = z_t @ self.proto.weight.T             # [B x K]

with torch.no_grad():                           # Sinkhorn assignments: no gradient
    q_s = sinkhorn(score_s)                     # [B x K]
    q_t = sinkhorn(score_t)                     # [B x K]

log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
```

**File:** `himloco/himloco/modules/him_estimator.py`, lines 101–119

---

#### 3.3 Total Estimator Loss

```
L_total = L_estimation + L_swap
```

**Implementation:**
```python
# him_estimator.py, lines 120-126
losses = estimation_loss + swap_loss
self.optimizer.zero_grad()
losses.backward()
nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
self.optimizer.step()
```

**File:** `himloco/himloco/modules/him_estimator.py`, lines 120–126

Gradient norms are clipped to `max_grad_norm` (default `10.0`).

---

#### 3.4 PPO Losses (Actor-Critic)

These are standard RSL-RL PPO losses computed in `HIMPPO.update()` and are inherited from the parent `PPO` class. They are listed here for completeness:

| Loss | Formula | Code Location |
|---|---|---|
| Surrogate (policy) | `max(-A*r, -A*clip(r, 1-ε, 1+ε))` | `him_ppo.py`, lines 243–250 |
| Value function | `max((V-R)^2, (V_clipped-R)^2)` | `him_ppo.py`, lines 253–261 |
| Entropy bonus | `-H(π)` (negated) | `him_ppo.py`, line 263 |
| Total PPO | `L_surr + c_v * L_val - c_e * H` | `him_ppo.py`, line 263 |

The estimator losses (`estimation_loss`, `swap_loss`) are tracked and logged separately — they do **not** flow into the PPO gradient update.

---

### 4. Key Classes and Interfaces

#### `HIMEstimator`

- **Source:** `himloco/himloco/modules/him_estimator.py`
- **Type:** `nn.Module`
- **Description:** The core internal model. Encodes proprioceptive history into a velocity prediction and a contrastive latent representation.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `temporal_steps` | `int` | — | Number of historical timesteps `T` |
| `num_one_step_obs` | `int` | — | Dimension of a single observation step `D_obs` |
| `enc_hidden_dims` | `list[int]` | `[128, 64, 16]` | Encoder MLP hidden layer sizes. Last element is `D_latent`. |
| `tar_hidden_dims` | `list[int]` | `[128, 64]` | Target MLP hidden layer sizes |
| `activation` | `str` | `"elu"` | Activation function name |
| `learning_rate` | `float` | `1e-3` | Estimator optimizer learning rate |
| `max_grad_norm` | `float` | `10.0` | Gradient norm clipping threshold |
| `num_prototype` | `int` | `32` | Number of prototype vectors `K` |
| `temperature` | `float` | `3.0` | Softmax temperature `τ` for swap loss |

**Member Variables:**

| Variable | Type | Description |
|---|---|---|
| `encoder` | `nn.Sequential` | History → `[pred_vel(3), z_s(D_latent)]` |
| `target` | `nn.Sequential` | Single obs → `z_t(D_latent)` |
| `proto` | `nn.Embedding(K, D_latent)` | Learnable prototype matrix |
| `optimizer` | `optim.Adam` | Dedicated Adam optimizer for estimator |
| `actor_obs_normalizer` | `callable` | Observation normalization function (set externally by `HIMActorCritic`) |
| `num_latent` | `int` | `enc_hidden_dims[-1]` — latent dimension `D_latent` |
| `temperature` | `float` | Temperature `τ` for softmax |
| `max_grad_norm` | `float` | Gradient clipping bound |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `forward` | `(obs_history: Tensor) -> (vel: Tensor, z: Tensor)` | Runs encoder, splits output, L2-normalizes `z`. Returns detached tensors. Used during actor rollout. Line 72. |
| `encode` | `(obs_history: Tensor) -> (vel: Tensor, z: Tensor)` | Same as `forward` but returns non-detached `vel`. Line 78. |
| `get_latent` | `(obs_history: Tensor) -> (vel: Tensor, z: Tensor)` | Calls `encode` and detaches both outputs. Line 68. |
| `update` | `(obs: dict, next_obs: dict, lr: float) -> (float, float)` | Runs one estimator training step. Returns `(estimation_loss, swap_loss)`. Line 84. |

---

#### `sinkhorn`

- **Source:** `himloco/himloco/modules/him_estimator.py`, line 132
- **Type:** Function (`@torch.no_grad()`)
- **Signature:** `sinkhorn(out: Tensor, eps: float = 0.05, iters: int = 3) -> Tensor`
- **Description:** Sinkhorn-Knopp normalization to compute soft prototype assignments `Q` from raw prototype scores. Enforces uniform marginals over prototypes (anti-collapse). Input shape: `[B × K]`. Output shape: `[B × K]`, values sum to 1 per row.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `out` | `Tensor [B×K]` | — | Raw prototype scores |
| `eps` | `float` | `0.05` | Sinkhorn regularization coefficient `ε` |
| `iters` | `int` | `3` | Number of alternating normalization iterations |

---

#### `HIMActorCritic`

- **Source:** `himloco/himloco/modules/him_actor_critic.py`
- **Type:** `nn.Module`, extends `rsl_rl.modules.ActorCritic`
- **Description:** Actor-critic policy that augments the standard actor input with velocity and latent from the `HIMEstimator`.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `obs` | `TensorDict` | Observation dict including `"policy"` and `"obsHistory"` keys |
| `obs_groups` | `dict[str, list[str]]` | Maps observation group names to keys |
| `num_actions` | `int` | Action space dimension |
| `encoder_cfg` | `dict` | Config for `HIMEstimator` (`hidden_dims`, `activation`) |
| `actor_hidden_dims` | `list[int]` | Actor MLP hidden sizes (default `[256, 256, 256]`) |
| `critic_hidden_dims` | `list[int]` | Critic MLP hidden sizes |
| `activation` | `str` | Activation function |
| `init_noise_std` | `float` | Initial action noise std |

**Member Variables:**

| Variable | Type | Description |
|---|---|---|
| `estimator` | `HIMEstimator` | The internal model estimator |
| `actor` | `MLP` | Policy network; input dim = `D_obs + 3 + D_latent` |
| `std` | `nn.Parameter` | Learned action noise (scalar per action) |
| `distribution` | `Normal` | Current action distribution |
| `history_size` | `int` | Number of historical timesteps `T` from `obs["obsHistory"].shape[1]` |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `_update_distribution` | `(obs: TensorDict) -> None` | Encodes obs history, concatenates `[obs, vel, latent]`, computes action mean and Normal distribution. Line 148. |
| `act` | `(obs: TensorDict) -> Tensor` | Updates distribution, samples action. Line 160. |
| `act_inference` | `(obs: TensorDict) -> Tensor` | Returns deterministic mean action (no sampling). Line 167. |
| `get_actions_log_prob` | `(actions: Tensor) -> Tensor` | Log probability of given actions under current distribution. Line 164. |

---

#### `HIMPPO`

- **Source:** `himloco/himloco/algorithms/him_ppo.py`
- **Type:** Class, extends `rsl_rl.algorithms.PPO`
- **Description:** PPO algorithm extended to run the estimator update on every mini-batch and to track/log the HIM-specific losses.

**Key Additions over PPO:**

| Addition | Description | Line |
|---|---|---|
| `transition.next_observations` | Stores next obs for contrastive learning in `process_env_step` | 117 |
| Estimator update call | `self.policy.estimator.update(obs_batch, next_obs_batch, lr=...)` per mini-batch | 238–240 |
| `mean_estimation_loss` | Tracked and averaged over all updates | 128, 347, 361 |
| `mean_swap_loss` | Tracked and averaged over all updates | 129, 348, 362 |
| Loss dict keys | `"estimattion_loss"`, `"swap_loss"` added to returned dict | 376–377 |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `init_storage` | `(training_type, num_envs, num_transitions_per_env, obs, action_shape)` | Instantiates `HIMRolloutStorage` instead of base `RolloutStorage`. Line 92. |
| `process_env_step` | `(obs, rewards, dones, extras)` | Stores `next_observations` in transition before calling parent. Line 110. |
| `update` | `() -> dict[str, float]` | Full PPO + estimator update loop. Returns loss dict including HIM losses. Line 120. |

---

#### `HIMRolloutStorage`

- **Source:** `himloco/himloco/storage/him_rollout_storage.py`
- **Type:** Class, extends `rsl_rl.storage.RolloutStorage`
- **Description:** Extends rollout storage to hold `next_observations` at every timestep, enabling the estimator's contrastive learning (which requires pairs of `(obs_t, obs_{t+1})`).

**Member Variables:**

| Variable | Type | Description |
|---|---|---|
| `next_observations` | `TensorDict [T × B × ...]` | Stores the observation at the next timestep for every transition |

**Key Methods:**

| Method | Signature | Description |
|---|---|---|
| `add_transitions` | `(transition: Transition)` | Copies `transition.next_observations` into `self.next_observations[step]` in addition to all standard fields. Line 80. |
| `mini_batch_generator` | `(num_mini_batches, num_epochs) -> Generator` | Yields `(obs_batch, next_obs_batch, ...)` tuples. `next_obs_batch` is passed to the estimator update. Line 127. |

**Inner Class `HIMRolloutStorage.Transition`:**

| Field | Type | Description |
|---|---|---|
| `observations` | `TensorDict` | Current step observations |
| `next_observations` | `TensorDict` | Next step observations (set in `HIMPPO.process_env_step`) |
| `actions` | `Tensor` | Sampled actions |
| `rewards` | `Tensor` | Step rewards |
| `dones` | `Tensor` | Episode termination flags |
| `values` | `Tensor` | Critic value estimates |
| `actions_log_prob` | `Tensor` | Log probability of sampled actions |
| `action_mean` | `Tensor` | Mean of action distribution |
| `action_sigma` | `Tensor` | Std of action distribution |
| `hidden_states` | `tuple` | RNN hidden states (unused — recurrent not supported) |

---

#### `HIMOnPolicyRunner`

- **Source:** `himloco/himloco/runners/him_on_policy_runner.py`
- **Type:** Class, extends `rsl_rl.runners.OnPolicyRunner`
- **Description:** Training runner that wires together the HIM-specific components. Requires `"encoder"` key in `train_cfg`.

**Key Methods:**

| Method | Signature | Description |
|---|---|---|
| `_construct_algorithm` | `(obs: TensorDict) -> HIMPPO` | Constructs `HIMActorCritic` and `HIMPPO` with encoder config. Line 59. |
| `save` | `(path, infos)` | Saves model state dict and **two** optimizer state dicts: PPO optimizer + estimator optimizer. Line 117. |
| `load` | `(path, load_optimizer)` | Loads model and both optimizer states. Line 129. |

**Constructor Requirements:**

| Key | Location | Description |
|---|---|---|
| `train_cfg["encoder"]` | `train_cfg` dict | Must be present. Contains `hidden_dims` and `activation` for `HIMEstimator`. |

---

#### `get_activation`

- **Source:** `himloco/himloco/modules/him_estimator.py`, line 148
- **Type:** Function
- **Signature:** `get_activation(act_name: str) -> nn.Module`
- **Description:** Maps activation name string to a PyTorch activation module. Supported: `"elu"`, `"selu"`, `"relu"`, `"crelu"` (→ ReLU), `"silu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`.

---

## Appendix

---

### Appendix A: Lagrange Multipliers

#### A.1 The Core Problem: Constrained Optimisation

Most optimisation problems in machine learning minimise an objective freely over all possible parameter values. Constrained optimisation is different: we want the minimum of an objective `f(x)` subject to one or more equality constraints `g_k(x) = 0`. The challenge is that naively ignoring the constraints and minimising `f` will almost never land exactly on the constraint surface.

The canonical example is: *minimise `f(x, y) = x² + y²` subject to `x + y = 1`.* The unconstrained minimum is the origin `(0,0)`, which violates the constraint. The constrained minimum lies somewhere on the line `x + y = 1`.

#### A.2 Geometric Intuition

At any feasible point `x*` that satisfies the constraint `g(x) = 0`, ask: what direction can we move along the constraint surface without increasing `f`? If `x*` is a constrained minimum, there is no such direction and we are at an optimum.

This has a precise geometric characterisation. The gradient `∇f(x*)` (the direction of steepest ascent of `f`) must be **perpendicular to the constraint surface** at `x*`. If it had any component *along* the surface, we could reduce `f` by moving in the opposite direction while staying feasible.

The constraint surface at `x*` is described locally by the set of vectors perpendicular to `∇g(x*)`. Therefore, the condition "∇f is perpendicular to the constraint surface" is equivalent to:

```
∇f(x*)  =  λ * ∇g(x*)        for some scalar λ
```

The scalar `λ` is the **Lagrange multiplier**. It captures how much the constraint is "pushing back" against the objective gradient.

#### A.3 The Lagrangian Function

Instead of reasoning geometrically, we can encode the same condition algebraically by defining the **Lagrangian**:

```
L(x, λ)  =  f(x)  −  λ * g(x)
```

Taking the gradient of `L` with respect to `x` and setting to zero recovers exactly the stationarity condition above:

```
∇_x L  =  ∇f(x)  −  λ * ∇g(x)  =  0
    ↔    ∇f(x)   =  λ * ∇g(x)
```

Taking the gradient of `L` with respect to `λ` and setting to zero recovers the constraint itself:

```
∂L / ∂λ  =  −g(x)  =  0
        ↔   g(x)   =  0
```

So the stationary points of `L(x, λ)`, where both partial derivatives are zero, are exactly the constrained optima of the original problem. The Lagrangian turns a constrained problem into a system of unconstrained equations in the joint variable `(x, λ)`.

#### A.4 Multiple Constraints

With multiple equality constraints `g_1(x) = 0, ..., g_m(x) = 0`, one multiplier is introduced per constraint:

```
L(x, λ_1, ..., λ_m)  =  f(x)  −  Σ_k  λ_k * g_k(x)
```

The stationarity conditions are:

```
∇_x L  =  ∇f(x)  −  Σ_k  λ_k * ∇g_k(x)  =  0       (stationarity in x)
∂L / ∂λ_k  =  −g_k(x)  =  0   for all k              (feasibility)
```

The multiplier `λ_k` can be interpreted as the **sensitivity of the optimal objective value to the constraint**: if the constraint `g_k(x) = 0` is relaxed to `g_k(x) = δ`, the optimal objective changes by approximately `λ_k * δ`. This is sometimes called the **shadow price** of the constraint.

#### A.5 KKT Conditions

When the objective and constraints are differentiable and the constraints satisfy a regularity condition (constraint qualification), the Karush-Kuhn-Tucker (KKT) conditions generalise the Lagrange conditions to also handle inequality constraints `h_k(x) ≤ 0`. For pure equality constraints — as in the EOT problem — the KKT conditions reduce exactly to the Lagrange stationarity conditions above. The EOT problem has only equality constraints (row marginals and column marginals), so plain Lagrange multipliers suffice.

#### A.6 Application in Entropic Optimal Transport

In the EOT problem, the optimisation variable is the transport plan `T ∈ R^(B×K)`, treated as a matrix of `B*K` independent scalars `T_ij`. There are `B + K` equality constraints:

- `B` row-marginal constraints: `Σ_j T_ij = μ_i` for each `i = 1, ..., B`
- `K` column-marginal constraints: `Σ_i T_ij = ν_j` for each `j = 1, ..., K`

Following the recipe in §A.4, introduce one multiplier per constraint: `λ_i` for each row constraint and `ρ_j` for each column constraint. The Lagrangian is:

```
L(T, λ, ρ)  =  Σ_{ij} T_ij * C_ij  +  ε * Σ_{ij} T_ij * ln(T_ij)
              − Σ_i  λ_i * ( Σ_j T_ij − μ_i )
              − Σ_j  ρ_j * ( Σ_i T_ij − ν_j )
```

Because the entropy term `ε * T_ij * ln(T_ij)` makes `L` strictly convex in each `T_ij`, the stationarity condition `∂L/∂T_ij = 0` has a unique solution. Solving it yields `T*_ij = u_i * K_ij * v_j`, where `u_i = exp((λ_i − ε/2)/ε)` and `v_j = exp((ρ_j − ε/2)/ε)` are the Lagrange multipliers exponentiated and rescaled. In this context the multipliers no longer have a shadow-price interpretation — they are purely the scaling vectors that enforce the marginal constraints, and finding them is equivalent to running the Sinkhorn algorithm.
