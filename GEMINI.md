# GEMINI Code Companion Report

This document provides a high-level overview of the `tron1-rl-isaaclab-cozum` repository, designed for training bipedal robot locomotion policies using Isaac Lab and RSL-RL.

> [!IMPORTANT]
> For a comprehensive technical deep-dive into the project structure, class hierarchies, and the framework for implementing updates or new tasks, please refer to **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Repository Overview

The repository is organized into three main functional areas:

-   **`environments/`**: Contains the `environments` extension, defining robot assets, environments, and Markov Decision Processes (MDPs).
-   **`scripts/`**: Entry-point scripts for training (`train.py`) and evaluation (`play.py`).
-   **`himloco/`**: Implementation of the History Information Model (HIM) architecture, including specialized algorithms and runners.

### Core Simulation & RL Framework

The project utilizes **Isaac Lab** for physics simulation and environment management, while **RSL-RL** handles the reinforcement learning algorithm (PPO). The integration is driven by a modular, inheritance-based configuration system that allows for rapid experimentation across different robot types (Point-Foot, Sole-Foot, Wheel-Foot) and architectures (Standard PPO, HIM).

-   **Entry Point:** Training is initiated via `scripts/rsl_rl/train.py`, which leverages `isaaclab.app.AppLauncher`.
-   **Task Selection:** The script parses a Task ID (e.g., `Isaac-Limx-SF-Blind-Flat-v0`) to load the corresponding environment and agent configurations registered in the system.

## Logging Management

Logging is handled by the `train.py` script and the RSL-RL runner.

-   **Log Directory:** The `train.py` script creates a log directory for each experiment in `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>`.
-   **Configuration Logging:** Before starting the training, the script saves the environment and agent configuration files (`env.yaml`, `agent.yaml`, `env.pkl`, `agent.pkl`) in the log directory.
-   **RSL-RL Logging:** The `OnPolicyRunner` from RSL-RL handles the logging of training statistics. While not explicitly detailed in the provided files, RSL-RL typically logs data such as:
    -   Mean reward and individual reward terms.
    -   Episode length.
    -   PPO-specific metrics like value loss, policy loss, and entropy.
    -   This data is usually saved in a format that can be visualized with tools like TensorBoard.
-   **Video Recording:** If the `--video` flag is used, the `train.py` script wraps the environment with `gym.wrappers.RecordVideo` to save video recordings of the training process in the log directory.
-   **Checkpoints:** The RSL-RL runner saves model checkpoints (`.pt` files) periodically in the log directory. The frequency of saving is controlled by the `save_interval` parameter in the agent configuration.
