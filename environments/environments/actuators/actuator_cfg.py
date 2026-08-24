# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.actuators import DCMotorCfg

from .actuator_pd import IdentifiedActuator


@configclass
class IdentifiedActuatorCfg(DCMotorCfg):
    """Configuration for direct control (DC) motor actuator model with identified friction."""

    class_type: type = IdentifiedActuator

    friction_static: float = MISSING
    """Static friction torque (in N-m)."""
    activation_vel: float = MISSING
    """Velocity at which static friction activates (in Rad/s)."""
    friction_dynamic: float = MISSING
    """Dynamic friction coefficient (in N-m-s/Rad)."""
