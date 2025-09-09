import io
from typing import List, Optional
from math import atan2, sqrt, pi

import carb
import numpy as np
import omni
import omni.kit.commands
import torch
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.rotations import euler_to_rot_matrix, quat_to_euler_angles, quat_to_rot_matrix
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf

class SwerveController:
    """For Swerve Control AMR"""

    def __init__(
        self,
        prim_path: str,
        wheelbase: float,
        trackwidth: float,
        name: str = "compal_amr",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load swerve controller.

        Args:
            prim_path {str} -- prim path of the robot on the stage
            name {str} -- name of the quadruped
            usd_path {str} -- robot usd filepath in the directory
            position {np.ndarray} -- position of the robot
            orientation {np.ndarray} -- orientation of the robot

        """
        self._stage = get_current_stage()
        self._prim_path = prim_path
        prim = get_prim_at_path(self._prim_path)

        if not prim.IsValid():
            prim = define_prim(self._prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("Could not find Robot assets path")

        self.robot = Articulation(prim_path=self._prim_path, name=name, position=position, orientation=orientation)

        self._dof_control_modes: List[int] = list()

        # control param
        self.L = wheelbase
        self.W = trackwidth
        self.R = sqrt(wheelbase**2 + trackwidth**2)

        # Wheel order: FL, FR, RL, RR
        self.wheel_positions = {
            'FL': [wheelbase/2, trackwidth/2],
            'FR': [wheelbase/2, -trackwidth/2],
            'RL': [-wheelbase/2, trackwidth/2],
            'RR': [-wheelbase/2, -trackwidth/2],
        }

        self.kp_gain = [50, 50, 50, 50, 0, 0, 0, 0]
        self.kd_gain = [1, 1, 1, 1, 1, 1, 1, 1]
    
    def normalize(self, angle, speed):
        angle = ((angle + pi) % (2*pi)) - pi

        if angle > pi/2:
            angle -= pi
            speed *= -1
        elif angle < -pi/2:
            angle += pi
            speed *= -1

        return angle, speed

    def compute(self, cmd_vel):
        vx, vy, wz = cmd_vel
        # Only use planar control
        w_z = wz

        wheel_angles = []
        wheel_speeds = []

        for name in ['FL', 'FR', 'RL', 'RR']:
            x_offset, y_offset = self.wheel_positions[name]

            # Relative rotational velocity at wheel position
            delta_vx = -w_z * y_offset
            delta_vy = w_z * x_offset

            total_vx = vx + delta_vx
            total_vy = vy + delta_vy

            speed = sqrt(total_vx**2 + total_vy**2)
            angle = atan2(total_vy, total_vx)
            angle, speed = self.normalize(angle, speed)

            wheel_speeds.append(speed)
            wheel_angles.append(angle)

        return wheel_angles, wheel_speeds

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """Calculates torques from position and velocity commands"""
        return (target_q - q) * kp + (target_dq - dq) * kd
    
    def advance(self, dt, command: np.array):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        """
        wheel_ang, wheel_speed = self.compute(command)
        pos_cmd = np.concatenate([wheel_ang, np.zeros(4)])
        vel_cmd = np.concatenate([np.zeros(4), wheel_speed])
        cur_pos = self.robot.get_joint_positions()
        cur_vel = self.robot.get_joint_velocities()

        action = ArticulationAction(joint_efforts=self.pd_control(pos_cmd, cur_pos, self.kp_gain, vel_cmd, cur_vel, self.kd_gain) )
        self.robot.apply_action(action)

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize robot the articulation interface, set up drive mode
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes("force")
        self.robot.get_articulation_controller().switch_control_mode("effort")

    def post_reset(self) -> None:
        """
        Post reset articulation
        """
        self.robot.post_reset()