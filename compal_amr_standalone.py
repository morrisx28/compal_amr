import argparse
from isaacsim import SimulationApp

simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

import omni
import carb
import numpy as np
import sys
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.nucleus import get_assets_root_path
from compal_amr.amr_ctrl import SwerveController
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.extensions import enable_extension

# enable ROS2 bridge extension
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

import time

# Note that this is not the system level rclpy, but one compiled for omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from dataclasses import dataclass

@dataclass
class Pose:
    position = np.zeros(3)
    orientation = np.array([1.0, 0.0, 0.0, 0.0])

class AMRNav(Node):
    def __init__(self, env_usd_path, amr_init_pose: Pose):
        super().__init__("AMR_Nav")

        # setting up environment
        self.loadEnv(env_usd_path)

        self.first_step = True
        self.reset_needed = False
        self.cmd_scale = 10
        self.cmd = np.zeros(3) # [v_x, v_y, w_z]
        self.initAMR(amr_init_pose)

        self.ros_sub = self.create_subscription(Twist, "cmd_vel", self.cmdCallback, 1)
        self.world.reset()

    def cmdCallback(self, msg: Twist):
        if self.world.is_playing():
            self.cmd[0] = msg.linear.x * self.cmd_scale
            self.cmd[1] = msg.linear.y * self.cmd_scale
            self.cmd[2] = msg.angular.z * self.cmd_scale

    def loadEnv(self, env_usd_path):
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)

        prim = define_prim("/World/Ground", "Xform")
        asset_path = env_usd_path
        prim.GetReferences().AddReference(asset_path)
    
    def onPhysicStep(self, step_size):
        if self.first_step:
            self.amr.initialize()
            self.first_step = False
        elif self.reset_needed:
            self.world.reset(True)
            self.reset_needed = False
            self.first_step = True
        else:
            self.amr.advance(step_size, self.cmd)
        

    def initAMR(self, amr_init_pose: Pose):
        self.amr = SwerveController(
            prim_path="/World/AMR",
            wheelbase=0.45,
            trackwidth=0.23,
            name="Compal_AMR",
            usd_path="/home/csl/workspaces/compal_amr/model/compal_amr.usd",
            position=amr_init_pose.position,
            orientation=amr_init_pose.orientation
        )
        self.world.add_physics_callback("physics_step", callback_fn=self.onPhysicStep)
        self.world.reset()

    def run_simulation(self):
        self.reset_needed = False
        while simulation_app.is_running():
            self.world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.world.is_stopped():
                self.reset_needed = True

        # Cleanup
        self.destroy_node()
        simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        choices=["hospital"],
        default="hospital",
        help="Choice of navigation environment.",
    )
    args, _ = parser.parse_known_args()

    USD_PATH = '/home/csl/workspaces/compal_amr/model/ground.usd'

    amr_init_pose = Pose()
    if args.env == 'hospital':
        ENV_USD_PATH = USD_PATH
        amr_init_pose.position = np.array([0, 0, 0.19])

    rclpy.init()
    subscriber = AMRNav(ENV_USD_PATH, amr_init_pose)
    subscriber.run_simulation()