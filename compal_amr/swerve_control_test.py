import numpy as np
from math import atan2, sqrt, pi

class SwerveDriveController:
    def __init__(self, wheelbase, trackwidth):
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

            wheel_speeds.append(speed)
            wheel_angles.append(angle)

        return np.array([wheel_angles]), np.array([wheel_speeds])

if __name__ == "__main__":
    controller = SwerveDriveController(wheelbase=0.45, trackwidth=0.23)

    # [vx, vy, vz, wx, wy, wz]
    cmd_vel = [0.0, 1.0, 0.0]  

    angles, speeds = controller.compute(cmd_vel)

    print("Wheel Angles (rad):", angles)
    print("Wheel Speeds (m/s):", speeds)

