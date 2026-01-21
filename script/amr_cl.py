#!/usr/bin/env python3
import math
import rclpy
import threading
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np

class SwerveController(Node):
    def __init__(self):
        super().__init__('Swerve_Controller')

        self.cmd_vel = self.create_subscription(
        Twist,
        'cmd_vel',
        self.cmd_vel_callback,
        10) 

        self.joint_states = self.create_subscription(
        JointState,
        'joint_states',
        self.joint_states_callback,
        10)

        self.joint_commmand = self.create_publisher(
        JointState,
        'joint_command',
        10)

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.cur_pos = np.zeros(8)
        self.cur_vel = np.zeros(8)

        self.wheelbase = 0.5
        self.trackwidth = 0.4
        self.R = math.sqrt(self.wheelbase**2 + self.trackwidth**2)

        self.wheel_positions = {
            'FL': [self.wheelbase/2, self.trackwidth/2],
            'FR': [self.wheelbase/2, -self.trackwidth/2],
            'RL': [-self.wheelbase/2, self.trackwidth/2],
            'RR': [-self.wheelbase/2, -self.trackwidth/2],
        }

        self.kp_gain = [50, 50, 50, 50, 0, 0, 0, 0]
        self.kd_gain = [1, 1, 1, 1, 1, 1, 1, 1]

        self.timer = self.create_timer(0.05,self.advance)


        self.joint_state_position = ["FL_Z_joint", "FR_Z_joint","RL_Z_joint","RR_Z_joint"]
        self.joint_state_velocity = ["FLY_joint", "FR_Y_joint","RL_Y_joint","RR_Y_joint"]
        self.joint_names = self.joint_state_position + self.joint_state_velocity


    def cmd_vel_callback(self, msg:Twist):
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.wz = msg.angular.z

    def joint_states_callback(self, msg:JointState):
        name_to_index = {name:i for i,name in enumerate(msg.name)}
        for i ,joint_name in enumerate(self.joint_names):
            if joint_name in name_to_index:
                idx = name_to_index[joint_name]
                self.cur_pos[i] = msg.position[idx]
                self.cur_vel[i] = msg.velocity[idx]

    def normalize(self, angle, speed):
        angle = ((angle + math.pi) % (2*math.pi)) - math.pi

        if angle > math.pi/2:
            angle -= math.pi
            speed *= -1
        elif angle < -math.pi/2:
            angle += math.pi
            speed *= -1

        return angle, speed

    def compute(self):

        wheel_angles = []
        wheel_speeds = []

        for name in ['FL', 'FR', 'RL', 'RR']:
            x_offset, y_offset = self.wheel_positions[name]

            # Relative rotational velocity at wheel position
            delta_vx = -self.wz * y_offset
            delta_vy = self.wz * x_offset

            total_vx = self.vx + delta_vx
            total_vy = self.vy + delta_vy

            speed = math.sqrt(total_vx**2 + total_vy**2)
            angle = math.atan2(total_vy, total_vx)
            angle, speed = self.normalize(angle, speed)

            wheel_speeds.append(speed)
            wheel_angles.append(angle)

        return np.array(wheel_angles),np.array(wheel_speeds)
    
    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """Calculates torques from position and velocity commands"""
        return (target_q - q) * kp + (target_dq - dq) * kd
    
    def advance(self):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        """
        wheel_angles, wheel_speeds = self.compute()
        pos_cmd = np.concatenate([wheel_angles,np.zeros(4)])
        vel_cmd = np.concatenate([np.zeros(4),wheel_speeds])

        action = self.pd_control(pos_cmd, self.cur_pos, self.kp_gain, vel_cmd, self.cur_vel, self.kd_gain) 

        msg = JointState() # -> instance a empty message
        msg.name = self.joint_names
        msg.position = pos_cmd.tolist()
        msg.velocity = vel_cmd.tolist()

        self.joint_commmand.publish(msg)


 
def main():
    rclpy.init()
    node = SwerveController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
