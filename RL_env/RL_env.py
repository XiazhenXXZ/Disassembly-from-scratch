import threading
import sys
import rospy
import numpy as np
import shlex
import time
import signal
import subprocess
import cv2
import queue
import gymnasium as gym
from gymnasium import spaces
import pyrealsense2 as rs
from ultralytics import YOLO
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client
from scipy.spatial.transform import Rotation as R
import math
import tf
import tf.transformations

from Impedance_controller import *
from constants import *
from init_results_detection import detector, pose_calculation, start_cameras, camera_cleanup
from dual_camera_detection import *

class FrankaRLEnv(gym.Env):
    """Franka Emika Panda Robot RL Environment"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, FLAGS=None):
        super(FrankaRLEnv, self).__init__()

        self.roscore_process = None
        self.impedence_controller = None
        self.imp_controller = None
        self.camera = DualRealsenseCamera()
        self.initialized = False
        self.control_rate = 30  # Hz
        
        self.action_space = spaces.Box(
            low=np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]).astype(np.float32),
            high=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).astype(np.float32),
            dtype=np.float32
        )
  
        self.observation_space = spaces.Dict({
            'robot_state': spaces.Box(
                low=np.array([min_EEP_x, min_EEP_y, min_EEP_z,
                             min_Ori_r, min_Ori_p, min_Ori_y,
                             min_F_x, min_F_y, min_F_z,
                             min_T_x, min_T_y, min_T_z]).astype(np.float32),
                high=np.array([max_EEP_x, max_EEP_y, max_EEP_z,
                              max_Ori_r, max_Ori_p, max_Ori_y,
                              max_F_x, max_F_y, max_F_z,
                              max_T_x, max_T_y, max_T_z]).astype(np.float32),
                dtype=np.float32
            ),
            'object_detected': spaces.Discrete(2), 
            'object_position': spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.inf]),
                high=np.array([np.inf, np.inf, np.inf]),
                dtype=np.float32
            ),
            'object_rotation': spaces.Box(
                low=np.array([-np.pi, -np.pi, -np.pi]),
                high=np.array([np.pi, np.pi, np.pi]),
                dtype=np.float32
            ),
            'camera_source': spaces.Discrete(3)  
        })
        
        if FLAGS is not None and not self.initialize(FLAGS):
            raise RuntimeError("Environment initialization failed")

    def initialize(self, FLAGS):
        """Initialize all components"""
        try:
            # Start ROS core if not running
            if not self._is_roscore_running():
                self.roscore_process = subprocess.Popen('roscore')
                time.sleep(2)

            # Start camera
            if not self.camera.start_cameras():
                raise RuntimeError("Failed to start cameras")

            # Start impedance controller
            self.impedence_controller = subprocess.Popen(
                ['roslaunch', 'serl_franka_controllers', 'impedance.launch',
                 f'robot_ip:={FLAGS.robot_ip}', f'load_gripper:={FLAGS.load_gripper}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(5)  # Wait for controller to initialize
            
            # Initialize ROS node
            if not rospy.core.is_initialized():
                rospy.init_node('franka_rl_env', anonymous=True)
                    
            # Initialize robot controller
            self.imp_controller = ImpedencecontrolEnv()

            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            self.cleanup()
            return False

    def step(self, action):
        if not self.initialized:
            raise RuntimeError("Environment not initialized")

        try:
            self.step_count += 1
            exc_action = np.array(action)
            self.ImpedencePosition(exc_action)  

            rospy.sleep(1.0/self.control_rate)      

            # End effector position
            EEP_x = self.F_T_EE[0, 3]
            EEP_y = self.F_T_EE[1, 3]
            EEP_z = self.F_T_EE[2, 3]

            self.z = EEP_z
            print("z:", self.z)
        
            robot_obs = self._get_robot_observation()
            camera_obs = self.camera.get_latest_detection()

            self.reward = 0
            self.reward += 1000 * self.z - 130
            self.reward += -3 * (EEP_y - 0.02)
            self.reward += 3 * (EEP_x / 0.5)

            observation = self._format_observation(robot_obs, camera_obs)


            done = self._check_termination(observation)
            
            return observation, reward, done, {}
            
        except Exception as e:
            print(f"Step failed: {str(e)}")
            return None, 0, True, {'error': str(e)}

    def reset(self):
        """Reset the environment to initial state"""
        if not self.initialized:
            raise RuntimeError("Environment not initialized")
        
        try:
            self.imp_controller.reset_arm()
            time.sleep(1)
            self.imp_controller.set_reference_limitation()
            time.sleep(1)

            self._open_gripper()
            time.sleep(2)

            robot_obs = self._get_robot_observation()
            camera_obs = self.camera.get_latest_detection()
            observation = self._format_observation(robot_obs, camera_obs)
            
            return observation
            
        except Exception as e:
            print(f"Reset failed: {str(e)}")
            return None

    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'rgb_array':
            detection = self.camera.get_latest_detection()
            if detection and 'image' in detection:
                return detection['image']
            return None
        elif mode == 'human':
            pass

    def close(self):
        """Clean up environment"""
        self.cleanup()

    def _get_robot_observation(self):
        """Get robot state observation"""
        pose = self.imp_controller.get_current_pose()
        ft = self.imp_controller.get_current_ft()
        return np.concatenate([pose, ft])

    def _format_observation(self, robot_obs, camera_obs):
        """Format observation into standardized dict"""
        return {
            'robot_state': robot_obs,
            'object_detected': 1 if camera_obs and camera_obs['object_detected'] else 0,
            'object_position': camera_obs['position'] if camera_obs else np.zeros(3),
            'object_rotation': camera_obs['rotation'] if camera_obs else np.zeros(3),
            'camera_source': camera_obs['camera_source'] if camera_obs else 0
        }

    def _check_termination(self, observation):
        position = observation['robot_state'][:3]

        if (position[0] < min_EEP_x or position[0] > max_EEP_x or
            position[1] < min_EEP_y or position[1] > max_EEP_y or
            position[2] < min_EEP_z or position[2] > max_EEP_z):
            return True

        if observation['object_detected'] and np.linalg.norm(position - observation['object_position']) < 0.01:
            return True
            
        return False

    def _is_roscore_running(self):
        try:
            master = rospy.get_master()
            return master.is_online()
        except:
            return False

    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up resources...")
        camera_cleanup()
        
        if self.impedence_controller:
            self.impedence_controller.terminate()
            self.impedence_controller.wait(timeout=5)
            
        if self.roscore_process:
            self.roscore_process.terminate()
            self.roscore_process.wait(timeout=5)