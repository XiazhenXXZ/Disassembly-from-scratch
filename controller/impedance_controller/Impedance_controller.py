import os
import sys
import rospy
import csv
import numpy as np
import shlex
import time
# from psutil import Popen
import geometry_msgs.msg as geom_msg
import time
import subprocess
# from subprocess import PIPE
from dynamic_reconfigure.client import Client
from absl import app, flags, logging
from scipy.spatial.transform import Rotation as R
import os
import gymnasium as gym
import math
import tf
import tf.transformations
import re
import json
import threading
import random
import matplotlib.pyplot as plt

import franka_msgs.msg
import message_filters
import actionlib

from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveGoal, MoveAction, GraspAction, GraspGoal
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

from save_result_as_csv import build_force_timestep

# FLAGS = flags.FLAGS
# flags.DEFINE_string("robot_ip", None, "IP address of the robot.", required=True)
# flags.DEFINE_string("load_gripper", 'false', "Whether or not to load the gripper.")

class ImpedencecontrolEnv(gym.Env):
    def __init__(self):
        super(ImpedencecontrolEnv, self).__init__()
        self.eepub = rospy.Publisher('/cartesian_impedance_controller/equilibrium_pose', geom_msg.PoseStamped, queue_size=10)
        self.client = Client("/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node")
        

        # Force
        self.franka_EE_trans = []
        self.franka_EE_quat = []
        self.F_T_EE = np.empty((4,4))
        self.K_F_ext_hat_K = []
        self.ext_force_ee = []
        self.ext_torque_ee = []
        self.quat_fl = []
        self.quat_rr = []
        self.franka_fault = None

        self.force_history = []
        self.stop_flag = 0
        self.time_window = 5
        # self.threshold = 0.1
        self.max_steps = 20

        self._lock = threading.Lock()
        self.Fx = 0
        self.Fy = 0
        self.Fz = 0
        self.Tx = 0
        self.Ty = 0
        self.Tz = 0
        self.resultant_force = 0
        self.resultant_torque = 0
        # sub and pub
        # self.sub = rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.convert_to_geometry_msg, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.franka_callback)

        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.GetEEforce)
        
        # sub.registerCallback(self.gripper_state_callback)

        # ts = message_filters.TimeSynchronizer([franka_sub, eeforce_sub, gripper_sub], queue_size=10)
        # ts.registerCallback(self.GetEEforce)

    def parm_to_selection(self, id):
        if id == 0:
            self.disassembly_state = self.move_up()

        elif id ==1:
            self.disassemblystate = self.move_down()

        elif id ==2:
            self.disassemblystate = self.move_left()

        elif id ==3:
            self.disassemblystate = self.move_right()

        elif id==4:
            self.disassemblystate = self.move_front()

        elif id==5:
            self.disassemblystate = self.move_back()

        return self.disassemblystate

    def move_up(self):
        print("+z")
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            # print(fh[0])
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, 0, 0.05, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([0, 0, 0.05, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all():
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break


        disassemblydiff = abs(EEP_z_-EEP_z)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1

        return self.disassemblystate

    def move_down(self):
        print("-z")
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, 0, -0.05, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([0, 0, -0.05, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.0001).all() or (diff_ <= 0.0001).all():
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        # EEP_z_ = self.F_T_EE[2, 3]
        # fh = []
        # while True:
        #     _ = ImpedencecontrolEnv.monitor_force_change(fh)
        #     stopf = _
        #     print("stop singal", stopf)
        #     if stopf == 0:
        #         self.ImpedencePosition(0, 0, -0.05, 0, 0, 0)
        #     if stopf == 1:
        #         self.ImpedencePosition(0,0,0,0,0,0)
        #         break
        disassemblydiff = abs(EEP_z_-EEP_z)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_right(self):
        print("+y")
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, 0.05, 0, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([0, 0.05, 0, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])

                diff__ = abs (EEP_y - EEP_y_)
                print(diff__)
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all() or (diff__ <= 0.045):
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        # EEP_y_ = self.F_T_EE[1, 3]
        # fh = []
        # while True:
        #     _ = ImpedencecontrolEnv.monitor_force_change(fh)
        #     stopf = _
        #     print("stop singal", stopf)
        #     if stopf == 0:
        #         self.ImpedencePosition(0, 0.05, 0, 0, 0, 0)
        #     if stopf == 1:
        #         self.ImpedencePosition(0,0,0,0,0,0)
        #         break
        disassemblydiff = abs(EEP_y_-EEP_y)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_left(self):
        print("-y")
        stopf = 0
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                target_c = np.array(target)
                # print("Original position:",target_c)
                self.ImpedencePosition(0, -0.05, 0, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                # print("target_ = ", target_)
                target0 = target_c + np.array([0, -0.05, 0, 0, 0, 0])
                # print("target0",target0)
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all():
                    stopf = 1
                    break

                # elif (diff_ <= 0.002).all():
                #     stopf = 1
                #     print("it doesn't move at all")
                #     break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        # while True:
        #     _ = ImpedencecontrolEnv.monitor_force_change(fh)
        #     stopf = _
        #     print("stop singal", stopf)
        #     if stopf == 0:
        #         self.ImpedencePosition(0, -0.05, 0, 0, 0, 0)
        #     if stopf == 1:
        #         self.ImpedencePosition(0,0,0,0,0,0)
        #         break
        disassemblydiff = abs(EEP_y_-EEP_y)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_front(self):
        print("+x")
        stopf = 0
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0.05, 0, 0, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([0.05, 0, 0, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all():
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break

        disassemblydiff = abs(EEP_x_-EEP_x)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1

        stopf = 0
        return self.disassemblystate
        

    def move_back(self):
        print("-x")
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(-0.05, 0, 0, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([-0.05, 0, 0, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all():
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        disassemblydiff = abs(EEP_x_-EEP_x)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        
    def set_reference_limitation(self):
        time.sleep(1)
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            self.client.update_configuration({"translational_clip_" + direction: 0.01})
            self.client.update_configuration({"rotational_clip_" + direction: 0.04})
        time.sleep(1)
    
    def GetEEforce(self, FandT):
        # self.gripper_width = gripper.position[0]
        self.forceandtorque= np.array(FandT.K_F_ext_hat_K)
        self.Force = self.forceandtorque[0:3]
        self.Torque = self.forceandtorque[3:6]
        # self.Force = self.franka_callself.GetEEforce()back[0:3]
        # self.Torque = self.franka_callback[3:6]
        self.Fx = self.Force[0]
        self.Fy = self.Force[1]
        self.Fz = self.Force[2]
        self.Tx = self.Torque[0]
        self.Ty = self.Torque[1]
        self.Tz = self.Torque[2]
        self.resultant_force = math.sqrt(self.Fx**2 + self.Fy**2 + self.Fz**2)
        self.resultant_torque = math.sqrt(self.Tx**2 + self.Ty**2 + self.Tz**2)
        # print(self.Fz)
        # print(self.Fx, self.Fy, self.Fz, self.resultant_force, self.Tx, self.Ty, self.Tz, self.resultant_torque)
        # build_force_timestep(self.resultant_force)

        # print("force:", Fz,Fy,Fz)
        
        return self.Fx, self.Fy, self.Fz, self.Tx, self.Ty, self.Tz, self.resultant_force, self.resultant_torque
    
    

    def start_logging(self, filename: str, stop_event: threading.Event, update_interval: float = 0.1):
        """
        Start logging force data to CSV in a background thread.
        
        Args:
            filename (str): Path to CSV file (e.g., 'data/force_log.csv').
            stop_event (threading.Event): Signal to stop logging.
            update_interval (float): Delay between writes (seconds).
        """
        f = None  # Initialize file handle outside try block
        try:
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                # Write header if file is empty
                if f.tell() == 0:
                    writer.writerow([
                        'Time', 'Fx', 'Fy', 'Fz', 'Force_Resultant',
                        'Tx', 'Ty', 'Tz', 'Torque_Resultant'
                    ])
                
                start_time = time.time()
                while not stop_event.is_set():
                    current_time = time.time() - start_time
                    
                    # Safely read data (thread-safe)
                    with self._lock:
                        data = [
                            current_time,
                            self.Fx, self.Fy, self.Fz, self.resultant_force,
                            self.Tx, self.Ty, self.Tz, self.resultant_torque
                        ]
                    
                    writer.writerow(data)
                    f.flush()  # Ensure data is written to disk
                    time.sleep(update_interval)
                    
        except Exception as e:
            print(f"[ERROR] Logging failed: {e}")
        finally:
            if f is not None:  # Only close if file was opened
                f.close()
            print("CSV logging stopped.")

    def update_force_data(self, FandT):
        """Call this from your sensor thread to update force/torque values."""
        with self._lock:
            # Replace with your actual sensor reading logic (e.g., self.GetEEforce())
            self.forceandtorque= np.array(FandT.K_F_ext_hat_K)
            self.Force = self.forceandtorque[0:3]
            self.Torque = self.forceandtorque[3:6]
            # self.Force = self.franka_callself.GetEEforce()back[0:3]
            # self.Torque = self.franka_callback[3:6]
            self.Fx = self.Force[0]
            self.Fy = self.Force[1]
            self.Fz = self.Force[2]
            self.Tx = self.Torque[0]
            self.Ty = self.Torque[1]
            self.Tz = self.Torque[2]
            self.resultant_force = math.sqrt(self.Fx**2 + self.Fy**2 + self.Fz**2)
            self.resultant_torque = math.sqrt(self.Tx**2 + self.Ty**2 + self.Tz**2)
                
    
    def generate_chart(self, stop_event, update_interval=0.1, save_path = 'zy_rl_demo/final_plot_pull_twist.png'):
        plt.ion()
        fig, ax = plt.subplots()
        
        x_data = []
        y_data = []
        start_time = time.time()

        line, = ax.plot([], [], 'b')
        while not stop_event.is_set():
            # print(self.K_F_ext_hat_K)
            y = self.Fz
            x_data.append(time.time() - start_time)
            y_data.append(y)

            line.set_xdata(x_data)
            line.set_ydata(y_data)

            ax.relim()
            ax.autoscale_view(True, True, True)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(update_interval)

        plt.ioff()
        plt.savefig(save_path)
        plt.show()
        
    
    def calculate_average_force(self, force_history):
        
        return np.mean(force_history)
    
    def is_average_force_change_abnormal(self,prev_avg_force, curr_avg_force):
        force_change_rate = np.abs(curr_avg_force - prev_avg_force) / (np.abs(prev_avg_force) + 1e-6)

        return np.any(force_change_rate > 0.5)
    
    def monitor_force_change(self, force_history):
        self.stop_flag = 0
        current_forces = self.resultant_force
        force_history.append(current_forces)

        if len(force_history) > self.time_window:
            # print(len(force_history))
            force_history.pop(0)
        

        if len(force_history) == self.time_window:
            previous_average_force = self.calculate_average_force(force_history[:self.time_window // 2])
            current_average_force = self.calculate_average_force(force_history[self.time_window // 2:])

            if self.is_average_force_change_abnormal(previous_average_force, current_average_force):
                self.stop_flag = 1

        return self.stop_flag


    def initialrobot(self):
        fh = self.force_history
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                print("!!!!!!!is 0000000")
                # target0 = np.array([0.61, -0.256, 0.2, np.pi, 0, np.pi/2 + np.pi/4])
                target0 = np.array([0.534,0.002,0.0547, np.pi, 0, 0])
                self.MovetoPoint(target0)
                print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                diff = abs (target  - target0)
                if(diff <= 0.003).all():
                    stopf == 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break

    
    
    def MovetoPoint(self, Target):
        time.sleep(1)
        target = Target
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"# input("\033[33mPress enter to move the robot down to position. \033[0m")
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(target[0], target[1], target[2])
        quat = R.from_euler('xyz', [target[3], target[4], target[5]]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat[0], quat[1], quat[2], quat[3])
        self.eepub.publish(msg)
        time.sleep(3)
        # print("success!!!!!!!!!!!!!!!!!!!!!!")

    def franka_callback(self, data):
        # print(data)
        self.K_F_ext_hat_K = np.array(data.K_F_ext_hat_K)
        # print(self.K_F_ext_hat_K)

        # Tip of finger gripper
        self.O_T_EE = np.array(data.O_T_EE).reshape(4, 4).T
        # print(O_T_EE[:3, :3])
        quat_ee = tf.transformations.quaternion_from_matrix(self.O_T_EE)        

        # Flange of robot
        self.F_T_EE_ = np.array(data.F_T_EE).reshape(4, 4).T
        self.F_T_EE_1 = np.asmatrix(self.O_T_EE) * np.linalg.inv(np.asmatrix(self.F_T_EE_))

        # Hand TCP of robot
        self.hand_TCP = np.array([[0.7071, 0.7071, 0, 0],
                                 [-0.7071, 0.7071, 0, 0],
                                 [0, 0, 1, 0.1034],
                                 [0, 0, 0, 1]])
        self.F_T_EE = np.asmatrix(self.F_T_EE_1) * np.asmatrix(self.hand_TCP)

        self.quat_fl_ = tf.transformations.quaternion_from_matrix(self.F_T_EE_1)
        # print(quat_fl)
        self.Euler_fl_ = tf.transformations.euler_from_quaternion(self.quat_fl_)

        # print("self.F_T_EE:", self.F_T_EE)
        self.quat_fl = tf.transformations.quaternion_from_matrix(self.F_T_EE)
        # print(quat_fl)
        self.Euler_fl = tf.transformations.euler_from_quaternion(self.quat_fl)

        return self.K_F_ext_hat_K, self.F_T_EE
        # print("Force:", self.K_F_ext_hat_K)
        # self.ext_force_ee = self.K_F_ext_hat_K[0:2]
        # self.ext_torque_ee = self.K_F_ext_hat_K[3:5]
    

    
    def franka_state_callback(self, msg):
        self.cart_pose_trans_mat = np.asarray(msg.O_T_EE).reshape(4,4,order='F')
        self.cartesian_pose = {
            'position': self.cart_pose_trans_mat[:3,3],
            'orientation': tf.transformations.quaternion_from_matrix(self.cart_pose_trans_mat[:3,:3]) }
        self.franka_fault = self.franka_fault or msg.last_motion_errors.joint_position_limits_violation or msg.last_motion_errors.cartesian_reflex
 

    def reset_arm(self):
        time.sleep(1)
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(0.39, 0, 0.35)
        quat = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat[0], quat[1], quat[2], quat[3])
        # input("\033[33m\nObserve the surroundings. Press enter to move the robot to the initial position.\033[0m")
        self.eepub.publish(msg)
        time.sleep(1)
        print("reset!!!")

    def robot_control_grasptarget(self,target):
        print("grasp_target:", target)
        # open_gripper()
        # gripper_control(1)
        # print("Gripper opened.")
        time.sleep(1)

        self.grasptarget = target
        self.pre_grasptarget = target + np.array([0,0,0.2,0,0,0])

        self.MovetoPoint(self.pre_grasptarget)
        time.sleep(3)

        stopf = 0
        # fh = self.force_history
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                # target0 = np.array([0.61, -0.256, 0.2, np.pi, 0, np.pi/2 + np.pi/4])
                target0 = np.array(target)
                self.MovetoPoint(target0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                diff = abs (target[:3]  - target0[:3])
                if(diff <= 0.003).all():
                    stopf == 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break   

        time.sleep(2)
        stopf = 0 
        gripper_control(0)
        # close_gripper()    
        
        # return self.approachfail

    def _initialrobot(self, target):
        open_gripper()
        print("Gripper opened.")
        time.sleep(1)

        self.grasptarget = target
        self.pre_grasptarget = target + np.array([0,0,0.2,0,0,0])

        self.MovetoPoint(self.pre_grasptarget)
        time.sleep(3)

        fh = self.force_history
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                print("!!!!!!!is 0000000")
                # target0 = np.array([0.61, -0.256, 0.2, np.pi, 0, np.pi/2 + np.pi/4])
                target0 = np.array(target)
                self.MovetoPoint(target0)
                print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                diff = abs (target  - target0)
                if(diff <= 0.003).all():
                    stopf == 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        

    def robot_control_place(self, tz):
        self.grasptarget = np.array([0.422,0.340,tz,np.pi, 0, 0])
        self.pre_grasptarget = self.grasptarget + np.array([0,0,0.2,0,0,0])

        self.MovetoPoint(self.pre_grasptarget)
        time.sleep(1)
        self.MovetoPoint(self.grasptarget)
        time.sleep(1)

        open_gripper()

def gripper_control(state):
    # print('1111111')
    gripper_state = str(state)
    Command = ['rosrun', 'franka_real_demo', 'gripper_run', gripper_state]
    node_process = subprocess.Popen(Command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
    # print("111111")
    stdout, stderr = node_process.communicate()
    message = stdout
    msg = message.strip()
    match = re.search(r'\{.*\}', msg)
    if match:
        try:
            dict_str = match.group(0)
            data_dict = json.loads(dict_str)  
        except json.JSONDecodeError as e:
            print(e)
    else:
        print("none!!!!!!!!!!")

    if stderr:
        print("stderr:", stderr)
    gripper_width = data_dict.get("width", [])
    node_process.wait()  

    # print("Gripper Opened")
    return gripper_width


def open_gripper():
    # gripper_state = str(state)
    node_process = subprocess.Popen(shlex.split('rosrun franka_real_demo gripper_run 1'), stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
    # print("111111")
    stdout, stderr = node_process.communicate()
    message = stdout
    msg = message.strip()
    # print(msg)        # target_0 = np.array([0.61, -0.166, 0.15, np.pi, 0, np.pi/2 + np.pi/4])
        # imp_controller.MovetoPoint(target_0)
        # time.sleep(1)
    match = re.search(r'\{.*\}', msg)
    # print("2222222")
    if match:
        # print(match)
        try:
            dict_str = match.group(0)
            # print(dict_str)
            data_dict = json.loads(dict_str)
            # print(data_dict)
        except json.JSONDecodeError as e:
            print(e)

    else:
        print("none!!!!!!!!!!")

    if stderr:
        print("stderr:", stderr)
    # print("msg!!!!!!!!!!!!!!!!!!:", data_dict.get("width", []))
    gripper_width = data_dict.get("width", [])
    node_process.wait()  # Wait for the command to complete
    # ImpedencecontrolEnv.gripper_state_callback()

    print("Gripper Opened")
    return gripper_width


def close_gripper():
    node_process = subprocess.Popen(shlex.split('rosrun franka_real_demo gripper_run 0'),stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
    # print("111111")
    stdout, stderr = node_process.communicate()
    message = stdout
    msg = message.strip()
    # print(msg)
    match = re.search(r'\{.*\}', msg)
    # print("2222222")
    if match:
        # print(match)
        try:
            dict_str = match.group(0)
            # print(dict_str)
            data_dict = json.loads(dict_str)
            # print(data_dict)
        except json.JSONDecodeError as e:
            print(e)

    else:
        print("none!!!!!!!!!!")

    if stderr:
        print("stderr:", stderr)
    # print("msg!!!!!!!!!!!!!!!!!!:", data_dict.get("width", []))
    gripper_width = data_dict.get("width", [])
    node_process.wait()  # Wait for the command to complete
    print("Gripper Closed")

    return gripper_width