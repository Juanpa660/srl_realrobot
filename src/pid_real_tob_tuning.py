#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Quaternion
from sensor_msgs.msg import LaserScan, JointState, Imu
from std_srvs.srv import Empty, EmptyRequest
from nav_msgs.msg import Odometry
import tf.transformations
import time
import numpy as np
import sys

class Tbot():
    
    def __init__(self):
        
        # Init Publisher
        rospy.init_node('tbot_pid_tuning', anonymous=True)
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
        self.cmd = Twist()

        self.imu_sub = rospy.Subscriber('/mobile_base/sensors/imu_data', Imu, self.imu_callback)
        self.imu_msg = Imu()

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.odom_msg = Odometry()

        # Other stuff
        self.ctrl_c = False
        rospy.on_shutdown(self.shutdownhook)
        self.rate = rospy.Rate(2)

    def imu_callback(self, msg):
        self.imu_msg = msg

    def odom_callback(self, msg):
        self.odom_msg = msg

    def shutdownhook(self):
        self.stop_robot()
        self.ctrl_c = True

    def stop_robot(self):
        rospy.loginfo("Stoping the robot")
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        for i in range(20):
            self.vel_pub.publish(self.cmd)
            self.rate.sleep()
        self.ctrl_c = True
        sys.exit("Stop program or Ctrl+C")

    def get_rob_vel(self):
        
        linear_velocity = self.odom_msg.twist.twist.linear.x
        angular_velocity = self.odom_msg.twist.twist.angular.z

        vels = np.array([linear_velocity, angular_velocity])
        return vels
    
    def get_rob_acc(self):
        x_acceleration = self.imu_msg.linear_acceleration.x
        y_acceleration = self.imu_msg.linear_acceleration.y
        return x_acceleration
    
    def step(self, target_vel, target_ang_vel, p_coeff, d_coeff, ang_p_coeff):
        vels = self.get_rob_vel()
        vel = vels[0]
        ang_vel = vels[1]
        acc = self.get_rob_acc()
        ang_acc = 0

        cmd_lineal = p_coeff*(target_vel - vel) + d_coeff*(0.0 - acc)
        cmd_angular = ang_p_coeff*(target_ang_vel - ang_vel) + d_coeff*(0.0 - ang_acc)

        self.cmd.linear.x = target_vel + target_vel*0.1
        self.cmd.angular.z = target_ang_vel + cmd_angular
        self.vel_pub.publish(self.cmd)
        self.rate.sleep()

        new_vels = self.get_rob_vel()
        new_vel = new_vels[0]
        new_ang_vel = new_vels[1]

        return np.round(new_vel, 2), np.round(cmd_lineal, 3), np.round(new_ang_vel, 3), np.round(cmd_angular, 3)