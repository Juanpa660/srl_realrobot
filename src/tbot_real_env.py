#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped, Quaternion
from sensor_msgs.msg import LaserScan, JointState, Imu
from std_srvs.srv import Empty, EmptyRequest
from nav_msgs.msg import Odometry
import tf.transformations
import time
import numpy as np
from pynput import keyboard
import sys


class Tbot():
    
    def __init__(self, args):
        
        # Init Publisher
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
        self.cmd = Twist()

        self.init_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        self.init_pose = PoseWithCovarianceStamped()

        # Init Subscribers
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.callback_lidar)
        self.lidar_msg = LaserScan()

        self.amcl_pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.sub_callback)
        self.sub_msg = PoseWithCovarianceStamped()

        #self.joints_sub = rospy.Subscriber('/joint_states', JointState, self.joints_callback)
        #self.joints_msg = JointState()

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.odom_msg = Odometry()

        self.imu_sub = rospy.Subscriber('/mobile_base/sensors/imu_data', Imu, self.imu_callback)
        self.imu_msg = Imu()

        # ROBOT PARAMETERS
        self.wheel_radius = args.wheel_radius # meters CONFIRMAR!!!
        self.wheels_separation = args.wheels_separation # meters CONFIRMAR!!!
        self.max_scan_value = args.max_scan_value
        self.max_goal_dist = args.max_goal_dist
        self.pre_goal_dist = self.max_goal_dist / 2
        # Lidar SICK TIM parameters (obtainable from /scan topic)
        self.angle_min_max = args.angle_min_max # radians
        self.angle_increment_lidar = args.angle_increment_lidar # radians
        self.angle_increment_agent = args.angle_increment_agent # radians
        # GOAL
        self.goal_init_pos = np.array(args.goal_pos)

        self.n_steps = args.n_steps
        self.lidar_scan_len = args.obs_dim - 6
        self.limit_distance = args.limit_distance
        self.h_coeff = 10
        self.num_time_steps = args.num_time_steps

        # Initialize Service Client
        rospy.wait_for_service('/global_localization')
        self.disperse_particles_service = rospy.ServiceProxy('/global_localization', Empty)
        self.srv_request = EmptyRequest()

        # Other stuff
        self.training_mode = args.fine_tune_training
        self.crash_condition = False
        self.finish_condition = False
        if self.training_mode:
            keyboard.Listener(on_press=self.on_press).start()
        self.ctrl_c = False
        rospy.on_shutdown(self.shutdownhook)
        self.rate = rospy.Rate(args.ros_freq_rate)
        self.speed_reduction = args.speed_reduction
        self.ang_speed_reduction = args.ang_speed_reduction

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

    def move_robot(self, linear_speed=0.2, angular_speed=0.0, iterations=20):
        for i in range(iterations):
            self.cmd.linear.x = linear_speed
            self.cmd.angular.z = angular_speed
            self.vel_pub.publish(self.cmd)
            self.rate.sleep()

    def move_scan(self, v, radius, time_steps):
        rospy.loginfo("Starting Circle")
        self.move_robot(v, v/radius, time_steps)
        
    def set_initial_pose(self, x=0, y=0, theta=0):
        self.init_pose.header.stamp = rospy.Time.now()
        self.init_pose.header.frame_id = 'map'

        # Set your desired initial position (x, y) and orientation (theta)
        self.init_pose.pose.pose.position.x = x
        self.init_pose.pose.pose.position.y = y
        # theta is the yaw angle (rotation around the Z axis in degrees), converted to a quaternion
        q = tf.transformations.quaternion_from_euler(0, 0, degree_to_rad(theta))
        self.init_pose.pose.pose.orientation = Quaternion(*q)
        # Publish the initial pose
        for _ in range(20):
            self.init_pose_pub.publish(self.init_pose)
            self.rate.sleep()

    def call_init_particles_service(self):
        rospy.loginfo("Calling Disperse Particles Service...")
        self.disperse_particles_service(self.srv_request)
        
    def callback_lidar(self, scan):
        self.lidar_msg = scan

    def sub_callback(self, msg):
        self.sub_msg = msg
    
    def joints_callback(self, msg):
        self.joints_msg = msg
    
    def imu_callback(self, msg):
        self.imu_msg = msg

    def odom_callback(self, msg):
        self.odom_msg = msg

    def calculate_covariance(self):
        
        rospy.loginfo("Calculating Covariance...")
        cov_x = self.sub_msg.pose.covariance[0]
        cov_y = self.sub_msg.pose.covariance[7]
        cov_z = self.sub_msg.pose.covariance[35]
        rospy.loginfo("Cov X: " + str(cov_x) + ", Cov Y: " + str(cov_y) + ", Cov Z: " + str(cov_z))
        cov = (cov_x+cov_y+cov_z)/3
        rospy.loginfo("Cov AVG: "+str(cov))
        
        return cov
    
    def getCost(self, h_dist):
        limit_d = self.limit_distance
        cost = 1.0/(1.0 + np.exp((h_dist - limit_d)*self.h_coeff))
        return cost
    
    def get_rob_pos(self):
        x_pos = self.sub_msg.pose.pose.position.x
        y_pos = self.sub_msg.pose.pose.position.y

        orientation = self.sub_msg.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        return np.array([x_pos, y_pos, yaw])
    
    def get_rob_vel(self):
        # USING /odom topic
        linear_velocity = self.odom_msg.twist.twist.linear.x
        angular_velocity = self.odom_msg.twist.twist.angular.z
        vels = np.array([linear_velocity, angular_velocity])
        return vels
    
    def get_rob_vel2(self):
        # USING /joint_states topic
        v_list = self.joints_msg.velocity
        if len(v_list) == 2:
            v_left, v_right = v_list
            linear_velocity = (v_left + v_right) * self.wheel_radius / 2.0
            angular_velocity = (v_right - v_left) * self.wheel_radius / self.wheels_separation
        else:
            linear_velocity = 0
            angular_velocity = 0
        vels = np.array([linear_velocity, angular_velocity])
        return vels

    def get_rob_acc(self):
        x_acceleration = self.imu_msg.linear_acceleration.x
        y_acceleration = self.imu_msg.linear_acceleration.y
        return x_acceleration
    
    def get_goal_dir(self):
        rob_pos = self.get_rob_pos()
        dif_pos = self.goal_pos - rob_pos[:2]
        theta = rob_pos[2]

        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        rel_goal_pos = np.matmul(rot_mat, dif_pos)
        goal_dist = np.linalg.norm(rel_goal_pos)
        goal_dir = rel_goal_pos/(goal_dist + 1e-8)
        #angle_to_goal = np.arctan2(dif_pos[1], dif_pos[0]) - rob_pos[2]
        #angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal)) / np.pi
        return goal_dir

    def set_random_goal(self):
        # Set random goal position
        goal_x = np.random.uniform(-4, 4)
        goal_y = np.random.uniform(-4, 4)
        self.goal_pos = [goal_x, goal_y]

    def get_lidar_scan(self):
        lidar_scan = lidar_downsampling(self.lidar_msg.ranges, self.angle_min_max, self.max_scan_value, 
                                        self.angle_increment_lidar, self.angle_increment_agent)
        return np.array(lidar_scan)

    def reset(self):
        self.action = np.zeros(2)
        self.rob_vel = np.zeros(2)
        self.goal_pos = self.goal_init_pos
        self.goal_met = False
        self.cur_step = 0
        self.max_steps = 20000
        state = self.get_state()

        return np.expand_dims(self.get_flatten_state(state), axis=0)

    def get_state(self):
        self.lidar_scan = self.get_lidar_scan()
        self.rob_pos = self.get_rob_pos()
        self.rob_vel = self.get_rob_vel()
        self.rob_acc = self.get_rob_acc()
        self.goal_dist = eucl_dist(self.goal_pos, self.rob_pos)
        self.goal_dir = self.get_goal_dir()

        if len(self.lidar_msg.ranges) == 0:
            self.lidar_scan = self.max_scan_value * np.ones(self.lidar_scan_len)

        state = {'goal_dir':self.goal_dir,  # done
                'goal_dist':self.goal_dist, # done
                'vel':self.rob_vel,         # done
                'acc':self.rob_acc,         # done
                'scan':self.lidar_scan}     # done
        
        return state
    
    def get_flatten_state(self, state):
        goal_dir = state['goal_dir']
        goal_dist = [state['goal_dist'] / self.max_goal_dist]
        vel = state['vel']
        acc = np.array([state['acc']])
        scan = 1.0 - (np.clip(state['scan'], 0.0, self.max_scan_value)/self.max_scan_value)
        state = np.concatenate([goal_dir, goal_dist, vel, acc/8.0, scan], axis=0)

        return state

    def on_press(self, key):
        if key == keyboard.KeyCode.from_char("x"):  # If 'x' is pressed
            if self.crash_condition:  # If 'x' was already pressed
                self.crash_condition = False  # Clear the crash flag
            else:
                print('Stopping episode because of crash')
                self.crash_condition = True
        elif key == keyboard.KeyCode.from_char("f"):  # If 'f' is pressed
            if self.finish_condition:  # If 'f' was already pressed
                self.finish_condition = False  # Clear the finish flag
            else:
                print('Stopping episode because of finish')
                self.finish_condition = True
    def step(self, action):
        done = False
        if len(action)==2:
            target_vel, target_ang_vel = action
        else:
            target_vel, target_ang_vel = action[0]
        target_vel = (target_vel+1)/2

        target_vel *= self.speed_reduction
        target_ang_vel *= self.ang_speed_reduction
        for j in range(self.num_time_steps):
            self.cur_step += 1
            #vel, ang_vel = self.get_rob_vel()
            #acc = self.get_rob_acc()

            self.cmd.linear.x = target_vel
            self.cmd.angular.z = target_ang_vel
            self.vel_pub.publish(self.cmd)
            self.rate.sleep()

        state = self.get_state()
        info = {"goal_dist": state['goal_dist'], "lidar_min": np.min(state["scan"]), "stop_flag": self.ctrl_c, "goal_met": self.goal_met}
        info["vel"] = target_vel
        info["ang_vel"] = target_ang_vel

        # reward
        goal_dist = state["goal_dist"]
        reward = self.pre_goal_dist - goal_dist
        self.pre_goal_dist = goal_dist
        if goal_dist < 0.3:
            reward += 1
            if self.goal_met:
                reward += 1
                print("Goal met, episode finished")
                self.finish_condition = True
                self.goal_met = False
            else:
                self.goal_pos = [0, 0]
                self.goal_met = True
        # CV
        num_cv = 0
        hazard_dist = np.min(state["scan"])
        if hazard_dist < self.limit_distance:
            num_cv += 1
        info["num_cv"] = num_cv
        info["cost"] = self.getCost(hazard_dist)
        # Keyboard Listener
        if self.crash_condition or self.finish_condition:
            done = True
            if self.crash_condition:
                temp_num_cv = int(max(self.max_steps - self.cur_step, 0) / 20)
                discount_factor = 0.99
                temp_cost = discount_factor*(1 - discount_factor**temp_num_cv)/(1 - discount_factor)
                info["num_cv"] += temp_num_cv
                info["cost"] += temp_cost
            if self.training_mode:
                if self.crash_condition:
                    print(" Reset environment and press x please...")
                elif self.finish_condition:
                    print(" Reset environment and press f please...")
                while self.crash_condition or self.finish_condition:
                    time.sleep(1)
                    pass # wait until a key is pressed again
                print(" Starting a new episode")
            self.cur_step = 0
        if done:
            info["terminal_observation"] = self.get_flatten_state(state)
                    
        return self.get_flatten_state(state), reward, done, info

### USEFUL FUNCTIONS ###
def eucl_dist(pos1, pos2):
    dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return dist

def lidar_downsampling(scan_ranges, angle_min_max, max_scan_value, original_angle_increment, target_angle_increment):
    scan_ranges = np.array(scan_ranges)
    scan_ranges[scan_ranges>max_scan_value] = max_scan_value
    scan_ranges[scan_ranges<0.01] = max_scan_value
    len_out_scan_values = int((angle_min_max[1]-angle_min_max[0]) / target_angle_increment)
    sample_rate = int(target_angle_increment / original_angle_increment)
    out_scan_ranges = []
    for i in range(len_out_scan_values):
        try:
            out_scan_ranges.append(np.min(scan_ranges[sample_rate * i:sample_rate * (i+1)]))
        except:
            out_scan_ranges.append(np.mean(scan_ranges[sample_rate * i:sample_rate * (i+1)]))
    return out_scan_ranges

def rad_to_degree(angle):
    return angle * 180 / np.pi

def degree_to_rad(angle):
    return angle * np.pi / 180
######
