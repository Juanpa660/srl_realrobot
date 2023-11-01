from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py import MjRenderContextOffscreen
import mujoco_py

from scipy.spatial.transform import Rotation
from copy import deepcopy
from gym import spaces
import numpy as np
import time
import gym
import io
import os

def theta2vec(theta):
    ''' Convert an angle (in radians) to a unit vector in that angle around Z '''
    return np.array([np.cos(theta), np.sin(theta), 0.0])

def random_rotation_quaternion():
    # Generate a random rotation angle between -180 and 180 degrees
    angle = np.random.uniform(-180, 180)

    # Convert to radians
    angle = np.deg2rad(angle)

    # Use scipy to create a rotation around z-axis and then convert to a quaternion
    rotation = Rotation.from_rotvec(angle * np.array([1, 0, 0]))
    quaternion = rotation.as_quat()

    return quaternion

def collision_conditions(name1, name2):
    obs_cond1 = ('wall' in name2 or 'box' in name2 or 'moving_object' in name2)
    obs_cond2 = ('wall' in name1 or 'box' in name1 or 'moving_object' in name1)
    rob1 = name1 in ['robot', 'rear', 'right', 'left'] 
    rob2 = name2 in ['robot', 'rear', 'right', 'left'] 

    return (rob1 and obs_cond1) or (rob2 and obs_cond2)



class Env(gym.Env):
    def __init__(self):
        abs_path = os.path.dirname(__file__)
        self.env_difficulty = "easy" # easy medium hard

        if self.env_difficulty == "easy":
            self.model = load_model_from_path(f'{abs_path}/env_easy.xml')
        elif self.env_difficulty == "medium":
            self.model = load_model_from_path(f'{abs_path}/env_medium.xml')
            self.num_hazard = 5
        elif self.env_difficulty == "hard":
            self.model = load_model_from_path(f'{abs_path}/env_hard.xml')
            self.num_hazard = 9
        else:
            print("Wrong env name, try with: easy/medium/hard")

        self.time_step = 0.002
        self.n_substeps = 1
        self.time_step *= self.n_substeps
        self.sim = MjSim(self.model, nsubsteps=self.n_substeps)
        self.viewer = None

        # for environment
        self.pre_goal_dist = 0.0
        self.control_freq = 30 # def 30
        self.num_time_step = int(1.0/(self.time_step*self.control_freq))
        self.limit_distance = 0.25 # LIMIT DISTANCE FROM HAZARD TO ROBOT TO BE A CV def:0.5
        self.limit_bound = 0.0
        self.goal_dist_threshold = 0.3
        self.h_coeff = 10.0
        self.max_steps = 1000
        self.cur_step = 0
        self.hazard_group = 2
        self.num_group = 6

        # goal coordinates
        if self.env_difficulty == "hard":
            self.list_obs_pos = []
            self.x_pos = np.arange(-1, 8, 0.25)
            self.y_pos = np.arange(-3, 3.25, 0.25)
            self.list_goal_pos = [[7.5, 2.5], [7.5, 0], [7.5, -2.5]]

        elif self.env_difficulty == "medium":
            self.list_goal_pos = [[[2.5, 0], [2.5, -2.5], [2.5, 2.5]], 
                                  [[5.25, 0], [5.25, -2.5], [5.25, 2.5]],
                                  [[7.5, 2.5], [7.5, -2.5]],
                                  [[5.25, 0], [5.25, -2.5], [5.25, 2.5]],
                                  [[2.5, 0], [2.5, -2.5], [2.5, 2.5]],
                                  [[0, 0]]
                                  ]
            
            self.copy_goal_pos = deepcopy(self.list_goal_pos)
        
        elif self.env_difficulty == "easy":
            self.x_pos = np.arange(-4.5, 4.75, 0.25)
            self.y_pos = np.arange(-3, 3.25, 0.25)


        # moving obstacle coordinates
        self.list_mov_obs_pos = [[6.8, 1.8], [6.8, -1.8]]
        self.amp = 1.2
        self.w = 2.5
        
        # for PID control
        self.p_coeff = 10.0
        self.d_coeff = 0.001
        self.ang_p_coeff = 2.0 # 6 before 2 default
        self.ang_d_coeff = 0.001

        # for state
        self.angle_resolution = 1
        self.angle_interval = 5
        self.angle_range = np.arange(-120.0, 120.0 + self.angle_resolution, self.angle_resolution)
        self.lidar_state_dim = int(len(self.angle_range)/self.angle_interval)

        self.max_scan_value = 4.5
        self.max_goal_dist = 8
        self.scan_value = np.zeros(self.lidar_state_dim, dtype=np.float32)
        self.robot_pose = np.zeros(3)
        self.robot_vel = np.zeros(2)
        self.pre_robot_vel = np.zeros(2)
        self.goal_pos = np.zeros(2)

        # domain randomization params
        self.control_noise = 0
        self.lidar_noise = 0
        self.pose_noise = 0

        # for action
        self.action = np.zeros(2)

        # state & action dimension
        self.action_dim = 2
        # GOAL DIR: 2, GOAL DIST: 1, VEL: 2 (LINEAR, ANGULAR), ACC: 1, SCAN: X, 
        self.state_dim = 2 + 1 + len(self.robot_vel) + 1 + len(self.scan_value)
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.state_dim), np.inf*np.ones(self.state_dim), dtype=np.float32)


    def set_pid(self, p, d, p_a, d_a):
        self.p = p
        self.d = d
        self.p_a = p_a
        self.d_a = d_a

    def getCost(self, h_dist):
        limit_d = self.limit_distance + self.limit_bound
        cost = 1.0/(1.0 + np.exp((h_dist - limit_d)*self.h_coeff))
        return cost

    def render(self, mode, **kwargs):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        self.viewer.render()

    def getSensor(self):
        sensor_dict = {'accelerometer':None, 'velocimeter':None, 'gyro':None}
        for sensor_name in sensor_dict.keys():
            id = self.sim.model.sensor_name2id(sensor_name)
            adr = self.sim.model.sensor_adr[id]
            dim = self.sim.model.sensor_dim[id]
            sensor_dict[sensor_name] = self.sim.data.sensordata[adr:adr + dim].copy()
        return sensor_dict

    def getLidar(self):
        lidar_value = np.zeros_like(self.angle_range, dtype=np.float32)
        pos = self.sim.data.get_body_xpos('robot').copy()
        rot_mat = self.sim.data.get_body_xmat('robot').copy()
        body = self.sim.model.body_name2id('robot')
        grp = np.array([i==self.hazard_group for i in range(self.num_group)], dtype='uint8')
        for i, angle in enumerate(self.angle_range):
            rad_angle = angle*np.pi/180.0
            vec = np.matmul(rot_mat, theta2vec(rad_angle))
            dist, _ = self.sim.ray_fast_group(pos, vec, grp, 1, body)
            if dist > 0:
                lidar_value[i] = dist
            else:
                lidar_value[i] = self.max_scan_value
        for i in range(len(self.scan_value)):
            self.scan_value[i] = np.min(lidar_value[self.angle_interval*i:self.angle_interval*(i+1)])
        self.scan_value *= 1 + np.random.normal(0, self.lidar_noise)
        return deepcopy(self.scan_value)

    def getState(self):
        self.sim.forward()
        sensor_dict = self.getSensor()
        self.robot_vel[0] = sensor_dict['velocimeter'][0]
        self.robot_vel[1] = sensor_dict['gyro'][2]
        robot_acc = np.array([self.robot_vel[0] - self.pre_robot_vel[0]])*self.control_freq
        self.pre_robot_vel = deepcopy(self.robot_vel)

        self.robot_pose = self.sim.data.get_body_xpos('robot').copy()
        self.robot_pose *= 1 + np.random.normal(0, self.pose_noise)

        robot_mat = self.sim.data.get_body_xmat('robot').copy()
        theta = Rotation.from_matrix(robot_mat).as_euler('zyx', degrees=False)[0]
        self.robot_pose[2] = theta

        rel_goal_pos = self.goal_pos - self.robot_pose[:2]
        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        rel_goal_pos = np.matmul(rot_mat, rel_goal_pos)
        goal_dist = np.linalg.norm(rel_goal_pos)
        goal_dir = rel_goal_pos/(goal_dist + 1e-8)

        ###
        #dif_pos = self.goal_pos - self.robot_pose[:2]
        #angle_to_goal = np.arctan2(dif_pos[1], dif_pos[0])
        #rotation_to_goal = angle_to_goal - self.robot_pose[2]
        #goal_dir = np.arctan2(np.sin(rotation_to_goal), np.cos(rotation_to_goal)) / np.pi
        ###

        vel = deepcopy(self.robot_vel)
        scan_value = self.getLidar()
        state = {'goal_dir':goal_dir,
                'goal_dist':goal_dist,
                'vel':vel,
                'acc':robot_acc,
                'scan':scan_value}
        return state

    def getFlattenState(self, state):
        #goal_dir = [state['goal_dir']]
        goal_dir = state['goal_dir']
        goal_dist = [state['goal_dist'] / self.max_goal_dist]
        vel = state['vel']
        acc = state['acc']
        scan = 1.0 - (np.clip(state['scan'], 0.0, self.max_scan_value)/self.max_scan_value)
        state = np.concatenate([goal_dir, goal_dist, vel, acc/8.0, scan], axis=0)
        return state

    def generate_positions(self, x_pos, y_pos, min_distance=1.5, min_origin_distance=1.5):
        # Initialize list to store obstacle positions
        obstacles = []
        # Define corners
        corners = np.array([[7.5, 2.5], [7.5, -2.5]])
        # Generate obstacle positions
        while len(obstacles) < self.num_hazard:
            # Generate a random position
            new_pos = np.array([np.random.choice(x_pos), np.random.choice(y_pos)])
            # Check if it's far enough from the existing positions, the origin, and the corners
            if (all(np.linalg.norm(new_pos - pos) >= min_distance for pos in obstacles) and
                np.linalg.norm(new_pos) >= min_origin_distance and
                new_pos[0] <= 6):
                #all(np.linalg.norm(new_pos - corner) >= min_origin_distance for corner in corners)):
                obstacles.append(new_pos)
        return obstacles
    
    def build(self):
        self.sim.reset()
        if self.env_difficulty == "hard":
            self.list_obs_pos = self.generate_positions(self.x_pos, self.y_pos)
            for i in range(len(self.list_obs_pos)):
                self.sim.data.set_joint_qpos('box{}'.format(i+1), [*self.list_obs_pos[i], 0.4, 1.0, 0.0, 0.0, 0.0])
                self.sim.data.set_joint_qvel('box{}'.format(i+1), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            self.mov_obs1 = self.sim.model.body_name2id("moving_object1")
            self.mov_obs2 = self.sim.model.body_name2id("moving_object2")

            mov_obs_idx = np.random.choice(len(self.list_mov_obs_pos), 2, replace=False)
            self.pos_mov_obs1 = self.list_mov_obs_pos[mov_obs_idx[0]]
            self.pos_mov_obs2 = self.list_mov_obs_pos[mov_obs_idx[1]]

        self.current_goal = 0
        random_rotation = random_rotation_quaternion()
        self.sim.data.set_joint_qpos('robot', [0, 0, 0.06344, *random_rotation])
        self.sim.data.set_joint_qvel('robot', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        goal_id = self.sim.model.body_name2id('goal')
        self.sim.data.xfrc_applied[goal_id] = [0.0, 0.0, 0.98, 0.0, 0.0, 0.0]

        self.sim.forward()

    def updateGoalPos(self):
        if self.env_difficulty == "easy":
            goal_x = np.random.choice(self.x_pos)
            goal_y = np.random.choice(self.y_pos)
            self.goal_pos = [goal_x, goal_y]

        elif self.env_difficulty == "medium":
            if self.goal_met:
                self.current_goal += 1
                self.goal_met = False
            goal_idx = np.random.choice(len(self.list_goal_pos[self.current_goal]))
            self.goal_pos = self.list_goal_pos[self.current_goal][goal_idx]

        elif self.env_difficulty == "hard":
            if self.goal_met:
                self.goal_pos = [0, 0]
            else:
                idx = np.random.choice(len(self.list_goal_pos))
                self.goal_pos = self.list_goal_pos[idx]

        self.sim.data.set_joint_qpos('goal', [*self.goal_pos, 0.25, 1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qvel('goal', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sim.forward()
        self.pre_goal_dist = self.getGoalDist()

    def getGoalDist(self):
        robot_pos = self.sim.data.get_body_xpos('robot').copy()
        return np.sqrt(np.sum(np.square(self.goal_pos - robot_pos[:2])))

    def reset(self):
        self.goal_met = False
        self.pre_vel = 0.0
        self.pre_ang_vel = 0.0

        self.pre_error = 0.0
        self.pre_ang_error = 0.0

        self.action = np.zeros(2)
        self.robot_vel = np.zeros(2)
        self.pre_robot_vel = np.zeros(2)
        self.build()
        self.updateGoalPos()
        state = self.getState()
        self.cur_step = 0
        return self.getFlattenState(state)

    def get_step_wise_cost(self):
        limit_d = self.limit_distance + self.limit_bound
        scan_value = self.getLidar()
        hazard_dist = np.min(scan_value)
        step_wise_cost = limit_d - hazard_dist
        return step_wise_cost

    def velocity_to_wheel_speeds(self, v, omega, b):
        w1 = v - (b * omega) / 2
        w2 = v + (b * omega) / 2

        return w1, w2

    def step(self, action):
        #print(self.control_freq)
        self.cur_step += 1
        target_vel, target_ang_vel = action
        real_vels = []
        real_ang_vels = []
        for j in range(self.num_time_step):
            self.sim.forward()
            sensor_dict = self.getSensor()
            vel = sensor_dict['velocimeter'][0]
            ang_vel = sensor_dict['gyro'][2]

            self.error = target_vel - vel
            self.error_diff = self.error - self.pre_error  # Error difference
            self.pre_error = deepcopy(self.error)  # Save the current error for next step

            self.ang_error = target_ang_vel - ang_vel
            self.ang_error_diff = self.ang_error - self.pre_ang_error  # Angular error difference
            self.pre_ang_error = deepcopy(self.ang_error)  # Save the current angular error for next step

            cmd = self.p*self.error + self.d*self.error_diff / self.time_step
            ang_cmd = self.p_a*self.ang_error + self.d_a*self.ang_error_diff / self.time_step

            left_wheel_cmd = (cmd - ang_cmd)
            right_wheel_cmd = (cmd + ang_cmd)

            self.sim.data.ctrl[0] = left_wheel_cmd
            self.sim.data.ctrl[1] = right_wheel_cmd

            self.sim.step()

        sensor_dict = self.getSensor()
        vel = sensor_dict['velocimeter'][0]
        ang_vel = sensor_dict['gyro'][2]

        aux_vel = deepcopy(vel)
        aux_ang_vel = deepcopy(ang_vel)

        real_vels.append(np.round(aux_vel, 2))
        real_ang_vels.append(np.round(aux_ang_vel, 2))

        return real_vels, real_ang_vels, 1, 1
    

    def step2(self, action):
        self.cur_step += 1
        real_vels = []
        real_ang_vels = []
        lin_acc = np.clip(action[0], -1.0, 1.0)
        action[0] = np.clip(action[0] + lin_acc/self.control_freq, 0.0, 1.0)
        action[1] = np.clip(action[1], -1.0, 1.0)

        target_vel, target_ang_vel = action
        for j in range(self.num_time_step):
            self.sim.forward()
            sensor_dict = self.getSensor()
            vel = sensor_dict['velocimeter'][0]
            ang_vel = sensor_dict['gyro'][2]
            acc = (vel - self.pre_vel)/self.time_step
            ang_acc = (ang_vel - self.pre_ang_vel)/self.time_step
            self.pre_vel = deepcopy(vel)
            self.pre_ang_vel = deepcopy(ang_vel)
            cmd = self.p_coeff*(target_vel - vel) + self.d_coeff*(0.0 - acc)
            ang_cmd = self.ang_p_coeff*(target_ang_vel - ang_vel) + self.ang_d_coeff*(0.0 - ang_acc)
            left_wheel_cmd = (cmd - ang_cmd)
            right_wheel_cmd = (cmd + ang_cmd)

            self.sim.data.ctrl[0] = left_wheel_cmd
            self.sim.data.ctrl[1] = right_wheel_cmd

            self.sim.step()

            sensor_dict = self.getSensor()
            vel = sensor_dict['velocimeter'][0]
            ang_vel = sensor_dict['gyro'][2]

            aux_vel = deepcopy(vel)
            aux_ang_vel = deepcopy(ang_vel)

            real_vels.append(np.round(aux_vel, 2))
            real_ang_vels.append(np.round(aux_ang_vel, 2))

        return real_vels, real_ang_vels, 1, 1
        

