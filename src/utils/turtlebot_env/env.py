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
    angle = np.random.uniform(-180, 180)
    angle = np.deg2rad(angle)
    # Use scipy to create a rotation around z-axis and then convert to a quaternion
    rotation = Rotation.from_rotvec(angle * np.array([1, 0, 0]))
    quaternion = rotation.as_quat()
    return quaternion

def generate_lab_config():
    rob_pos = [0, 0]
    obs_pos = []
    goal_pos = [9.5, -1]
    return rob_pos, obs_pos, goal_pos

# function to generate obstacles and goal positions randomly
def generate_random_positions(x_pos, y_pos, total=12):
    # Define a function to check if the distance is greater than 1 between two points.
    def is_far_enough_from(pos1, pos2, dist):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) >= dist

    while True:
        all_positions = [(x, y) for x in x_pos for y in y_pos]
        np.random.shuffle(all_positions)

        positions = []
        while len(positions) < total and len(all_positions) > 0:  # 6 positions + 1 goal
            new_pos = all_positions.pop()
            if len(positions) == 0:
                # Robot
                rob_pos = new_pos
                positions.append(rob_pos)
            elif len(positions) < total - 5:
                # Obstacles
                if is_far_enough_from(new_pos, rob_pos, 0.8) and all(is_far_enough_from(new_pos, pos, 1.25) for pos in positions):
                    positions.append(new_pos)
            elif len(positions)== total - 5 or len(positions)== total - 3:
                # Moving obstacles
                if is_far_enough_from(new_pos, rob_pos, 0.8) and all(is_far_enough_from(new_pos, pos, 1.6) for pos in positions):
                    positions.append(new_pos)
            else:
                # Goal
                if is_far_enough_from(new_pos, rob_pos, 3.5) and all(is_far_enough_from(new_pos, pos, 1.5) for pos in positions):
                    positions.append(new_pos)

        # if positions found are less than 7, there is not enough positions meeting the conditions
        if len(positions) < total:
            continue
        # Split the list into positions and the goal
        rob_pos, positions, goal = positions[0], positions[1:-1], positions[-1]
        break
        
    return rob_pos, positions, goal

def collision_conditions(name1, name2):
    obs_cond1 = ('wall' in name2 or 'box' in name2 or 'moving_object' in name2)
    obs_cond2 = ('wall' in name1 or 'box' in name1 or 'moving_object' in name1)
    rob1 = name1 in ['robot', 'rear', 'right', 'left'] 
    rob2 = name2 in ['robot', 'rear', 'right', 'left'] 

    return (rob1 and obs_cond1) or (rob2 and obs_cond2)


class Env(gym.Env):
    def __init__(self):
        abs_path = os.path.dirname(__file__)
        self.model = load_model_from_path(f'{abs_path}/env_lab2.xml')
        self.time_step = 0.002
        self.n_substeps = 1
        self.time_step *= self.n_substeps
        self.sim = MjSim(self.model, nsubsteps=self.n_substeps)
        self.viewer = None

        # for environment
        self.pre_goal_dist = 3.0
        self.control_freq = 5
        self.num_time_step = int(1.0/(self.time_step*self.control_freq))
        self.limit_distance = 0.4 # LIMIT DISTANCE FROM HAZARD TO ROBOT TO BE A CV
        self.limit_bound = 0.0
        self.goal_dist_threshold = 0.25
        self.h_coeff = 10.0
        self.max_steps = 30000
        self.cur_step = 0
        self.hazard_group = 2
        self.num_group = 6

        # goal coordinates
        self.x_pos = np.arange(0, 6.25, 0.25)
        self.y_pos = np.arange(-2, 2.25, 0.25)

        #self.x_pos = np.arange(0, 7.75, 0.25)
        #self.y_pos = np.arange(-2.25, 2.5, 0.25)

        self.max_scan_value = 3
        self.max_goal_dist = 5
        
        # For PID control Determined by pid_simulation.ipynb (For control freq 5)
        self.p =   181.4158772303444
        self.d =   0.014068658299230652
        self.p_a = 850.8275149846955
        self.d_a = 0.02747346154346146

        # for state
        self.angle_resolution = 1
        self.angle_interval = 5
        self.angle_range = np.arange(-120.0, 120.0 + self.angle_resolution, self.angle_resolution)
        self.lidar_state_dim = int(len(self.angle_range)/self.angle_interval)
        self.scan_value = np.zeros(self.lidar_state_dim, dtype=np.float32)
        self.robot_pose = np.zeros(3)
        self.robot_vel = np.zeros(2)
        self.pre_robot_vel = np.zeros(2)
        self.goal_pos = np.zeros(2)

        # domain randomization variance params
        self.control_noise = 0#.2 #0.1
        self.lidar_noise = 0#.1 #0.05
        self.pose_noise = 0#.2 #0.1
        self.distribution = "normal" #"normal"
        self.random_orientation = True
        self.random_env_positions = True
        self.lab_testing = False

        # for action
        self.action = np.zeros(2)
        self.speed_reduction = 0.0
        self.ang_speed_reduction = 0.0

        # moving obstacle params
        if self.random_env_positions:
            self.amp = 0.5
            self.w = 0.8
        else:
            self.amp = 0
            self.w = 0

        # state & action dimension
        self.action_dim = 2
        # GOAL DIR: 2, GOAL DIST: 1, VEL: 2 (LINEAR, ANGULAR), ACC: 1, SCAN: X, 
        self.state_dim = 2 + 1 + len(self.robot_vel) + 1 + len(self.scan_value)
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.state_dim), np.inf*np.ones(self.state_dim), dtype=np.float32)


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
        if self.distribution == "normal":
            self.scan_value *= 1 + np.random.normal(0, self.lidar_noise)
        else:
            self.scan_value *= 1 + np.random.uniform(-self.lidar_noise, self.lidar_noise)
        return deepcopy(self.scan_value)

    def getState(self):
        self.sim.forward()
        sensor_dict = self.getSensor()
        self.robot_vel[0] = sensor_dict['velocimeter'][0]
        self.robot_vel[1] = sensor_dict['gyro'][2]
        robot_acc = np.array([self.robot_vel[0] - self.pre_robot_vel[0]])*self.control_freq
        self.pre_robot_vel = deepcopy(self.robot_vel)

        self.robot_pose = self.sim.data.get_body_xpos('robot').copy()
        #self.robot_pose *= 1 + np.random.normal(0, self.pose_noise)

        robot_mat = self.sim.data.get_body_xmat('robot').copy()
        theta = Rotation.from_matrix(robot_mat).as_euler('zyx', degrees=False)[0]
        self.robot_pose[2] = theta

        rel_goal_pos = self.goal_pos - self.robot_pose[:2]
        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        rel_goal_pos = np.matmul(rot_mat, rel_goal_pos)
        goal_dist = np.linalg.norm(rel_goal_pos)
        goal_dir = rel_goal_pos/(goal_dist + 1e-8)
        if self.distribution == "normal":
            goal_dir *= 1 + np.random.normal(0, self.pose_noise)
        else:
            goal_dir *= 1 + np.random.uniform(-self.pose_noise, self.pose_noise)

        vel = deepcopy(self.robot_vel)
        scan_value = self.getLidar()
        state = {'goal_dir':goal_dir,
                'goal_dist':goal_dist,
                'vel':vel,
                'acc':robot_acc,
                'scan':scan_value}
        return state

    def getFlattenState(self, state):
        goal_dir = state['goal_dir']
        goal_dist = [state['goal_dist'] / self.max_goal_dist]
        vel = state['vel']
        acc = state['acc']
        scan = 1.0 - (np.clip(state['scan'], 0.0, self.max_scan_value)/self.max_scan_value)
        state = np.concatenate([goal_dir, goal_dist, vel, acc/8.0, scan], axis=0)
        return state

    def build(self):
        self.sim.reset()
        if self.lab_testing:
            self.rob_pos, self.list_obs_pos, self.goal_pos = generate_lab_config()
        else:
            if self.random_env_positions:
                self.rob_pos, self.list_obs_pos, self.goal_pos = generate_random_positions(self.x_pos, self.y_pos)
            else:
                self.list_obs_pos = [(5.5, 2), (2, 0), (5.5, -2), 
                                    (3.5, 1.5), (3.5, -1.5), (4.5, 0), 
                                    (1, -2), (1, 2)]
                self.goal_pos = [[4.5, 2], [4.5, -2], [6, 0]][np.random.randint(3)-1]
            for i in range(len(self.list_obs_pos)):
                self.sim.data.set_joint_qpos('box{}'.format(i+1), [*self.list_obs_pos[i], 0.4, 1.0, 0.0, 0.0, 0.0])
                self.sim.data.set_joint_qvel('box{}'.format(i+1), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.current_goal = 0
        if self.random_orientation:
            random_rotation = random_rotation_quaternion()
            self.sim.data.set_joint_qpos('robot', [self.rob_pos[0], self.rob_pos[1], 0.06344, *random_rotation])
        else:
            self.rob_pos = [0, 0]
            self.sim.data.set_joint_qpos('robot', [0, 0, 0.06344, 0, 0, 0, 0])
        self.sim.data.set_joint_qvel('robot', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        goal_id = self.sim.model.body_name2id('goal')
        self.sim.data.xfrc_applied[goal_id] = [0.0, 0.0, 0.98, 0.0, 0.0, 0.0]

        self.sim.forward()

    def updateGoalPos(self):
        if self.goal_met:
            self.goal_pos = self.rob_pos
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

    def step(self, action):
        target_vel, target_ang_vel = action
        target_vel = (target_vel + 1)/2

        #Speed reduction for more control
        target_vel *= self.speed_reduction
        target_ang_vel *= self.ang_speed_reduction

        for j in range(self.num_time_step):
            self.cur_step += 1
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
            
            # DR: noise added to wheels' speed
            if self.distribution == "normal":
                self.sim.data.ctrl[0] *= 1 + np.random.normal(0, self.control_noise)
                self.sim.data.ctrl[1] *= 1 + np.random.normal(0, self.control_noise)
            else:
                self.sim.data.ctrl[0] *= 1 + np.random.uniform(-self.control_noise, self.control_noise)
                self.sim.data.ctrl[1] *= 1 + np.random.uniform(-self.control_noise, self.control_noise)
            
            if not self.lab_testing:
                # moving obstacle movement definitions
                x1_movement = self.list_obs_pos[-2][0] + self.amp*np.cos(-self.w * self.cur_step/1000)
                y1_movement = self.list_obs_pos[-2][1] + self.amp*np.sin(-self.w * self.cur_step/1000)
                x2_movement = self.list_obs_pos[-1][0] + self.amp*np.cos(self.w * self.cur_step/1000)
                y2_movement = self.list_obs_pos[-1][1] + self.amp*np.sin(self.w * self.cur_step/1000)
                x3_movement = self.list_obs_pos[-3][0] + self.amp*np.sin(self.w * self.cur_step/1000)
                y3_movement = self.list_obs_pos[-3][1] + self.amp*np.cos(self.w * self.cur_step/1000)
                self.sim.data.set_joint_qpos('box'+str(len(self.list_obs_pos)-2), [x1_movement, y1_movement, 0.4 + self.cur_step*0.00004, 1.0, 0.0, 0.0, 0.0])
                self.sim.data.set_joint_qpos('box'+str(len(self.list_obs_pos)-1), [x2_movement, y2_movement, 0.4 + self.cur_step*0.00004, 1.0, 0.0, 0.0, 0.0])
                self.sim.data.set_joint_qpos('box'+str(len(self.list_obs_pos)), [x3_movement, y3_movement, 0.4 + self.cur_step*0.00004, 1.0, 0.0, 0.0, 0.0])
            self.sim.step()
        
        state = self.getState()
        done = False
        info = {"goal_met":False, 'cost':0.0, 'num_cv':0}

        # reward
        goal_dist = state['goal_dist']
        reward = self.pre_goal_dist - goal_dist
        #####################################
        #reward = np.clip(reward, -0.05, 0.05)
        #####################################
        self.pre_goal_dist = goal_dist
        if goal_dist < self.goal_dist_threshold:
            reward += 1.0
            if self.goal_met:
                reward += 1.0
                done = True

            info['goal_met'] = True
            self.goal_met = True
            if not done:
                self.updateGoalPos()

        # cv
        num_cv = 0
        hazard_dist = np.min(state['scan'])
        if hazard_dist < self.limit_distance:
            num_cv += 1
        info['num_cv'] = num_cv
        info['cost'] = self.getCost(hazard_dist)

        # done
        wall_contact = False
        for contact_item in self.sim.data.contact:
            name1 = self.sim.model.geom_id2name(contact_item.geom1)
            name2 = self.sim.model.geom_id2name(contact_item.geom2)
            if name1 is None or name2 is None or name1=='floor' or name2=='floor':
                continue
            if collision_conditions(name1, name2):
                wall_contact = True
                break

        """
        NEW WALL CONTACT CONDITION
        """
        if self.cur_step >= self.max_steps or wall_contact:
            done = True
            temp_num_cv = int(max(self.max_steps - self.cur_step, 0) / 70)
            discount_factor = 0.99
            temp_cost = discount_factor*(1 - discount_factor**temp_num_cv)/(1 - discount_factor)
            info['num_cv'] += temp_num_cv
            info['cost'] += temp_cost


        #add raw state
        info['raw_state'] = state
        return self.getFlattenState(state), reward, done, info