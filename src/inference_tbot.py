#!/usr/bin/env python3

import glob
import sys
import os
PATH = os.getcwd()
for dir_idx, dir_name in enumerate(PATH.split('/')):
    dir_path = '/'.join(PATH.split('/')[:(dir_idx+1)])
    file_list = [os.path.basename(sub_dir) for sub_dir in glob.glob(f"{dir_path}/.*")]
    if '.trc_package' in file_list:
        PATH = dir_path
        break
if not PATH in sys.path:
    sys.path.append(PATH)

from agent import Agent
import numpy as np
import argparse
import random
import torch
import time
from tbot_real_env import Tbot, degree_to_rad
import rospy

t0 = time.time()

def getPaser():
    parser = argparse.ArgumentParser(description='TRC')
    # common
    parser.add_argument('--name', type=str, default='TRC', help='save name.')
    parser.add_argument('--n_steps', type=int, default=4000, help='number of steps in training.')
    parser.add_argument('--model_name', type=str, default='model', help='model name.')
    parser.add_argument('--fine_tune_training',  action='store_true', help='use fine tuning on last layer?')
    # for networks
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function. ReLU, Tanh, Sigmoid...')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the number of hidden layer\'s node.')
    parser.add_argument('--log_std_init', type=float, default=-1.0, help='log of initial std.')
    parser.add_argument('--out_activation', type=str, default='tanh', help='activation function of the output.')
    # for RL
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor.')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate.')
    parser.add_argument('--n_epochs', type=int, default=200, help='update epochs.')
    parser.add_argument('--gae_coeff', type=float, default=0.97, help='gae coefficient.')
    parser.add_argument('--ent_coeff', type=float, default=0.0, help='gae coefficient.')
    # trust region
    parser.add_argument('--damping_coeff', type=float, default=0.01, help='damping coefficient.')
    parser.add_argument('--num_conjugate', type=int, default=10, help='# of maximum conjugate step.')
    parser.add_argument('--line_decay', type=float, default=0.8, help='line decay.')
    parser.add_argument('--max_kl', type=float, default=0.001, help='maximum kl divergence.')
    # constraint
    parser.add_argument('--cost_d', type=float, default=25.0/1000.0, help='constraint limit value.')
    parser.add_argument('--cost_alpha', type=float, default=0.125, help='CVaR\'s alpha.')
    parser.add_argument('--goal_x', type=float, default=2, help='x position of goal')
    parser.add_argument('--goal_y', type=float, default=2, help='y position of goal')
    parser.add_argument('--speed_reduction', type=float, default=0.5, help='max speed reduction')
    parser.add_argument('--ang_speed_reduction', type=float, default=0.9, help='max angular speed reduction')
    parser.add_argument('--angle_resolution', type=float, default=5, help='angle agent resolution')
    parser.add_argument('--ros_freq_rate', type=int, default=15, help=' ROS frequency rate.')
    parser.add_argument('--max_scan_value', type=float, default=3, help=' max scan value.')
    parser.add_argument('--max_goal_dist', type=float, default=5, help=' max goal dist.')
    
    return parser


if __name__ == "__main__":
    parser = getPaser()
    args = parser.parse_args()
    # save_dir
    args.save_dir = "results"
    # device
    args.device = "cpu"
    # ========================= #

    # set args value for env
    args.action_dim = 2 # Linear acc and Angular acc or Target vel/ang_vel
    args.action_bound_min = [-1., -1.]
    args.action_bound_max = [1., 1.]
    # Lidar SICK TIM parameters (obtainable from /scan topic)
    args.angle_min_max = [-2.0943951023931953, 2.0943951023931953] # radians
    args.angle_increment_lidar = 0.0058171823620796
    args.angle_increment_agent = degree_to_rad(args.angle_resolution) # 5 degrees to rad
    # GOAL DIR: 2, GOAL DIST: 1, VEL: 2 (LINEAR, ANGULAR), ACC: 1, SCAN: X
    args.obs_dim = 6 + int((args.angle_min_max[1]-args.angle_min_max[0]) / args.angle_increment_agent)
    # ROBOT PARAMS
    args.wheel_radius = 0.035 # meters CONFIRM!!
    args.wheels_separation = 0.23 # meters CONFIRM!!
    # SETTINGS
    args.goal_pos = [args.goal_x, args.goal_y]
    args.localize = False
    args.num_time_steps = 1 # int(1/(self.time_steps*self.control_freq))
    args.limit_distance = 0.35
    # define agent
    agent = Agent(args)
    agent.load(args.model_name)

    # define Tbot in real world and get state
    rospy.init_node('tbot_control', anonymous=True)
    tbot = Tbot(args)
    state = tbot.reset()
    if args.localize:
        cov = 5
        tbot.call_init_particles_service()
        while cov > 0.12:
            tbot.move_scan(0.2, 0.2, 100)
            tbot.move_robot(0, 0, 5)
            cov = tbot.calculate_covariance()
    else:
        tbot.set_initial_pose(0, 0, 0)
        print("Initial position set")
        time.sleep(1)

    step = 0
    done = False
    t0 = time.time()
    total_reward = 0
    total_cv = 0
    mean_min_scan = 0
    total_dist = 0
    while not done:
        step += 1
        with torch.no_grad():
            obs_tensor = torch.tensor(state, device=args.device, dtype=torch.float32)
            action_tensor, clipped_action_tensor = agent.getAction(obs_tensor, False)
            action = action_tensor.detach().cpu().numpy() # REMOVABLE (it's only for debugging)
            clipped_action = clipped_action_tensor.detach().cpu().numpy()
        next_state, reward, done, info = tbot.step(clipped_action)

        total_reward += reward
        total_cv += info["num_cv"]
        mean_min_scan += info["lidar_min"]
        total_dist += info["vel"] * (time.time()-t0)
        t0 = time.time()
        print(np.round(info["goal_dist"], 2), np.round(info["lidar_min"], 2), total_dist)
        print()

        state = next_state
        if info["stop_flag"]:
            break
    mean_min_scan /= step

    print(total_reward, total_cv, mean_min_scan, total_dist)
    

