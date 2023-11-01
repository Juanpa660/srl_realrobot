# ===== add python path ===== #
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
# =========================== #
# python main.py --name Train_Tbot --seed 1 --env_name Turtlebot-v0 --test
# sudo rmmod nvidia_uvm
# sudo modprobe nvidia_uvm

#CPU: for 8k eps 17h 8m
#GPU: for 8k eps 10h 14m

from utils.vectorize import SingleEnvWrapper
from utils.logger import Logger
from utils import register
#from utils.env import Env

from agent import Agent

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import argparse
import random
import wandb
import torch
import time
from pynput import keyboard # for real world training
import rospy
from tbot_real_env import Tbot, degree_to_rad

t0 = time.time()

def getPaser():
    parser = argparse.ArgumentParser(description='TRC')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--test',  action='store_true', help='test or train?')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--name', type=str, default='TRC', help='save name.')
    parser.add_argument('--save_freq', type=int, default=int(3000), help='# of time steps for save.')
    parser.add_argument('--total_steps', type=int, default=int(30000000), help='total training steps.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--fine_tune_training',  action='store_true', help='use fine tuning on last layer?')
    parser.add_argument('--load_model', action='store_true', help='load a pretrained model?')
    parser.add_argument('--model_name', type=str, default="model", help='pretrained model name')
    # for env
    parser.add_argument('--max_episode_steps', type=int, default=2000, help='# of maximum episode steps.')
    parser.add_argument('--n_steps', type=int, default=1500, help='update after collecting n_steps.')
    parser.add_argument('--load_ep', type=int, default=10000000, help='model after # training episodes.')
    # for network 
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function. ReLU, Tanh, Sigmoid...')
    parser.add_argument('--out_activation', type=str, default='sigmoid', help='activation function. sigmoid, tanh...')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the number of hidden layer\'s node.')
    parser.add_argument('--log_std_init', type=float, default=-1.0, help='log of initial std.')
    # for RL
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--n_epochs', type=int, default=64, help='update epochs.')
    parser.add_argument('--gae_coeff', type=float, default=0.97, help='gae coefficient.')
    parser.add_argument('--ent_coeff', type=float, default=0, help='entropy coefficient.')
    # trust region
    parser.add_argument('--damping_coeff', type=float, default=0.01, help='damping coefficient.')
    parser.add_argument('--num_conjugate', type=int, default=10, help='# of maximum conjugate step.')
    parser.add_argument('--line_decay', type=float, default=0.8, help='line decay.')
    parser.add_argument('--max_kl', type=float, default=0.001, help='maximum kl divergence.')
    # constraint
    parser.add_argument('--cost_d', type=float, default=0.025, help='constraint limit value.')
    parser.add_argument('--cost_alpha', type=float, default=0.125, help='CVaR\'s alpha.')
    parser.add_argument('--goal_x', type=float, default=3, help='x position of goal')
    parser.add_argument('--goal_y', type=float, default=0, help='y position of goal')
    parser.add_argument('--speed_reduction', type=float, default=0.5, help='max speed reduction')
    parser.add_argument('--ang_speed_reduction', type=float, default=0.9, help='max angular speed reduction')
    parser.add_argument('--angle_resolution', type=float, default=5, help='angle agent resolution')
    parser.add_argument('--ros_freq_rate', type=int, default=15, help=' ROS frequency rate.')
    return parser

def real_world_train(args):
    # wandb
    if args.wandb:
        project_name = 'Tbot Real Lab Environment'
        wandb.init(
            project=project_name, 
            config=args,
        )
        run_idx = wandb.run.name.split('-')[-1]
        wandb.run.name = f"{args.name}-{run_idx}"

    # for random seed
    np.random.seed(args.seed)
    random.seed(args.seed)    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define Environment
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
    args.wheel_radius = 0.035
    args.wheels_separation = 0.23
    # SETTINGS
    args.goal_pos = [args.goal_x, args.goal_y]
    args.max_scan_value = 3
    args.max_goal_dist = 5
    args.limit_distance = 0.45
    args.localize = False
    args.num_time_steps = 1 # int(1/(self.time_steps*self.control_freq))
    # define agent
    agent = Agent(args)
    agent.load(args.model_name)
    # define Tbot in real world and get state
    rospy.init_node('tbot_control', anonymous=True)
    tbot = Tbot(args)
    state = tbot.reset()
    # uses manual localization
    tbot.set_initial_pose(0, 0, 0)
    print("Initial position set")
    time.sleep(1)

    env_cnts = 0
    total_step = 0
    models_saved = 0

    episode = 0
    ep_rewards = []
    ep_costs = []
    ep_cvs = []
    ep_lens = []
    ep_len = 0

    while episode <= 150:
        trajectories = []
        step = 0
        while step < args.n_steps:
            total_step += 1
            step += 1
            env_cnts += 1
            with torch.no_grad():
                obs_tensor = torch.tensor(state, device=args.device, dtype=torch.float32)
                action_tensor, clipped_action_tensor = agent.getAction(obs_tensor, False)
                actions = action_tensor.detach().cpu().numpy()
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
            next_observations, rewards, dones, info = tbot.step(clipped_action)
            next_observations = np.expand_dims(next_observations, axis=0)
            ep_rewards.append(rewards)
            ep_cvs.append(info['num_cv'])
            ep_costs.append(info['cost'])
            ep_len += 1
            fail = env_cnts < args.max_episode_steps if dones else False
            dones = True if env_cnts >= args.max_episode_steps else dones
            next_observation = state[0] if dones else next_observations[0]
            trajectories.append([state[0], actions[0], rewards, info['cost'], dones, fail, next_observation])

            if dones:
                if args.wandb:
                    wandb_log = {"Reward": np.sum(ep_rewards),
                            "CV": np.sum(ep_cvs),
                            "Cost": np.sum(ep_costs),
                            "Episode Length": np.sum(ep_lens), 
                            "Time (hours)": (time.time()-t0)/3600}
                    wandb.log(wandb_log, step=episode)
                print()
                print("Episode:", episode, "Rewards:", np.sum(ep_rewards), "Steps:", total_step,
                    "CVs:", np.sum(ep_cvs), "Costs:", np.sum(ep_costs))
                ep_rewards = []
                ep_cvs = []
                ep_costs = []
                ep_lens = []
                # reset position
                ep_lens.append(ep_len)
                ep_len = 0
                episode += 1
                env_cnts = 0
                state = tbot.reset()
                tbot.set_initial_pose(0, 0, 0)

            if info["stop_flag"]:
                break
            state = next_observations

        if info["stop_flag"]:
            break
        # ==================================== #
        if args.fine_tune_training:
            print("Training...")
            v_loss, cost_v_loss, cost_var_v_loss, objective, cost_surrogate, kl, entropy, optim_case = agent.train(trajectories)
            if total_step % args.save_freq == 0 or total_step >= args.total_steps:
                models_saved += 1
                print("Episode:", episode, "Model saved:", models_saved)
                agent.save(models_saved)


if __name__ == "__main__":
    parser = getPaser()
    args = parser.parse_args()
    # ==== processing args ==== #
    # save_dir
    args.save_dir = "results"
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_idx}"
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')
    args.device = device

    # ========================= #
    real_world_train(args)
