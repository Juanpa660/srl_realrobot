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


from utils.vectorize import SingleEnvWrapper
from utils.logger import Logger
from utils import register
from utils.env import Env

from agent import Agent

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import argparse
import random
import wandb
import torch
import time

t0 = time.time()

def getPaser():
    parser = argparse.ArgumentParser(description='TRC')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--test',  action='store_true', help='test or train?')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--name', type=str, default='TRC', help='save name.')
    parser.add_argument('--save_freq', type=int, default=int(100000), help='# of time steps for save model.')
    parser.add_argument('--total_steps', type=int, default=int(30000000), help='total training steps.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--fine_tune_training',  action='store_true', help='use fine tuning on last layer?')
    parser.add_argument('--load_model', action='store_true', help='load a pretrained model?')
    parser.add_argument('--model_name', type=str, default="model", help='pretrained model name')
    # for env
    parser.add_argument('--env_name', type=str, default='Turtlebot-hard-v0', help='gym environment name.')
    parser.add_argument('--env_difficulty', type=str, default='hard', help='gym environment difficulty.')
    parser.add_argument('--max_episode_steps', type=int, default=2000, help='# of maximum episode steps.')
    parser.add_argument('--n_steps', type=int, default=4000, help='update after collecting n_steps.')
    parser.add_argument('--load_ep', type=int, default=10000000, help='model after # training episodes.')
    # for networks
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function. ReLU, Tanh, Sigmoid...')
    parser.add_argument('--out_activation', type=str, default='sigmoid', help='activation function. sigmoid, tanh...')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the number of hidden layer\'s node.')
    parser.add_argument('--log_std_init', type=float, default=-1.0, help='log of initial std.')
    # for RL
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor.')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate.')
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
    return parser

def train(args):
    # wandb
    if args.wandb:
        project_name = 'Tbot Final Env'
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
    vec_env = SingleEnvWrapper(Env(args.env_name, args.seed, args.max_episode_steps, args.env_difficulty))

    # set args value for env
    args.obs_dim = vec_env.observation_space.shape[0]
    args.action_dim = vec_env.action_space.shape[0]
    args.action_bound_min = vec_env.action_space.low
    args.action_bound_max = vec_env.action_space.high

    # define agent
    agent = Agent(args)
    if args.load_model:
        agent.load_pretrained(args.model_name)

    # train
    observations = vec_env.reset()
    env_cnts = 0
    total_step = 0
    models_saved = 0

    episode = 0
    ep_rewards = []
    ep_costs = []
    ep_cvs = []
    ep_lens = []
    ep_len = 0
    wandb_ep_save = 50
    while total_step < args.total_steps:

        # ======= collect trajectories ======= #
        trajectories = []
        step = 0
        while step < args.n_steps:
            env_cnts += 1
            step += 1
            total_step += 1

            with torch.no_grad():   
                obs_tensor = torch.tensor(observations, device=args.device, dtype=torch.float32)
                action_tensor, clipped_action_tensor = agent.getAction(obs_tensor, True)
                actions = action_tensor.detach().cpu().numpy()
                clipped_actions = clipped_action_tensor.detach().cpu().numpy()
            next_observations, rewards, dones, info = vec_env.step(clipped_actions)
            ep_rewards.append(rewards)
            ep_cvs.append(info['num_cv'])
            ep_costs.append(info['cost'])
            ep_len += 1

            fail = env_cnts < args.max_episode_steps if dones else False
            dones = True if env_cnts >= args.max_episode_steps else dones
            next_observation = info['terminal_observation'] if dones else next_observations[0]
            trajectories.append([observations[0], actions[0], rewards, info['cost'], dones, fail, next_observation])

            if dones:
                if episode % wandb_ep_save == 0:
                    # WANDB SAVE METRICS
                    if args.wandb:
                        wandb_log = {"Reward": np.sum(ep_rewards)/wandb_ep_save,
                                "CV": np.sum(ep_cvs)/wandb_ep_save,
                                "Cost": np.sum(ep_costs)/wandb_ep_save,
                                "Episode Length": np.sum(ep_lens)/wandb_ep_save, 
                                "Time (hours)": (time.time()-t0)/3600}
                        wandb.log(wandb_log, step=episode)
                    ep_rewards = []
                    ep_cvs = []
                    ep_costs = []
                    ep_lens = []
                
                ep_lens.append(ep_len)
                ep_len = 0
                episode += 1
                env_cnts = 0

            observations = next_observations
        # ==================================== #
        v_loss, cost_v_loss, cost_var_v_loss, objective, cost_surrogate, kl, entropy, optim_case = agent.train(trajectories)

        if total_step % args.save_freq == 0 or total_step >= args.total_steps:
            models_saved += 1
            print("Episode:", episode, "Model saved:", models_saved)
            agent.save(models_saved)
    print("Total time: ", time.time()-t0)

def test(args):
    # define Environment
    env = Env("Turtlebot-medium-v0", args.seed, args.max_episode_steps, args.env_difficulty)
    # set args value for env
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.action_bound_min = env.action_space.low
    args.action_bound_max = env.action_space.high

    # define agent
    agent = Agent(args)
    agent.load(args.load_ep)
    scores = []
    cvs = []
    epochs = 100
    for epoch in range(epochs):
        state = env.reset()
        done = False
        score = 0
        cv = 0
        step = 0

        while True:
            step += 1
            with torch.no_grad():
                obs_tensor = torch.tensor(state, device=args.device, dtype=torch.float32)
                action_tensor, clipped_action_tensor = agent.getAction(obs_tensor, False)
                action = action_tensor.detach().cpu().numpy()
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
            next_state, reward, done, info = env.step(clipped_action)
            env.render()
            state = next_state
            score += reward
            cv += info['num_cv']

            if done or step >= args.max_episode_steps:
                break

        scores.append(score)
        cvs.append(cv)
        print("Ep:", epoch, 'Score:', score, 'CV:', cv)
    
    print(np.mean(scores), np.mean(cvs))
    env.close()

if __name__ == "__main__":
    parser = getPaser()
    args = parser.parse_args()
    # ==== processing args ==== #
    # save_dir
    args.save_dir = f"results/{args.name}_s{args.seed}"
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

    if args.test:
        test(args)
    else:
        train(args)
