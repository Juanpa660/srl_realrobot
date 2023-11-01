from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym.envs.mujoco import MujocoEnv
from env import Env
import os

abs_path = os.path.dirname(__file__)
model = load_model_from_path(f'{abs_path}/turtle_laser.xml')
sim = MjSim(model, nsubsteps=1)



