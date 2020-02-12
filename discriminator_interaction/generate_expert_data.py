"""
Generate expert trajectories as numpy dataset from a trained policy.
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.append('..')
import TD3

from manipulator_learning.sim.envs.thing_reaching import ThingReachingXYState, ThingReachingXYImage
from manipulator_learning.sim.envs.thing_pushing import ThingPushingXYState, ThingPushingXYImage

parser = argparse.ArgumentParser()
parser.add_argument('--environment', type=str, default="ThingPushingXYState")
parser.add_argument('--data_directory', type=str, default='expert_data')
parser.add_argument('--expert_directory', type=str, default='../models')
parser.add_argument('--num_demos', type=int, default=500)
parser.add_argument('--num_timesteps', type=int, default=-1)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

if args.num_timesteps > 0:
    use_timesteps = True
    exp_dir_name = args.environment + '_' + str(args.num_timesteps) + 'steps'
else:
    use_timesteps = False
    exp_dir_name = args.environment + '_' + str(args.num_demos) + 'demos'

expert_path = os.path.join(args.data_directory, exp_dir_name)
os.makedirs(expert_path, exist_ok=True)

env = getattr(sys.modules[__name__], args.environment)()
policy = TD3.Actor(state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0],
                 max_action=float(env.action_space.high[0]))
policy.load_state_dict(torch.load(
    args.expert_directory + '/TD3_' + args.environment + '_' + str(args.seed) + '_actor'))

np_filename = expert_path + '/data.npy'
traj_lens_filename = expert_path + '/traj_lens.npy'
if os.path.isfile(np_filename):
    data = np.load(np_filename)
    total_ts = len(data)
else:
    data = None
    total_ts = 0
if os.path.isfile(traj_lens_filename):
    traj_lens = np.load(traj_lens_filename)
    num_demos = len(traj_lens)
else:
    traj_lens = None
    num_demos = 0

while((use_timesteps and total_ts < args.num_timesteps) or
      (not use_timesteps and num_demos < args.num_demos)):
    obs = env.reset()
    done = False
    traj_data = []
    ts = 0
    ep_r = 0

    while not done:
        with torch.no_grad():
            act = policy(torch.FloatTensor(obs.reshape(1, -1)))
        next_obs, rew, done, info = env.step(act.numpy().flatten())
        ep_r += rew
        traj_data.append(np.concatenate((obs, np.array(act).flatten(), np.array([rew]))))
        ts += 1
        total_ts += 1
        obs = next_obs

    num_demos += 1
    if data is None:
        data = np.array(traj_data)
    else:
        data = np.concatenate((data, np.array(traj_data)))
    if traj_lens is None:
        traj_lens = np.array([ts])
    else:
        traj_lens = np.concatenate((traj_lens, [ts]))
    np.save(np_filename, data)
    np.save(traj_lens_filename, traj_lens)

    if use_timesteps:
        print('%d timesteps out of %d generated.' % (total_ts, args.num_timesteps))
    else:
        print('%d demos out of %d generated (%d timesteps).' % (num_demos, args.num_demos, total_ts))