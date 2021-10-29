#!/usr/bin/python3

import argparse

from infrastructure.environment_wrapper import EnvironmentWrapper
from infrastructure.utils import open_file_and_save
import pickle
from algo.sarsa import *

env_wrapper = EnvironmentWrapper()


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='name of gym environment to use', default='CartPole-v0',
                        choices=['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1'])

    parser.add_argument('--modeldir', help='directory from which to load the model', default='trained_models/sarsa')
    parser.add_argument('--trajectorydir', help='directory to save the expert trajectory', default='trajectory/sarsa')

    parser.add_argument('--initial_epsilon', default=0.1, type=float)  # probability of choosing a random action
    parser.add_argument('--gamma', default=1.0, type=float)  # discount rate
    parser.add_argument('--lambda_val', default=0.9, type=float)  # trace decay rate
    parser.add_argument('--alpha', default=0.5, type=float)  # learning rate
    parser.add_argument('--memory_size', default=30000, type=int)  # memory for storing parameters
    parser.add_argument('--num_tilings', default=10, type=int)
    parser.add_argument('--num_tiles', default=8, type=int)

    parser.add_argument('--iterations', default=200, type=int)
    parser.add_argument('--seed', default=2, type=int)

    return parser.parse_args()


def main(args):
    env_wrapper.set_env(args.env)
    env = env_wrapper.get_env()
    env.seed(0)
    np.random.seed(args.seed)

    theta = None
    rnd_seq = None

    # load model
    with open(f"{args.modeldir}/model_{args.env}", 'rb') as f:
        theta, rnd_seq = pickle.load(f)

    # sample trajectory
    for episode_num in range(args.iterations):
        _, sum_rewards, observations, actions = episode(0, theta, env.spec.max_episode_steps, rnd_seq, env, args, test=True)
        if episode_num % 100 == 0:
            print(f"Episode {episode_num}: rewards={sum_rewards}, num_steps={actions.size}")
        open_file_and_save(f'{args.trajectorydir}/observations_{args.env}.csv', observations)
        open_file_and_save(f'{args.trajectorydir}/actions_{args.env}.csv', actions)


if __name__ == '__main__':
    args = argparser()
    main(args)
