#!/usr/bin/python3
"""
Translated from https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp
Algorithm described at https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node89.html
Some minor adjustments to constants were made to make the program work on environments
besides Mountain Car.
"""

import argparse
import os
import gym
from infrastructure.environment_wrapper import EnvironmentWrapper
import pickle
from algo.sarsa import *

env_wrapper = EnvironmentWrapper()


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='name of gym environment to use', default='CartPole-v0',
                        choices=['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1'])
    parser.add_argument('--logdir', help='log directory', default='log/train/sarsa')
    parser.add_argument('--savedir', help='save directory', default='trained_models/sarsa')

    parser.add_argument('--initial_epsilon', default=0.1, type=float)  # probability of choosing a random action
    parser.add_argument('--gamma', default=1.0, type=float)  # discount rate
    parser.add_argument('--lambda_val', default=0.9, type=float)  # trace decay rate
    parser.add_argument('--alpha', default=0.5, type=float)  # learning rate
    parser.add_argument('--memory_size', default=30000, type=int)  # memory for storing parameters
    parser.add_argument('--num_tilings', default=10, type=int)
    parser.add_argument('--num_tiles', default=8, type=int)

    parser.add_argument('--iterations', default=2000, type=int)
    parser.add_argument('--seed', default=2, type=int)

    return parser.parse_args()


def main(args):
    env_wrapper.set_env(args.env)
    env = env_wrapper.get_env()
    env.seed(0)
    np.random.seed(args.seed)

    env = gym.wrappers.Monitor(env, args.logdir, force=True)

    epsilon = args.initial_epsilon
    theta = np.zeros(args.memory_size)  # parameters (memory)
    rnd_seq = np.random.randint(0, 2 ** 32 - 1, 2048)

    for episode_num in range(args.iterations):
        # max_episode_steps may be timestep_limit
        steps_taken, sum_rewards, _, _ = episode(epsilon, theta, env.spec.max_episode_steps, rnd_seq, env, args)
        if episode_num % 100 == 0:
            print(f"Episode: {episode_num}, steps taken: {steps_taken}, sum of rewards: {sum_rewards}")
        epsilon = epsilon * 0.999  # added epsilon decay

    # save model
    print("Saving SARSA model parameters.")
    os.makedirs(f"{args.savedir}", exist_ok=True)
    with open(f"{args.savedir}/model_{args.env}", 'wb') as f:
        pickle.dump((theta, rnd_seq), f)

    env.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
