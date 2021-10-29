#!/usr/bin/python3
import argparse
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain
from infrastructure.environment_wrapper import EnvironmentWrapper

env_wrapper = EnvironmentWrapper()


def argparser():
    parser = argparse.ArgumentParser()
    # add ability to specify environment
    parser.add_argument('--env', help='name of gym environment to use', default='CartPole-v0',
                        choices=['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1'])
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory, i.e. trained_models/gail/sarsa or trained_models/gail/ppo',
                        default='trained_models/gail/sarsa')
    parser.add_argument('--trajectorydir', help='directory with expert trajectory, i.e. trajectory/sarsa or '
                                                'trajectory/ppo',
                        default='trajectory/sarsa')
    parser.add_argument('--gamma', help='gamma for PPO', default=0.95)
    parser.add_argument('--iterations', help='number of training iterations', default=int(1e3), type=int)

    # even with render=True, rendering only happens if the agent reaches a high performance threshold
    parser.add_argument('--no-render', help='never render during training', dest='render', action='store_false')
    parser.set_defaults(render=True)

    parser.add_argument('--force_save_final_model', help="force save the last iteration's model",
                        dest='force_save_final_model', action='store_true')
    parser.set_defaults(force_save_final_model=False)

    return parser.parse_args()


def main(args):
    env_wrapper.set_env(args.env)
    env = env_wrapper.get_env()
    env.seed(0)
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    D = Discriminator(env)

    expert_observations = np.genfromtxt(f'{args.trajectorydir}/observations_{args.env}.csv')
    expert_actions = np.genfromtxt(f'{args.trajectorydir}/actions_{args.env}.csv', dtype=np.int32)

    saver = tf.compat.v1.train.Saver()
    model_name = f"model_{args.env}"

    success_threshold = None
    if args.env == 'CartPole-v0':
        success_threshold = 195  # solved value according to OpenAI
    elif args.env == 'MountainCar-v0':
        success_threshold = -110  # solved value according to OpenAI
    elif args.env == 'Acrobot-v1':
        success_threshold = -100  # no solved value, but this is pretty hard to get

    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())

        obs = env.reset()
        reward = 0  # do NOT use rewards to update policy
        rewards = [float('nan')]  # allow print at beginning of loop
        success_num = 0
        render = False

        for iteration in range(args.iterations):
            if iteration % 100 == 0:
                print(f"Iteration {iteration}. Sum of last iteration's rewards: {sum(rewards)}")
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            while True:
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.ndarray.item(act)
                v_pred = np.ndarray.item(v_pred)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                next_obs, reward, done, info = env.step(act)
                if render:
                    env.render()
                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if sum(rewards) >= success_threshold:
                success_num += 1
                render = args.render
                if success_num >= 100:
                    saver.save(sess, f'{args.savedir}/{model_name}.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0
                render = False

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            # train discriminator
            for i in range(2):
                D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)

            # output of this discriminator is reward
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()
            for epoch in range(6):
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])

            writer.add_summary(summary, iteration)
        writer.close()

        if args.force_save_final_model:
            saver.save(sess, f'{args.savedir}/{model_name}.ckpt')


if __name__ == '__main__':
    # Added for compatibility with Tensorflow 2.x
    tf.compat.v1.disable_eager_execution()

    args = argparser()
    main(args)
