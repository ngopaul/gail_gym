#!/usr/bin/python3
import argparse

import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.ppo import PPOTrain
from infrastructure.environment_wrapper import EnvironmentWrapper

env_wrapper = EnvironmentWrapper()


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='name of gym environment to use', default='CartPole-v0',
                        choices=['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1'])
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', help='gamma for PPO', default=0.95, type=float)
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
       
        reward = 0
        rewards = [float('nan')]  # allow print at beginning of loop
        success_num = 0

        render = False
        for iteration in range(args.iterations):
            if iteration % 100 == 0:
                print(f"Iteration {iteration}. Sum of last iteration's rewards: {sum(rewards)}")

            observations = []
            actions = []
            v_preds = []
            rewards = []
            episode_length = 0

            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                episode_length += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.ndarray.item(act)
                v_pred = np.ndarray.item(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

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

            writer.add_summary(tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag='episode_length', simple_value=episode_length)]), iteration)
            writer.add_summary(tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag='episode_reward', simple_value=sum(rewards))]), iteration)

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

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            PPO.assign_policy_parameters()

            inp = [observations, actions, gaes, rewards, v_preds_next]

            # train
            for epoch in range(6):
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
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

        if args.force_save_final_model:
            saver.save(sess, f'{args.savedir}/{model_name}.ckpt')
        writer.close()


if __name__ == '__main__':
    # Added for compatibility with Tensorflow 2.x
    tf.compat.v1.disable_eager_execution()

    args = argparser()
    main(args)
