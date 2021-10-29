import argparse
import gym
import numpy as np
import tensorflow as tf
from datetime import datetime
from network_models.policy_net import Policy_net
from algo.behavior_clone import BehavioralCloning
from infrastructure.environment_wrapper import EnvironmentWrapper

env_wrapper = EnvironmentWrapper()


def argparser():
    parser = argparse.ArgumentParser()
    # add ability to specify environment
    parser.add_argument('--env', help='name of gym environment to use', default='CartPole-v0',
                        choices=['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1'])

    parser.add_argument('--savedir', help='save directory, i.e. trained_models/bc/sarsa or trained_models/bc/ppo',
                        default='trained_models/bc/sarsa')
    parser.add_argument('--trajectorydir', help='directory with expert trajectory, i.e. trajectory/sarsa or '
                                                'trajectory/ppo',
                        default='trajectory/sarsa')

    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/bc')
    parser.add_argument('--iterations', default=int(1e3), type=int)
    parser.add_argument('--interval', help='save interval', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)

    return parser.parse_args()


def main(args):
    env_wrapper.set_env(args.env)
    env = env_wrapper.get_env()
    env.seed(0)
    Policy = Policy_net('policy', env)
    BC = BehavioralCloning(Policy)
    saver = tf.compat.v1.train.Saver(max_to_keep=args.max_to_keep)
    model_name = f"model_{args.env}"

    observations = np.genfromtxt(f'{args.trajectorydir}/observations_{args.env}.csv')
    actions = np.genfromtxt(f'{args.trajectorydir}/actions_{args.env}.csv', dtype=np.int32)

    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())

        obs = env.reset()
        eval_reward = 0  # do NOT use rewards to update policy
        eval_rewards = [float('nan')]  # allow print at beginning of loop

        inp = [observations, actions]

        for iteration in range(args.iterations):  # episode
            # train
            for epoch in range(args.epoch_num):
                # select sample indices in [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=args.minibatch_size)

                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                BC.train(obs=sampled_inp[0], actions=sampled_inp[1])

            # get/print/log BC performance
            if iteration % 100 == 0:
                print(f"Iteration {iteration}. Sum of last iteration's rewards: {sum(eval_rewards)}")
            eval_observations = []
            eval_actions = []
            eval_rewards = []
            v_preds = []
            run_policy_steps = 0
            while True:
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

                act, v_pred = Policy.act(obs=obs, stochastic=False)

                act = np.ndarray.item(act)
                v_pred = np.ndarray.item(v_pred)

                eval_observations.append(obs)
                eval_actions.append(act)
                eval_rewards.append(eval_reward)
                v_preds.append(v_pred)
                next_obs, eval_reward, done, info = env.step(act)
                if done:
                    obs = env.reset()
                    eval_reward = -1
                    break
                else:
                    obs = next_obs
            writer.add_summary(tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag='episode_reward', simple_value=sum(eval_rewards))])
                               , iteration)

            summary = BC.get_summary(obs=inp[0], actions=inp[1])

            if (iteration + 1) % args.interval == 0:
                saver.save(sess,  f'{args.savedir}/{model_name}.ckpt', global_step=iteration + 1)

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    # Added for compatibility with Tensorflow 2.x
    tf.compat.v1.disable_eager_execution()

    args = argparser()
    main(args)
