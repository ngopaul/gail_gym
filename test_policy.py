import pickle

import gym
import numpy as np
import tensorflow as tf
import argparse
from network_models.policy_net import Policy_net
from infrastructure.environment_wrapper import EnvironmentWrapper
from algo.sarsa import *
from infrastructure.utils import open_file_and_save

env_wrapper = EnvironmentWrapper()

# TODO add test_policy capability for SARSA model

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='name of gym environment to use', default='CartPole-v0',
                        choices=['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1'])
    parser.add_argument('--modeldir', help='directory of trained models', default='trained_models')

    parser.add_argument('--alg', help='chose algorithm, one of gail/ppo, gail/sarsa, bc/ppo, bc/sarsa, ppo, sarsa',
                        default='gail/sarsa', choices=['gail/ppo', 'gail/sarsa', 'bc/ppo', 'bc/sarsa', 'ppo', 'sarsa'])

    # even with render=True, rendering only happens if the agent reaches a high performance threshold
    parser.add_argument('--no-render', help='never render during training', dest='render', action='store_false')
    parser.set_defaults(render=True)

    parser.add_argument('--model', help='for bc, number of model to test. model.ckpt-[number]', default='')
    parser.add_argument('--logdir', help='log directory', default='log/test')
    parser.add_argument('--iteration', default=int(1e3))
    parser.add_argument('--stochastic', action='store_false')
    parser.add_argument('--seed', default=2, type=int)
    return parser.parse_args()


def main(args):
    env_wrapper.set_env(args.env)
    env = env_wrapper.get_env()
    env.seed(0)
    np.random.seed(args.seed)
    Policy = Policy_net('policy', env)
    saver = tf.compat.v1.train.Saver()
    model_name = f"model_{args.env}"
    render = False

    if args.alg == 'sarsa':
        with open(f"{args.modeldir}/model_{args.env}", 'rb') as f:
            theta, rnd_seq = pickle.load(f)

        with tf.compat.v1.Session() as sess:
            writer = tf.compat.v1.summary.FileWriter(args.logdir + '/' + args.alg, sess.graph)
            sess.run(tf.compat.v1.global_variables_initializer())

            # sample trajectory
            for iteration in range(args.iterations):
                steps, sum_rewards, _, _ = episode(0, theta, env.spec.max_episode_steps, rnd_seq, env,
                                                                args, test=True)
                if iteration % 100 == 0:
                    print(f"Episode {iteration}: rewards={sum_rewards}, num_steps={steps}")

                writer.add_summary(tf.compat.v1.Summary(
                    value=[tf.compat.v1.Summary.Value(tag='episode_length', simple_value=steps)])
                                   , iteration)
                writer.add_summary(tf.compat.v1.Summary(
                    value=[tf.compat.v1.Summary.Value(tag='episode_reward', simple_value=sum_rewards)])
                                   , iteration)
            writer.close()
        return

    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(args.logdir+'/'+args.alg, sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())
        if args.model == '':
            saver.restore(sess, f"{args.modeldir}/{args.alg}/{model_name}.ckpt")
        else:
            saver.restore(sess, f"{args.modeldir}/{args.alg}/{model_name}.ckpt-{args.model}")
        obs = env.reset()
        reward = 0
        success_num = 0

        for iteration in range(args.iteration):
            rewards = []
            run_policy_steps = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, _ = Policy.act(obs=obs, stochastic=args.stochastic)

                act = np.ndarray.item(act)

                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)
                if render:
                    env.render()
                if done:
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            # end condition of test
            if sum(rewards) >= 195:
                success_num += 1
                render = args.render
                # do not break here, finish the test
                # break
            else:
                success_num = 0

        writer.close()


if __name__ == '__main__':
    # Added for compatibility with Tensorflow 2.x
    tf.compat.v1.disable_eager_execution()

    args = argparser()
    main(args)
