import argparse
import gym
import numpy as np
from network_models.policy_net import Policy_net
import tensorflow as tf
from infrastructure.environment_wrapper import EnvironmentWrapper
from infrastructure.utils import open_file_and_save

env_wrapper = EnvironmentWrapper()


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='name of gym environment to use', default='CartPole-v0',
                        choices=['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1'])

    parser.add_argument('--modeldir', help='directory from which to load the model', default='trained_models/ppo')
    parser.add_argument('--trajectorydir', help='directory to save the expert trajectory', default='trajectory/ppo')

    parser.add_argument('--iterations', default=100, type=int)

    return parser.parse_args()


def main(args):
    model_name = f"{args.modeldir}/model_{args.env}.ckpt"

    env_wrapper.set_env(args.env)
    env = env_wrapper.get_env()
    env.seed(0)
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, model_name)
        obs = env.reset()

        for iteration in range(args.iterations):  # episode
            observations = []
            actions = []
            run_steps = 0
            while True:
                run_steps += 1
                # prepare to feed placeholder Policy.obs
                obs = np.stack([obs]).astype(dtype=np.float32)

                act, _ = Policy.act(obs=obs, stochastic=True)
                act = np.ndarray.item(act)

                observations.append(obs)
                actions.append(act)

                next_obs, reward, done, info = env.step(act)

                if done:
                    print(run_steps)
                    obs = env.reset()
                    break
                else:
                    obs = next_obs

            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            open_file_and_save(f'{args.trajectorydir}/observations_{args.env}.csv', observations)
            open_file_and_save(f'{args.trajectorydir}/actions_{args.env}.csv', actions)


if __name__ == '__main__':
    args = argparser()
    main(args)
