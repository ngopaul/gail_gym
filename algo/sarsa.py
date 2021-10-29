import math
import numpy as np


def episode(epsilon, theta, max_steps, rnd_seq, env, args, test=False):
    N = args.memory_size
    NUM_TILINGS = args.num_tilings
    NUM_TILES = args.num_tiles
    M = env.action_space.n

    observations = []
    actions = []
    rewards = []

    Q = np.zeros(M)  # action values
    e = np.zeros(N)  # eligibility traces
    F = np.zeros((M, NUM_TILINGS), dtype=np.int32)  # features for each action

    def load_F(observation):
        state_vars = []
        for i, var in enumerate(observation):
            range_ = (env.observation_space.high[i] - env.observation_space.low[i])

            if range_ == float('inf'):
                np.seterr(over='ignore')  # Suppress any additional warnings
                # observation_space_cut_off = env.observation_space.high
                # observation_space_cut_off[observation_space_cut_off == float('inf')] = 1
                # range_ = max(observation_space_cut_off) * 2
                # originally, with CartPole, defaulted to 1
                range_ = 1
            state_vars.append(var / range_ * NUM_TILES)

        for a in range(M):
            F[a] = get_tiles(NUM_TILINGS, state_vars, N, a, rnd_seq)

    def load_Q():
        for a in range(M):
            Q[a] = 0
            for j in range(NUM_TILINGS):
                Q[a] += theta[F[a, j]]

    observation = env.reset()
    load_F(observation)
    load_Q()
    action = np.argmax(Q)  # numpy argmax chooses first in a tie, not random like original implementation
    if np.random.random() < epsilon:
        action = env.action_space.sample()

    step = 0
    while True:
        step += 1

        observations.append(np.stack([observation]).astype(dtype=np.float32))
        actions.append(action)

        e *= args.gamma * args.lambda_val
        for a in range(M):
            v = 0.0
            if a == action:
                v = 1.0

            for j in range(NUM_TILINGS):
                e[F[a, j]] = v

        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        delta = reward - Q[action]
        load_F(observation)
        load_Q()
        next_action = np.argmax(Q)
        if np.random.random() < epsilon:
            next_action = env.action_space.sample()
        if not done:
            delta += args.gamma * Q[next_action]
        if not test:
            theta += args.alpha / NUM_TILINGS * delta * e
        load_Q()
        if done or step > max_steps:
            break
        action = next_action

    observations = np.reshape(observations, newshape=[-1] + list(env.observation_space.shape))
    actions = np.array(actions).astype(dtype=np.int32)

    return step, sum(rewards), observations, actions


# translated from https://web.archive.org/web/20030618225322/http://envy.cs.umass.edu/~rich/tiles.html
def get_tiles(num_tilings, variables, memory_size, hash_value, rnd_seq):
    num_coordinates = len(variables) + 2
    coordinates = [0 for i in range(num_coordinates)]
    coordinates[-1] = hash_value

    qstate = [0 for i in range(len(variables))]
    base = [0 for i in range(len(variables))]
    tiles = [0 for i in range(num_tilings)]

    for i, variable in enumerate(variables):
        qstate[i] = int(math.floor(variable * num_tilings))
        base[i] = 0

    for j in range(num_tilings):
        for i in range(len(variables)):
            if (qstate[i] >= base[i]):
                coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % num_tilings)
            else:
                coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1) % num_tilings) - num_tilings

            base[i] += 1 + (2 * i)
        coordinates[len(variables)] = j
        tiles[j] = hash_coordinates(coordinates, memory_size, rnd_seq)

    return tiles


def hash_coordinates(coordinates, memory_size, rnd_seq):
    total = 0
    for i, coordinate in enumerate(coordinates):
        index = coordinate
        index += (449 * i)
        index %= 2048
        while index < 0:
            index += 2048

        total += rnd_seq[index]

    index = total % memory_size
    while index < 0:
        index += memory_size

    return index