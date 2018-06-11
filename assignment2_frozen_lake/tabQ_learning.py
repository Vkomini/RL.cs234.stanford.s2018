import numpy as np
import gym
import time
from lake_envs import *


def learn_Q_QLearning(
    env,
    num_episodes=1000,
    gamma=1.0,
    lr=0.25,
    e=0.8,
    decay_rate=0.99):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function for. Must have nS, nA, and P as
      attributes.
    num_episodes: int 
      Number of episodes of training.
    gamma: float
      Discount factor. Number in range [0, 1)
    learning_rate: float
      Learning rate. Number in range [0, 1)
    e: float
      Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
      Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    Q = np.zeros((env.nS, env.nA))
    policy = np.ones((env.nS, env.nA)) * 1. / env.nA
    policy_idx = np.ones((env.nS, 1)) * np.arange(env.nA)
    return_per_episode_lst = []
    for episode in range(num_episodes):
        s = env.reset()
        terminal = False
        sarsn_lst = []
        return_per_episode = 0.
        if episode % 100 == 0:
            print("episode: " + str(episode))
        while not terminal:
#             env.render()
            a = int(np.random.choice(env.nA, 1, p=policy[s]))
            sn, r, terminal, _ = env.step(a)
            sarsn_lst.append((s, a, r, sn))
            # update Q
            Q[s, a] = Q[s, a] + lr * (r + gamma * Q[sn].max() - Q[s, a])
            # update policy
            policy[s] = e * (decay_rate ** episode) / (env.nA - 1)
            a_star = Q[s].argmax()
            policy[s, a_star] = 1. - e * (decay_rate ** episode)
            
            return_per_episode += r
            s = sn
            
        return_per_episode_lst.append(return_per_episode)
        
#         # update Q
#         for sarsn in sarsn_lst:
#             s, a, r, sn = sarsn
#             Q[s, a] = Q[s, a] + lr * (r + gamma * Q[sn].max()) - Q[s, a]
#         # improve policy
#         actions_best = Q.argmax(axis=1).reshape(-1, 1)
#         policy = np.ones((env.nS, env.nA)) * e * (decay_rate ** episode) / (env.nA - 1)
#         policy[policy_idx == actions_best] = 1. - \
#           policy[policy_idx != actions_best].reshape((env.nS, -1)).sum(axis=1)
    
    ############################

    return Q, policy, return_per_episode_lst


def render_single_Q(env, Q):
    """Renders Q function once on environment. Watch your agent play!

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play Q function on. Must have nS, nA, and P as
        attributes.
      Q: np.array of shape [env.nS x env.nA]
        state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print "Episode reward: %f" % episode_reward

# Feel free to run your own debug code in main!


def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q, policy, return_per_episode_lst = learn_Q_QLearning(env)
    num_episodes = len(return_per_episode_lst)
    return_run_avg_lst = []
    for episode in range(1, num_episodes):
        return_run_avg = (float(sum(return_per_episode_lst[:episode][-100:])) /
                          len(return_per_episode_lst[:episode][-100:])
                         )
        return_run_avg_lst.append(return_run_avg)
#     print(return_run_avg_lst)
#     print(return_per_episode_lst)
    # plt.plot(return_run_avg_lst)
    # plt.show()
    # print(policy)
#     print(policy.argmax(axis=1))
    render_single_Q(env, Q)

if __name__ == '__main__':
    main()
