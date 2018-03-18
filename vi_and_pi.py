### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    policy: np.array
        The policy to evaluate. Maps states to actions.
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns
    -------
    value function: np.ndarray
        The value function from the given policy.
    """
    ############################
    # initialize value function
    V = np.zeros(nS)
    V_prev = np.copy(V)
    iter_counter = 0
    tol_curr = np.inf
    # loop over Bellman updates:
    while iter_counter <= max_iteration and tol_curr >= tol:
        for s in range(nS):
            p_sp_r_t_lst = P[s][policy[s]]
            exp_r_v = 0
            for p_sp_r_t in p_sp_r_t_lst:
                p, sp, r, t = p_sp_r_t
                exp_r_v += p * (r + gamma * V_prev[sp])
            V[s] = exp_r_v
        tol_curr = np.abs(V - V_prev).max()
        V_prev = np.copy(V)
        iter_counter += 1
    ############################
    return V


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new policy: np.ndarray
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """
    ############################
    V = value_from_policy
    q_np = np.zeros((nS, nA), dtype='float')
    policy_next = np.zeros(nS, dtype='int')
    
    for s in range(nS):
        # argmax Q(s, a)
        # Q(s, a) = E[R(s, a) + gamma * V(s')]
        for a in P[s].keys():
            p_sp_r_t_lst = P[s][a]
            exp_r_v = 0.
            for p_sp_r_t in p_sp_r_t_lst:
                p, sp, r, t = p_sp_r_t
                exp_r_v += p * (r + gamma * V[sp])
            q_np[s, a] = exp_r_v
    policy_next = q_np.argmax(axis=1)
    ############################
    return policy_next


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """Runs policy iteration.

    You should use the policy_evaluation and policy_improvement methods to
    implement this method.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    iter_counter = 0
    policy_change = np.inf
    while iter_counter == 0 or policy_change > 0:
        value_from_policy = policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=max_iteration, tol=tol)
        policy_next = policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9)
        policy_change = np.abs((policy_next - policy).max())
        iter_counter += 1
        del policy
        policy = np.copy(policy_next)
    V = policy_evaluation(P, nS, nA, policy_next, gamma=0.9, max_iteration=max_iteration, tol=tol)
    ############################
    return V, policy_next

def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    V_next = np.zeros(nS, dtype=int)
    q_np = np.zeros((nS, nA), dtype='float')
    iter_counter = 0
    tol_curr = np.inf
    while iter_counter <= max_iteration and tol_curr >= tol:
        for s in range(nS):
            for a in range(nA):                
                p_sp_r_t_lst = P[s][a]
                exp_r_v = 0.
                for p_sp_r_t in p_sp_r_t_lst:
                    p, sp, r, t = p_sp_r_t
                    exp_r_v += p * (r + gamma * V[sp])
                q_np[s, a] = exp_r_v
        V_next = q_np.max(axis=1)
        tol_curr = np.abs(V_next - V).max()
        del V
        V = np.copy(V_next)
        iter_counter += 1
    policy = q_np.argmax(axis=1)
    ############################
    return V, policy

def example(env):
    """Show an example of gym
    Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
    """
    env.seed(0);
    from gym.spaces import prng; prng.seed(10) # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render();

def render_single(env, policy):
    """Renders policy once on environment. Watch your agent play!

        Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
        Policy: np.array of shape [env.nS]
            The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.5) # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
        raw_input()
    assert done
    env.render();
    print "Episode reward: %f" % episode_reward


def print_policy(policy):
	a2str_dict = {
		0: "<",
		1: "v",
		2: ">",
		3: "^"

	}
	policy_str_lst = np.asarray(
		map(lambda x: a2str_dict[x], policy),
		dtype='str'
		)
	print(policy_str_lst.reshape(4, 4)) 

# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    # print env.__doc__
    print "Here is an example of state, action, reward, and next state"
    # example(env)
    P_test = env.P
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)

    print_policy(p_vi)
    print_policy(p_pi)
    # render_single(env, p_vi)
    # render_single(env, p_pi)
    print("stochastic")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=1000000, tol=1e-6)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=1000000, tol=1e-6)
    
    print_policy(p_vi)
    print_policy(p_pi)
    # print(V_vi.reshape(4, 4))
    # print(V_pi.reshape(4, 4))

    # render_single(env, p_vi)
    # render_single(env, p_pi)

