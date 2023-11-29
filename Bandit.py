import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Any

class Bandit:
    def __init__(self, k=10, num_problems=2000):
        self.k = k
        self.num_problems = num_problems
        self.q_star = np.random.normal(0, 1, (num_problems, k))
        self.arms = [0] * k

        for i in range(10):
            self.arms[i] = np.random.normal(self.q_star[0, i], 1, 2000)  # First problem as a sample
        plt.figure(figsize=(12,8))
        plt.ylabel('Rewards distribution')
        plt.xlabel('Actions')
        plt.xticks(range(1,11))
        plt.yticks(np.arange(-5,5,0.5))

        plt.violinplot(self.arms, positions=range(1,11), showmedians=True)
        plt.show()

    def bandit(self, action, problem):
        return np.random.normal(self.q_star[problem, action], 1)

    def simple_max(self, Q, N, t):
        return np.random.choice(np.flatnonzero(Q == Q.max()))  # Breaking ties randomly

    def simple_bandit(self, epsilon, steps, initial_Q, alpha=0, argmax_func=None):
        if argmax_func is None:
            argmax_func = self.simple_max

        rewards = np.zeros(steps)
        actions = np.zeros(steps)

        for i in tqdm(range(self.num_problems)):
            Q = np.ones(self.k) * initial_Q  # Initial Q
            N = np.zeros(self.k)  # Initialize number of rewards given
            best_action = np.argmax(self.q_star[i])
            for t in range(steps):
                if np.random.rand() < epsilon:  # Explore
                    a = np.random.randint(self.k)
                else:  # Exploit
                    a = argmax_func(Q, N, t)

                reward = self.bandit(a, i)

                N[a] += 1
                if alpha > 0:
                    Q[a] = Q[a] + (reward - Q[a]) * alpha
                else:
                    Q[a] = Q[a] + (reward - Q[a]) / N[a]

                rewards[t] += reward

                if a == best_action:
                    actions[t] += 1

        return np.divide(rewards, self.num_problems), np.divide(actions, self.num_problems)

    @staticmethod
    def ucb(Q, N, t):
        c = 2
        if N.min() == 0:
            return np.random.choice(np.flatnonzero(N == N.min()))

        M = Q + c * np.sqrt(np.divide(np.log(t),N))
        return np.argmax(M) # breaking ties randomly

    def plot_results(self, ep_0, ep_01, ep_1, ac_0, ac_01, ac_1, opt_0, ac_opt_0, ucb_2, ac_ucb_2):
        plt.figure(figsize=(12, 6))
        plt.plot(ep_0, 'g', label='epsilon = 0')
        plt.plot(ep_01, 'r', label='epsilon = 0.01')
        plt.plot(ep_1, 'b', label='epsilon = 0.1')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.plot(ac_0, 'g', label='epsilon = 0')
        plt.plot(ac_01, 'r', label='epsilon = 0.01')
        plt.plot(ac_1, 'b', label='epsilon = 0.1')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12,6))
        plt.yticks(np.arange(0,3,0.2))
        plt.plot(ac_1, 'r', label='Realistic')
        plt.plot(ac_opt_0, 'b', label='Optimistic')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12,6))
        plt.plot(ep_1, 'g', label='e-greedy e=0.1')
        plt.plot(ucb_2, 'b', label='ucb c=2')
        plt.legend()
        plt.show()

    def run_experiments(self):
        #epsilon - greedy action selection
        ep_0, ac_0 = self.simple_bandit(epsilon=0, steps=1000, initial_Q=0)
        ep_01, ac_01 = self.simple_bandit(epsilon=0.01, steps=1000, initial_Q=0)
        ep_1, ac_1 = self.simple_bandit(epsilon=0.1, steps=1000, initial_Q=0)
        #optimistic initial value selection
        opt_0, ac_opt_0 = self.simple_bandit(epsilon=0, steps=1000, initial_Q=5, alpha=0.2)
        #Upper-Confidence-Bound Action selection
        ucb_2, ac_ucb_2 = self.simple_bandit(epsilon=0, steps=1000, initial_Q=0, argmax_func=Bandit.ucb)

        self.plot_results(ep_0, ep_01, ep_1, ac_0, ac_01, ac_1, opt_0, ac_opt_0, ucb_2, ac_ucb_2)
