import numpy as np
import random
import matplotlib.pyplot as plt

def get_reward(p, action):
    if np.random.random() < p[action]:
        return 1
    else:
        return 0

def greedy(values):
    max_values = np.where(values ==values.max())[0]
    return np.random.choice(max_values)

def random_propability(k):
    return np.random.rand(k)

def shuffle_all_move(items):
    rand_ord = random.sample(range(len(items)), k=len(items))
    return [items[r] for i, r in sorted(zip(rand_ord, rand_ord[1:]+rand_ord[:1]))]

def image_show(regret_policy, n_sim, steps ,k):
    plt.figure(figsize=(10,6))
    plt.xlabel("step")
    plt.ylabel("regret")
    for policy, regret in regret_policy.items():
        plt.plot(np.mean(regret, axis=0), label="Meta {}".format(policy))
    plt.title("n_sim={}, steps={}, arm_size={}".format(n_sim, steps, k))
    plt.legend()
    plt.show()

