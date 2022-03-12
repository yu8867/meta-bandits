import numpy  as np

class Agent():
    def __init__(self, k):
        self.k = k
        self.arm_counts = np.zeros(k)
        self.arm_rewards = np.zeros(k)
        self.E = np.array([0.5]*k)

    def recet_params(self):
        self.arm_counts = np.zeros(self.k)
        self.arm_rewards = np.zeros(self.k)
        self.E = np.array([0.5]*self.k)

    def select_arm(self):
        pass

    def update(self, selected, reward):
        pass