from agent import Agent
import numpy as np
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

# RS
class RS(Agent):
    def __init__(self, k, opt, aleph):
        super().__init__(k)
        self.count = 0
        self.reward_sum = 0
        self.aleph = 1
        self.alpha = 0.0005

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        RS = (self.arm_counts/ self.count)*(self.E - self.aleph)
        return np.argmax(RS)

    def update(self, action, reward):
        self.count += 1
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward
        self.reward_sum += reward

        a = 1/(self.arm_counts[action] + 1)
        self.E[action] = (1- a) * self.E[action] + a * reward
        self.aleph += self.alpha * (self.E[action] - self.aleph)

# RS OPT
class RS_OPT(Agent):
    def __init__(self, k, opt, aleph):
        super().__init__(k)
        self.count = 0
        self.reward_sum = 0
        self.opt = opt

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        RS = (self.arm_counts/ self.count)*(self.E - self.opt)
        return np.argmax(RS)

    def update(self, action, reward):
        self.count += 1
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward
        self.reward_sum += reward

        alpha = 1/(self.arm_counts[action] + 1)
        self.E[action] = (1- alpha) * self.E[action] + alpha * reward

# Tompson Sampling
class TS(Agent):
    def __init__(self, k, opt, aleph):
        super().__init__(k)
        self.count = 0
        self.reward_sum = 0
        self.success = np.zeros(k)
        self.fail = np.zeros(k)
        self.mu = np.zeros(k)

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        return np.argmax(self.mu)

    def update(self, action, reward):
        self.count += 1
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward
        self.reward_sum += reward

        if reward == 1:
          self.success[action] += 1
        else:
          self.fail[action] += 1
        self.E[action] = ((self.arm_counts[action]-1)/self.arm_counts[action])*self.E[action] + (1/self.arm_counts[action]) * reward
        self.mu = np.array([np.random.beta(self.success[action]+1, self.fail[action]+1) for action in range(self.k)])

# UCB1T
class UCB1T(Agent):
    def __init__(self, k, opt, aleph):
        super().__init__(k)
        self.count = 0
        self.reward_sum = 0
        self.var = np.zeros(k)
        self.ucb = np.zeros(k)
        self.c = 1.0

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        return np.argmax(self.ucb)

    def update(self, action, reward):
        self.count += 1
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward
        self.reward_sum += reward
        # self.E[action] = self.E[action] + (reward - self.E[action])/self.arm_counts[action]
        self.E[action] = (1/(self.arm_counts[action]+1))*(self.arm_counts[action]*self.E[action] + reward)

        self.var[action] += ((reward - self.E[action])**2) / self.arm_counts[action]
        v = self.var + np.sqrt(2*np.log(self.count)/self.arm_counts)
        min = []
        for i in range(self.k):
          if v[i]>1/4:
            min.append(1/4)
          else:
            min.append(v[i])       
        self.ucb = self.E + self.c*np.sqrt((np.log(self.count)/self.arm_counts)*min)

# Greedy
class Greedy(Agent):
    def __init__(self, k, opt, aleph):
      super().__init__(k)
      self.count = 0
      self.reward_sum = 0
      self.opt = opt

    def reset_params(self):
      super().recet_params()

    def select_arm(self):
      max_values = np.where(self.E ==self.E.max())[0]
      return np.random.choice(max_values)

    def update(self, action, reward):
      self.count += 1
      self.arm_counts[action] += 1
      self.arm_rewards[action] += reward
      self.reward_sum += reward

      self.E[action] = self.E[action] + (reward - self.E[action])/self.arm_counts[action]

# SRS
class SRS(Agent):
    def __init__(self, k, opt, aleph):
      super().__init__(k)
      self.count = 0
      self.reward_sum = 0
      self.epsilon = 10**-4
      self.aleph = aleph
      self.pi = np.array([1/k]*k)

    def reset_params(self):
      super().recet_params()

    def select_arm(self):
      return np.random.choice(len(self.pi), p=self.pi)

    def update(self, action, reward):
      self.count += 1
      self.arm_counts[action] += 1
      self.arm_rewards[action] += reward
      self.reward_sum += reward
      self.E[action] = self.E[action] + (reward - self.E[action])/self.arm_counts[action]
      # a = 1/(self.arm_counts[action] + 1)
      # self.E[action] = (1- a) * self.E[action] + a * reward

      E_max = self.E.max()
      tmp_E = self.E
      if E_max > self.aleph:
        tmp_E = tmp_E - E_max + self.aleph - self.epsilon
      
      Z = 1/(np.sum(1/(self.aleph - tmp_E)))
      row = Z/(self.aleph - tmp_E)
      b = (self.arm_counts / row) - self.count + self.epsilon

      SRS = (self.count + b.max()) * row - self.arm_counts
      self.pi = SRS/np.sum(SRS)

# SRS OPT
class SRS_OPT(Agent):
    def __init__(self, k, opt, aleph):
      super().__init__(k)
      self.count = 0
      self.reward_sum = 0
      self.epsilon = 10**-4
      self.opt = opt
      self.pi = np.array([1/k]*k)

    def reset_params(self):
      super().recet_params()

    def select_arm(self):
      return np.random.choice(len(self.pi), p=self.pi)

    def update(self, action, reward):
      self.count += 1
      self.arm_counts[action] += 1
      self.arm_rewards[action] += reward
      self.reward_sum += reward
      self.E[action] = self.E[action] + (reward - self.E[action])/self.arm_counts[action]
      # a = 1/(self.arm_counts[action] + 1)
      # self.E[action] = (1- a) * self.E[action] + a * reward

      E_max = self.E.max()
      tmp_E = self.E
      if E_max > self.opt:
        tmp_E = tmp_E - E_max + self.opt - self.epsilon
      
      Z = 1/(np.sum(1/(self.opt - tmp_E)))
      row = Z/(self.opt - tmp_E)
      b = (self.arm_counts / row) - self.count + self.epsilon

      SRS = (self.count + b.max()) * row - self.arm_counts
      self.pi = SRS/np.sum(SRS)

# RS CH
class RS_CH(Agent):
    def __init__(self, k, opt, aleph):
      super().__init__(k)
      self.count = 0
      self.reward_sum = 0
      self.pi = np.array([1/k]*k)

    def reset_params(self):
      super().recet_params()

    def select_arm(self):
      return np.argmax(self.pi)

    def update(self, action, reward):
        self.count += 1
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward
        self.reward_sum += reward

        a = 1/(self.arm_counts[action] + 1)
        self.E[action] = (1- a) * self.E[action] + a * reward

        E_max = self.E.max()
        Dkl  =self. E*np.log(self.E/E_max) + (1-self.E)*np.log((1-self.E)/(1-E_max))
        myu = np.exp((-1)*self.arm_counts*Dkl)
        aleph = np.nan_to_num(E_max*((1-(self.E/E_max)*myu)/(1-myu)),nan=0)
        aleph_max = max(aleph)

        self.pi = (self.arm_counts/self.count)*(self.E - aleph_max)

# SRS CH
class SRS_CH(Agent):
    def __init__(self, k, opt, aleph):
      super().__init__(k)
      self.count = 0
      self.reward_sum = 0
      self.epsilon = 10**-4
      self.pi = np.array([1/k]*k)

    def reset_params(self):
      super().recet_params()

    def select_arm(self):
      return np.random.choice(len(self.pi), p=self.pi)

    def update(self, action, reward):
      self.count += 1
      self.arm_counts[action] += 1
      self.arm_rewards[action] += reward
      self.reward_sum += reward
      a = 1/(self.arm_counts[action] + 1)
      self.E[action] = (1- a) * self.E[action] + a * reward

      E_max = self.E.max()
      Dkl  =self. E*np.log(self.E/E_max) + (1-self.E)*np.log((1-self.E)/(1-E_max))
      myu = np.exp((-1)*self.arm_counts*Dkl)
      aleph = np.nan_to_num(E_max*((1-(self.E/E_max)*myu)/(1-myu)),nan=0)
      aleph_max = max(aleph)

      tmp_E = self.E
      if E_max > aleph_max:
        tmp_E = tmp_E - E_max + aleph_max - self.epsilon
      
      Z = 1/(np.sum(1/(aleph_max - tmp_E)))
      row = Z/(aleph_max - tmp_E)
      b = (self.arm_counts / row) - self.count + self.epsilon

      SRS = (self.count + b.max()) * row - self.arm_counts
      self.pi = SRS/np.sum(SRS)