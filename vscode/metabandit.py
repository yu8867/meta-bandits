from agent import Agent
from enviroment import greedy
import copy as cp

class metabandit(Agent):
    def __init__(self, p, k, agent, higher_agent, L, delta=0, lmd=30): # ２つの方策が必要
        super().__init__(k)
        self.k = k
        self.alert = 0
        self.step = 0
        # RS
        p = sorted(p, reverse=True)
        self.aleph = p[0]
        self.opt = (p[0] + p[1]) / 2
        # agent
        self.copy_agent = agent(k, self.opt, self.aleph)
        self.old_agent = agent(k, self.opt, self.aleph)
        self.new_agent = agent(k, self.opt, self.aleph)
        # agent_copy
        self.copy_higher_agent = higher_agent(2, self.opt, self.aleph)
        self.higher_agent = higher_agent(2, self.opt, self.aleph)
        # new or old
        self.select_agent = 0
        # param
        self.delta = 0
        # detection
        self.lmd = lmd
        self.L = L
        self.l_count = 0
        self.mt_sum = 0
        self.MT = 0

    def reset_params(self):
        self.alert = 0
        self.l_count = 0
        self.mt_sum = 0
        self.MT = 0

    def select_arm(self):

        if self.alert:
        # 1--> new,    0--> old
            self.select_agent = self.higher_agent.select_arm()
            if self.select_agent ==0:
                return  self.old_agent.select_arm()
            else:
                return  self.new_agent.select_arm()
        else:
            return  self.old_agent.select_arm()

    def meta_updata(self, action, reward):
        self.step += 1

        if self.alert==1: # True
            self.l_count+=1
            self.higher_agent.update(self.select_agent, reward)
            self.old_agent.update(action, reward)
            self.new_agent.update(action, reward)

            if self.l_count == self.L:
                self.reset_params()

                if greedy(self.higher_agent.arm_rewards) == 1:
                    self.old_agent = self.new_agent

        else: # False
            self.old_agent.update(action, reward)
            average = self.old_agent.reward_sum / self.old_agent.count
            mt = reward - average + self.delta
            self.mt_sum += mt

            if self.mt_sum > self.MT:
                self.MT = self.mt_sum
            PHt = self.MT - self.mt_sum

            # 超えたらアラーム: True
            if PHt > self.lmd:
                self.alert = 1
                self.new_agent = cp.deepcopy(self.copy_agent)
                self.higher_agent = cp.deepcopy(self.copy_higher_agent)