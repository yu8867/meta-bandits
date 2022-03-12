import numpy as np
from tqdm import tqdm
from enviroment import get_reward, random_propability, shuffle_all_move,image_show
from metabandit import metabandit

class Simulator:
    def __init__(self, n_sim, steps, k, unsteady):
        self.Step = steps
        self.Sim = n_sim
        self.k = k
        self.unsteady = unsteady

    def simulation(self, agent, higher_agent, L):
        regret = np.zeros((self.Sim, self.Step))
        for sim in tqdm(range(self.Sim)):
            p = random_propability(self.k)
            meta = metabandit(p=p, k=self.k, agent=agent, higher_agent=higher_agent, L=L)
            for step in range(1,self.Step):
                if self.unsteady == 1:
                    if step%2000==0:
                        p = shuffle_all_move(p)
                action = meta.select_arm()
                reward = get_reward(p, action)
                meta.meta_updata(action, reward)
                regret[sim, step] += regret[sim, step-1] + np.max(p) - p[action]

        return regret
    
    
class Main:
    def __init__(self, policy, n_sim, steps, k, unsteady):
        self.Simulator = Simulator(n_sim, steps, k, unsteady)
        self.n_sim = n_sim
        self.steps = steps
        self.k = k
        self.policy = policy
        
    def main(self):
        dic = {}
        for i in self.policy:
            regret = self.Simulator.simulation(i[0],i[1],L=30)
            dic["Meta {} {}".format(i[0].__name__, i[1].__name__)] = regret
        
        image_show(dic, self.n_sim, self.steps, self.k)
        