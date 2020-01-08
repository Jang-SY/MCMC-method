# Simulated Annealing
# For situations where the target distribution may have multiple peaks

import numpy as np
import math as m
from matplotlib import pyplot as plt

class SA(object):
    def __init__(self, df, scaling_factor, tot_num, step = 1, init_value = 1):
        self.df = df
        self.scaling_factor = scaling_factor
        self.tot_num = tot_num
        self.step = step
        self.init_value = init_value

    def base_function(self, x):
        #inverse-X2 distribution(inverse 카이제곱 분포)
        result = m.pow(x, -self.df/2)*m.exp(-self.scaling_factor/(2*x))
        return result

    def uni_dist(self, low, high):
        #uniform distribution : symmetric
        candidate_x = np.random.uniform(low, high)
        if candidate_x == 0:
            self.uni_dist(low, high)
        return candidate_x

    def temp(self, itemp, ftemp):
        #simulated annealing
        val = itemp * m.pow((ftemp/itemp),self.step/self.tot_num)
        ctemp = max(val, ftemp)
        return ctemp


    def  acceptance(self, candidate_x, pre_x):
        #probability(alpha)
        alpha = m.pow((self.base_function(candidate_x)/self.base_function(pre_x)), 1/self.temp(100, 1))
        probability = min(alpha, 1)
        return probability

    def sampling(self):
        n = np.arange(1,self.tot_num+1)
        arr=[]
        for i in range(self.tot_num):
            cpoint = self.uni_dist(0, 100)
            alpha = self.acceptance(cpoint, self.init_value)
            if alpha == 1:
                self.init_value = cpoint
            else:
                if np.random.random() <= alpha:
                    self.init_value = cpoint
            self.step += 1
            arr.append(self.init_value)

        plt.plot(n,arr)
        plt.xlabel('number of sampling')
        plt.ylabel('sample')
        plt.title('Simulated Annealing')
        plt.show()

        



if __name__ == "__main__":     
    sa = SA(5,4,500)
    sa.sampling()
