# Metropolis algorithm
# The candidate distribution is symmetric

import numpy as np
import math as m
from matplotlib import pyplot as plt

class Metropolis(object):
    def __init__(self, df, scaling_factor, init_value = 1):
        self.df = df
        self.scaling_factor = scaling_factor
        self.init_value = init_value

    def base_function(self, x):
        #inverse-χ2 distribution(inverse 카이제곱 분포)
        result = m.pow(x, -self.df/2)*m.exp(-self.scaling_factor/(2*x))
        return result

    def uni_dist(self, low, high):
        #uniform distribution : symmetric
        candidate_x = np.random.uniform(low, high)
        if candidate_x == 0:
            self.uni_dist(low, high)
        return candidate_x
    
    def  acceptance(self, candidate_x, pre_x):
        #probability(alpha)
        alpha = self.base_function(candidate_x)/self.base_function(pre_x)
        probability = min(alpha, 1)
        return probability

    def sampling(self, number):
        n = np.arange(1,number+1)
        arr=[]
        for i in range(number):
            cpoint = self.uni_dist(0, 100)
            alpha = self.acceptance(cpoint, self.init_value)
            if alpha == 1:
                self.init_value = cpoint
            else:
                if np.random.random() <= alpha:
                    self.init_value = cpoint
            arr.append(self.init_value)

        plt.plot(n,arr)
        plt.xlabel('number of sampling')
        plt.ylabel('sample')
        plt.title('Metropolis algorithm')
        plt.show()

        



if __name__ == "__main__":     
    ma = Metropolis(5,4)
    ma.sampling(500)
