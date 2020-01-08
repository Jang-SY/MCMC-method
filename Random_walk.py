# Metropolis-Hasting algorithm
# The candidate distribution isn't symmetric
# proposal distribution의 option으로 random walks  y = x + z

import numpy as np
import math as m
from matplotlib import pyplot as plt

class MH_randomwalk(object):
    def __init__(self, df, init_value = 50):
        self.df = df
        self.init_value = init_value

    def base_function(self, x, n = 5, a = 4):
        #inverse-χ2 distribution(inverse 카이제곱 분포)
        result = m.pow(x, -n/2)*m.exp(-a/(2*x))
        return result

    def chisquare_func(self, x):
        #χ2 distribution(카이제곱 분포)
        result = m.pow(x, (self.df/2)-1) * m.exp(-x/2)
        return result
    
    def random_walk(self):
        #y = x + z
        #z는 -1 or 1
        a = round(np.random.random())
        ran_var = 10*(2*a - 1)
        candidate_x = self.init_value + ran_var
        if candidate_x <= 0:
            candidate_x = 1
            self.init_value = 50
        return candidate_x

    def  acceptance(self, candidate_x, pre_x):
        #probability(alpha)
        alpha = (self.base_function(candidate_x)*self.random_walk()) / (self.base_function(pre_x)*self.random_walk())
        probability = min(alpha, 1)
        return probability

    def sampling(self, number):
        n = np.arange(1,number+1)
        arr=[]
        for i in range(number):
            cpoint = self.random_walk()
            print(i, cpoint)
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
        plt.title('Metropolis Hasting (random_walk)')
        plt.show()

        



if __name__ == "__main__":     
    mhr = MH_randomwalk(10)
    mhr.sampling(500)
