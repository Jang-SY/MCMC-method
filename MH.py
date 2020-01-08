# Metropolis-Hasting algorithm
# The candidate distribution isn't symmetric
# Simply drawing from a χ2 distribution independent of the current position

import numpy as np
import math as m
from matplotlib import pyplot as plt

class MH(object):
    def __init__(self, df, init_value = 1):
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
    
    def chisquare_dist(self):
        #χ2 distribution : unsymmetric
        candidate_x = np.random.chisquare(self.df)
        if candidate_x == 0:
            self.chisquare_dist()
        return candidate_x

    def  acceptance(self, candidate_x, pre_x):
        #probability(alpha)
        alpha = (self.base_function(candidate_x)*self.chisquare_func(pre_x)) / (self.base_function(pre_x)*self.chisquare_func(candidate_x))
        probability = min(alpha, 1)
        return probability

    def sampling(self, number):
        n = np.arange(1,number+1)
        arr=[]
        for i in range(number):
            cpoint = self.chisquare_dist()
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
        plt.title('Metropolis Hasting algorithm')
        plt.show()

        



if __name__ == "__main__":     
    mh = MH(10)
    mh.sampling(500)
