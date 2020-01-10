# Gibbs Sampling
# A special case of Metropolis-Hastings sampling wherein the random value is always accepted (i.e. α =1)


import numpy as np
import math as m
from matplotlib import pyplot as plt

class Gibbs(object):
    def __init__(self, n, alpha, beta, x0 = None, y0 = None):
        #초기값으로 x0와 y0 한 곳에만 값 입력
        #x0는 정수, y0는 0~1사이의 값
        self.n = n
        self.alpha = alpha
        self.beta = beta
        if x0 != None:
            if int(x0) != x0:
                print('x는 정수를 입력해야 함')
            else:
                self.x = x0
        else:
            self.x = x0
        if y0 != None:
            if y0 < 0 or y0 > 1:
                print('y는 0~1 사이의 수 입력해야 함')
            else:
                self.y = y0
        else:
            self.y = y0

    def x_given_y(self,y):
        #y값이 주어졌을 때 x sample, binomial distribution
        result = np.random.binomial(self.n, y)
        return result

    def y_given_x(self,x):
        #x값이 주어졌을 때 y sample, beta distribution
        result = np.random.beta(x + self.alpha, self.n - x + self.beta)
        return result

    def sampling(self, number):
        self.arr_x=[]
        self.arr_y=[]

        for i in range(number):
            if i == 0:
                if self.x != None:
                    self.arr_x.append(self.x)
                elif self.y != None:
                    self.arr_y.append(self.y)

            if len(self.arr_x) < len(self.arr_y):
                 x = self.x_given_y(self.y)
                 self.arr_x.append(x)
                 if i != number-1:    
                    self.y = self.y_given_x(x)
                    self.arr_y.append(self.y)

            else:
                y = self.y_given_x(self.x)
                self.arr_y.append(y)
                if i != number-1:    
                    self.x = self.x_given_y(y)
                    self.arr_x.append(self.x)

        plt.figure(1)
        plt.scatter(self.arr_x,self.arr_y)
        plt.scatter(self.arr_x[number-1], self.arr_y[number-1],color='r')
        plt.xlabel('x value')
        plt.ylabel('y value')
        plt.title('Gibbs Sampling')
        plt.show()


# Using the Gibbs Sampler to Approximate Marginal Distributions
# Monte-Carlo (MC) estimate
    def MC_estimate(self):
        x_bar = np.mean(self.arr_x)
        y_bar = np.mean(self.arr_y)
        print('MC_estimate = ', x_bar,',', y_bar)

# A better approach is to use the average of the conditional densities p(x | y = yi), 
    def marginal_x(self, x):
        base = 0
        for i in range(len(self.arr_y)):
            base += m.factorial(self.n)*((m.pow(self.arr_y[i],x)*m.pow(1-self.arr_y[i],self.n-x))/(m.factorial(x)*m.factorial(self.n-x)))

        base /= len(self.arr_y)
        return base

    def graph_x(self):
        n = []
        value = []
        for i in range(len(self.arr_y)):
            n.append(i)
            value.append(self.marginal_x(i))

        plt.figure(2)
        plt.plot(n, value)
        plt.show()

# A better approach is to use the average of the conditional densities p(y | x = xi), 
    def marginal_y(self, y):
        base = 0
        for i in range(len(self.arr_x)):
            base += m.pow(y,self.arr_x[i]+self.alpha-1)*m.pow(1-y,self.n-self.arr_x[i]+self.beta-1)

        base /= len(self.arr_x)
        return base

    def graph_y(self):
        n = []
        value = []
        n = range(0,1,0.1)
        for i in range(len(n)):
            value.append(self.marginal_y(n[i]))

        plt.figure(3)
        plt.plot(n, value)
        plt.show()

if __name__ == "__main__":     
    gibbs = Gibbs(100,2,5,y0=0.5)
    gibbs.sampling(100)
    gibbs.graph_x()
    gibbs.graph_y()
    gibbs.MC_estimate()
    
    
