# importing the used packages 
from types import SimpleNamespace
from scipy import optimize
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


# importing package to create plots and setting basic, visual settings
import matplotlib.pyplot as plt
import ipywidgets as widgets
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"-"})
plt.rcParams.update({'font.size': 10})

class LaborAdjustmentCosts():
    def __init__(self):
        """Create the model"""
        # namespaces
        par = self.par = SimpleNamespace() # namespace for parameters
        sol = self.sol = SimpleNamespace() # namespace for solutions

        # baseline parameters
        par.eta = 0.5 
        par.w = 1.0 
        par.kappas = [1.0, 2.0]
        
        # Simulation parameters
        par.T = 120 # Planning horizon
        par.K = 10000 # number of shocks to simulate

        # solutions
        sol.lt = np.zeros(len(par.kappas))
        sol.h_values = np.zeros(par.K)
        sol.epsilon = np.zeros(par.T)


        # extended model parameters
        par.rho = 0.90
        par.iota = 0.01
        par.sigma_epsilon = 0.10
        par.R = (1+0.01)**(1/12)
        


    ############## Question 1 ################
    
    def profits(self, kappat, lt): # defining the profits function
        profits = kappat * lt**(1-self.par.eta) - self.par.w * lt #calculating profits given the formula
        return profits
    
    def obj(self, kappat, x): # defining the objective function
        return -self.profits(kappat, x) # negative sign because we want to maximize profits using a minimizer
        
    def find_optimal_lt(self): # defining the function that finds the optimal lt for each value of kappa
        for i,kappat in enumerate(self.par.kappas): # for loop to iterate over the different values of kappa
            
            obj = lambda x: self.obj(kappat, x) #calling the objective function
            
            result = optimize.minimize_scalar(obj, bounds=(0, 2), method='bounded') #using the bounded method to find the optimal lt that maximizes profits
            
            self.sol.lt[i] = result.x #saving the optimal lt in the solution namespace
            
            print(f'For kappa = {kappat:.1f} the optimal labor supply is lt = {self.sol.lt[i]:.2f} which yields profits of {self.profits(kappat, self.sol.lt[i]):.2f} \n') #printing results
            
    ############## Question 2 ################                    

    def calc_H(self, do_print=False):
        h_values = np.zeros(self.par.K) #initializing h-values
        
        np.random.seed(1999)
        for k in range(self.par.K): #for loop to iterate over the different shocks
            epsilon = np.random.normal(-0.5 * self.par.sigma_epsilon**2, self.par.sigma_epsilon, size=self.par.T) # Initialize shock series

            #creatuing variables for the shocks in each period
            lt_prev = 0.0
            kappat_prev = 1.0
            h_k = 0
            
            for t in range(self.par.T):
                #calculating the demand shock in period t
                kappat = np.exp(self.par.rho * np.log(kappat_prev) + epsilon[t])
                
                #calculating the labor supply in period t given the optimal labor supply function
                lt = ((1-self.par.eta) * kappat / self.par.w) ** (1/(self.par.eta))
                
                #calculating profits given the labor supply in period t
                profits = kappat * lt ** (1-self.par.eta) - self.par.w * lt - (lt != lt_prev) * self.par.iota

                #Now calculating the discounted value of the salon in time t
                h_k += self.par.R ** (-t) * profits
                
                #updating lt_prev and kappat_prev for the next iteration
                lt_prev = lt
                kappat_prev = kappat
                
            #saving the discounted value of the salon in the h_values vector for the k'th shock
            h_values[k] = h_k
        
        # now we calculate the average value function H by taking the mean of the h_values vector
        H = np.mean(h_values)
        
        if do_print==True:
            print(f'The ex ante expected value function H is calculated as the mean of the h_values vector. \n')
            print(f'For K={self.par.K} the value of H is {H:.2f}')








    #def kappat(self, t):
     #   self.epsilon = np.random_normal(-0.5*self.par.sigma_epsilon**2, self.par.sigma_epsilon, size=self.par.T)
     #   
     #   if t==0:
     #       self.kappat = np.exp(self.rho * np.log(self.kappat_prev)+self.epsilon[t])
     #       return self.kappat
     #   else:
     #       self.kappat = np.exp(self.rho) * np.log(self.kappat[-1]+self.epsilon[t])
     #       return self.kappat
    #
    #def ex_ante_expected_value(self):
     #   for k in range(self.par.K):
     #   #initialize shocks series
     #   epsilon = np.random_normal(-0.5*self.par.sigma_epsilon**2, self.par.sigma_epsilon, size=self.par.T)
     #   
     #   # finding each shock series
     #   for t in range(self.T):
     #       if t==0:
     #           self.kappat = np.exp(self.rho * np.log(self.kappat_prev)+epsilon[t])
     #           return self.kappat
     #       else:
     #           def kappat(self, t):
     #               return 
     #       # calculating demand shock
     #       
        

    #def demand_shock(self, t):
    #    return self.par.R**(t) * self.par.kappam1 * self.sol.lt[-1]**(1-self.par.eta)
    
    
    
         
    
        
        