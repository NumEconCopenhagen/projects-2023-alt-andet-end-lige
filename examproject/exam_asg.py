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
        sol = self. sol = SimpleNamespace() #namespace for solutions

        # baseline parameters
        par.eta = 0.5 
        par.w = 1.0 
        par.kappas = [1.0, 2.0]

        # solutions
        sol.lt = np.zeros(len(par.kappas))
        sol.h = np.zeros(par.K)


        # extended model parameters
        par.rho = 0.90
        par.iota = 0.01
        par.sigma_epsilon = 0.10
        par.R = (1+0.01)**(1/12)
        par.kappam1 = 1.0
        par.lm1 = 0.0
        
        # Simulation parameters
        par.T = 120 # Planning horizon
        par.K = 10000 # number of shocks to simulate

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

    def ex_ante_expected_value(self):
        for k in range(self.par.K)
        #initialize shocks series
        epsilon = 

    def demand_shock(self, t):
        return self.par.R**(t) * self.par.kappam1 * self.sol.lt[-1]**(1-self.par.eta)
    
    
    
         
    
        
        