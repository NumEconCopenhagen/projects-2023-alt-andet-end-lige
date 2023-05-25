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


        # extended model parameters
        par.rho = 0.90
        par.iota = 0.01
        par.sigma = 0.10
        par.R = (1+0.01)**(1/12)
        par.kappam1 = 1.0
        par.lm1 = 0.0
        
    
    #def supply(lt, yt):
      #  lt = yt
      #  return yt
    
    #def demand(self, kappat, yt):
       # pt = kappat * yt**(-self.par.eta)
       # return pt
       

    
    def profits(self, kappat, lt):
        profits = kappat * lt**(1-self.par.eta) - self.par.w * lt
        return profits
    
    def obj(self, kappat, x):
        return -self.profits(kappat, x)
        
    def find_optimal_lt(self):
        for i,kappat in enumerate(self.par.kappas): # for loop to iterate over the different values of kappa
            # calculating optimal lt given our formula
            
            obj = lambda x: self.obj(kappat, x)
            
            result = optimize.minimize_scalar(obj, bounds=(0, 2), method='bounded')
            
            self.sol.lt[i] = result.x
            
            print(f'For kappa = {kappat:.1f} the optimal labor supply is lt = {self.sol.lt[i]:.2f} which yields profits of {self.profits(kappat, self.sol.lt[i]):.2f} \n')
            
            
            
            #optimal_lt = ((1-self.par.eta)*kappat/self.par.w)**(1/(self.par.eta))
            #self.sol.lt = optimal_lt 
            
            #calculating profits given the optimal lt
            #profits = self.profits(self, kappat)
            #print(f'For kappa = {kappat:.1f}, optimal lt = {self.sol.lt:.2f}, profits = {profits:.2f}')
    
        
        