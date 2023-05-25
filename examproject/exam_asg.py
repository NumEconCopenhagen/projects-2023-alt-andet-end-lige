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
        sol = self.sol = SimpleNamespace() #namespace for solutions

        # baseline parameters
        par.eta = 0.5 
        par.w = 1.0 

        # solutions
        sol.l = np.nan



    
        # extended model parameters
        par.rho = 0.90
        par.iota = 0.01
        par.sigma = 0.10
        par.R = (1+0.01)**(1/12)
        
        
        