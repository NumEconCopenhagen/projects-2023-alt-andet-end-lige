from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class OptimalTaxation: 
    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. parameters 
        par.alpha = 0.5 
        par.kappa = 1.0 
        par.nu = 1/(2*16**2)
        par.w = 1.0 
        par.w_vec = np.linspace(0.5,1.5,100)
        par.tau = 0.3
        par.G = 1.0
        
        #c. solution
        sol.L = np.nan

    def calc_utility(self, L):
        """ utility function """
        par = self.par
        sol = self.sol

        #a. consumption of market goods
        C = par.kappa+(1-par.tau)*par.w*L


        #c. utility gain from total consumption
        utility = np.log(C**par.alpha*par.G**(1-par.alpha))

        #d. disutility from work 
        disutility = (par.nu*L**2)/2        

        #e. total utility
        
        return utility - disutility
    
    def solve(self, do_print=False):
        """ solve model """
        par = self.par
        sol = self.sol

        # a. objective function 
        obj = lambda x: -self.calc_utility(x)
        
        # b. initial guess
        initial_guess = 24 #all hours are spent working 

        #c. bounds and constraints 
        bounds = [(0,24)]
        constraints = ({'type': 'ineq', 'fun': lambda x: 24-x})

        #d. call solver 
        results = optimize.minimize(obj, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-08)

        #e. Setting the solution equal to the solution namespace:
        sol.L = results.x

        #f. Printing result
        if do_print:
            print(f'L = {sol.L:.2f}')
    
    def solve_w_vec(self, do_print=False):
        """ solve model for different wage rates """
        par = self.par
        sol = self.sol

        # a. create empty list
        sol.L_vec = []

        # b. loop over wage rates
        for w in par.w_vec:
            par.w = w
            self.solve()
            sol.L_vec.append(sol.L)
        
        if do_print:
            print(sol.L_vec)
    
    def plot_L(self):
        """ plot labour supply as a function of wage rate """
        par = self.par
        sol = self.sol

        #Solution for different wage rates 
        self.solve_w_vec()  
        
        # Plot L as a function of wage rate
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(par.w_vec, sol.L_vec)
        ax.set_xlabel('Wage rate')
        ax.set_ylabel('Hours worked')
        plt.title(r'$L^{\star}(\tilde{w})$')
        plt.grid(True)
        plt.show()
    
    def ext_government(self): 
        """solve model for different government consumption"""
       
        par = self.par
        sol = self.sol

        # a. Define new government consumption
        G = par.tau * par.w * sol.L * ((1 - par.tau) * par.w)
        par.G = G

        #b. solve model with new government consumption
        self.solve() 
    
    def implied_G(self): 
        """Calculate implied government consumption"""

        par = self.par
        sol = self.sol

        return par.tau * par.w * sol.L * ((1 - par.tau) * par.w)
    
    
    def plot_3d(self): 
        """ plot 3d graph of utility function as a function of hours worked and implied government consumption for a grid of tau"""

        par = self.par
        sol = self.sol

        taus = np.linspace(0,1,100)

        utility = np.empty((100))

        for tau in enumerate(taus):
                utility[i] = self.ext_government(tau)

