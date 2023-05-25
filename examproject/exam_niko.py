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

    def calc_utility(self, L, extention = False):
        """ utility function """
        par = self.par
        sol = self.sol

        #a. consumption of market goods
        C = par.kappa+(1-par.tau)*par.w*L

        #b. Handle case when C is zero
        if C == 0:
            C = 1e-10

        #c. utility gain from total consumption
        utility = np.log(C**par.alpha*par.G**(1-par.alpha))

        #d. disutility from work 
        if L == 0:
            disutility = 0  # Set disutility to zero when L is zero
        else:
            disutility = (par.nu * L**2) / 2   

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
    
    def plot_results(self, tau_grid):
        """ plot results for a grid of tau values """
        par = self.par
        sol = self.sol

        # Create empty lists to store results
        L_vec = []
        G_vec = []
        utility_vec = []

        # Loop over tau values
        for tau in tau_grid:
            # Set new tau value
            par.tau = tau

            # Solve model
            self.ext_government()

            # Append results to lists
            L_vec.append(sol.L)
            G_vec.append(par.tau * par.w * sol.L * ((1 - par.tau) * par.w))
            utility_vec.append(self.calc_utility(sol.L))

        # Plotting
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))

        # Labor supply (L) plot
        ax[0].plot(tau_grid, L_vec)
        ax[0].set_xlabel('Tau')
        ax[0].set_ylabel('Labor Supply (L)')
        ax[0].set_title('Labor Supply as a Function of Tau')

        # Government consumption (G) plot
        ax[1].plot(tau_grid, G_vec)
        ax[1].set_xlabel('Tau')
        ax[1].set_ylabel('Government Consumption (G)')
        ax[1].set_title('Government Consumption as a Function of Tau')

        # Utility plot
        ax[2].plot(tau_grid, utility_vec)
        ax[2].set_xlabel('Tau')
        ax[2].set_ylabel('Worker Utility')
        ax[2].set_title('Worker Utility as a Function of Tau')

        plt.tight_layout()
        plt.show()

    def optimal_tax_cd(self): 
        """ find socially optimal tax rate maximizing worker utility """
        
        par = self.par
        sol = self.sol

        # Objective function
        obj = lambda tau: -self.calc_utility(tau)

        # Initial guess
        initial_guess = 0.5  # Initial guess for tau

        # Bounds and constraints
        bounds = [(0, 1)]  # Tau must be within (0, 1)

        # Call optimizer
        results = optimize.minimize(obj, initial_guess, method='SLSQP', bounds=bounds, tol=1e-08)

         # Get the optimal tax rate
        tau_star = results.x[0]

        return tau_star

