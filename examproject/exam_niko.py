from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt
import sympy as sm
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
        par.rho = 1.001
        par.sigma = 1.001
        par.epsilon = 1.0
        
        #c. solution
        sol.L = np.nan

    def calc_utility(self, L, extension=False, CES=False):
        """ utility function """
        par = self.par
        sol = self.sol

        # a. consumption of market goods and alternative government consumption
        if CES: 
            par.w = 1.0
            C = par.kappa + (1 - par.tau) * par.w * L
            G = par.tau * par.w * sol.L * ((1 - par.tau) * par.w)
        elif extension:
            par.w = 1.0
            C = par.kappa + (1 - par.tau) * par.w * L
            G = par.tau * par.w * sol.L * ((1 - par.tau) * par.w)
        else: 
            C = par.kappa + (1 - par.tau) * par.w * L

        # b. utility gain from total consumption
        if CES:
            utility = ((((par.alpha * C ** ((par.sigma - 1) / par.sigma)) + 
                         ((1 - par.alpha) * G ** ((par.sigma - 1) / par.sigma))) ** (par.sigma / (par.sigma - 1))) 
                         ** (1 - par.rho) - 1) / (1 - par.rho) #CES utility function
        elif extension:
            utility = np.log(C ** par.alpha * G ** (1 - par.alpha)) #Utility with government consumption
        else:
            utility = np.log(C ** par.alpha * par.G ** (1 - par.alpha)) #Utility 

        # c. disutility from work
        if CES:
            disutility = par.nu * (L ** (1 + par.epsilon) / (1 + par.epsilon))  # Set disutility to zero when L is zero
        else:
            disutility = (par.nu * L ** 2) / 2

        # d. total utility
        return utility - disutility

    
    def solve(self, extension=False, CES=False, do_print=False):
        """ solve model """
        par = self.par
        sol = self.sol

        # a. objective function 
        obj = lambda x: -self.calc_utility(x, extension=extension, CES=CES)
        initial_guess = 24 #all hours are spent working 

        #c. bounds and constraints 
        bounds = [(0,24)]
        constraints = ({'type': 'ineq', 'fun': lambda x: 24-x})

        #d. call solver 
        results = optimize.minimize(obj, initial_guess, method='SLSQP', 
                                    bounds=bounds, constraints=constraints, tol=1e-08)

        #e. Setting the solution equal to the solution namespace:
        sol.L = results.x[0]
    
        #f. Printing result
        if do_print & CES:
            print(f'Optimal labor supply with CES utility function is {sol.L:.2f}')
        elif do_print:
            print(f'Optimal labor supply with baseline parameters is {sol.L:.2f}')
        else: 
            return sol.L
        
        
    
    def solve_w_vec(self, extension=False, do_print=False):
        """ solve model for different wage rates """
        par = self.par
        sol = self.sol

        # a. create empty list
        sol.L_vec = []

        # b. loop over wage rates
        for w in par.w_vec:
            par.w = w
            self.solve(extension=extension)
            sol.L_vec.append(sol.L)

        if do_print:
            print(sol.L_vec)
    
    def plot_L(self, extension=False):
        """ plot labour supply as a function of wage rate """
        par = self.par
        sol = self.sol

        #Solution for different wage rates 
        self.solve_w_vec(extension=extension)  

        # Plot L as a function of wage rate
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(par.w_vec, sol.L_vec)
        ax.set_xlabel('Wage rate')
        ax.set_ylabel('Hours worked')
        plt.title(r'$L^{\star}(\tilde{w})$')
        plt.grid(True)
        plt.show()
    
    def plot_results(self, tau_grid, extension=False, optimal_tax=False):
        """ plot results for a grid of tau values """
        par = self.par
        sol = self.sol

        par.w = 1.0
        
        # Create empty lists to store results
        L_vec = []
        G_vec = []
        utility_vec = []

        # Loop over tau values
        for tau in tau_grid:
            # Set new tau value
            par.tau = tau

            # Solve model
            self.solve(extension=extension)

            # Append results to lists
            L_vec.append(sol.L)
            G_vec.append(par.tau * par.w * sol.L * ((1 - par.tau) * par.w))
            utility_vec.append(self.calc_utility(sol.L, extension=extension))

        # Plotting
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))

        # Labor supply (L) plot
        ax[0].plot(tau_grid, L_vec)
        if optimal_tax:
            self.optimal_tax_cd(extension=extension)
            ax[0].scatter(self.optimal_tax_cd(extension=extension), self.solve(extension=extension), 
                          marker='o', color='red')
        ax[0].set_xlabel('Tau')
        ax[0].set_ylabel('Labor Supply (L)')
        ax[0].set_title('Labor Supply as a Function of Tau')
        ax[0].grid(True)

        # Government consumption (G) plot
        ax[1].plot(tau_grid, G_vec)
        if optimal_tax:
            self.optimal_tax_cd(extension=extension)
            ax[1].scatter(self.optimal_tax_cd(extension=extension), 
                          par.tau * par.w * self.solve(extension=extension) * ((1 - par.tau) * par.w), marker='o', color='red')
        ax[1].set_xlabel('Tau')
        ax[1].set_ylabel('Government Consumption (G)')
        ax[1].set_title('Government Consumption as a Function of Tau')
        ax[1].grid(True)

        # Utility plot
        ax[2].plot(tau_grid, utility_vec)
        if optimal_tax:
            self.optimal_tax_cd(extension=extension)
            ax[2].scatter(self.optimal_tax_cd(extension=extension), 
                          self.calc_utility(self.solve(extension=extension), extension=extension), marker='o', color='red')
        ax[2].set_xlabel('Tau')
        ax[2].set_ylabel('Worker Utility')
        ax[2].set_title('Worker Utility as a Function of Tau')
        ax[2].grid(True)
        
        plt.tight_layout()
        plt.show()

    def optimal_tax_cd(self, extension=False, do_print=False):
        """Find the tax rate that maximizes worker utility"""
        par = self.par
        sol = self.sol

        def obj(tau):
            """Objective function to maximize worker utility"""
            self.par.tau = tau
            self.solve(extension=extension)
            return -self.calc_utility(self.sol.L, extension=extension)

        # Call optimizer to maximize the objective function
        results = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')

        # Get the optimal tax rate
        tau_star = results.x

        # Do print
        if do_print:
            print(f'The tax rate that maximizes workers utility is {tau_star:3.2f}')
        else:
            return tau_star
        
    def calc_optimal_government_consumption(self, extension=False, CES=False, do_print=False, set_2=False):
        """ calculate the government consumption corresponding to tau_star """
        par = self.par
        sol = self.sol

        # Solve the model for the given tau_star and different parameters:  
        if set_2:
            par.w = 1.0
            par.sigma = 1.5 
            par.rho = 1.5 
            par.epsilon = 1.0
            par.tau = self.optimal_tax_cd(extension=extension)
            self.solve(CES=CES)
        else:
            par.w = 1.0
            par.tau = self.optimal_tax_cd(extension=extension)
            self.solve(CES=CES)

        # Calculate the government consumption
        calc_G = par.tau * par.w * sol.L * ((1 - par.tau) * par.w)

        if do_print & set_2:
            print(f'Government consumption with the second set of parameters is {calc_G:3.2f}')
        elif do_print:   
            print(f'Government consumption with the first set of parameters is {calc_G:3.2f}')
        else:
            return calc_G
    

    def optimal_tax_ces(self, extension=False, CES=False, do_print=False, set_2=False): 
        """ calculate the tax rate keeping G"""
        par = self.par
        sol = self.sol

        if set_2: 
            def obj(tau):
                """Objective function to maximize worker utility"""
                G = self.calc_optimal_government_consumption(extension=extension, CES=CES, set_2=set_2)
                utility = self.calc_utility(sol.L, CES=CES)
                return -utility

            results = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')

            # Get the optimal tax rate
            tau_star_set_2 = results.x

            #print
            if do_print:
                print(f'The tax rate that maximizes workers utility keeping G with the second set of parameters is {tau_star_set_2:3.2f}')

        else: 
            def obj(tau):
                """Objective function to maximize worker utility"""
                G = self.calc_optimal_government_consumption(extension=extension, CES=CES)
                utility = self.calc_utility(sol.L, CES=CES)
                return -utility

            results = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')

            # Get the optimal tax rate
            tau_star_set_1 = results.x

            #print
            if do_print:
                print(f'The tax rate that maximizes workers utility keeping G with the first set of parameters is {tau_star_set_1:3.2f}')







