from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        opt = self.opt = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0
        par.kappa = 0.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        sol.alpha_hat = np.nan
        sol.sigma_hat = np.nan
        sol.kappa_hat = np.nan

    def calc_utility(self,LM,HM,LF,HF, extension=False):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. consumption of home production
        def H(HM,HF):
            if par.sigma == 0: #minimum
                return np.min(HM,HF)
            elif par.sigma == 1: # Cobb-douglas
                return HM**(1-par.alpha)*HF**par.alpha
            else: #CES
                return ((1-par.alpha)*HM**((par.sigma-1)/(par.sigma)) + par.alpha*HF**((par.sigma-1)/(par.sigma)))**(par.sigma/(par.sigma-1.0))
            
        # c. total consumption
        Q = C**par.omega*H(HM,HF)**(1-par.omega)

        # d. utility gain from total consumption
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutility of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        if extension:
            disutility+=par.kappa*(HM)

        return utility - disutility

    def solve_discrete(self,do_print=False, extension=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = self.opt
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF,extension=extension)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt




    def solve_continous(self,do_print=False, extension=False): 
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = self.opt

        # Objective function
        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3],extension=extension)

        # intitial guess
        intial_guess = [12, 12, 12, 12] # Initial guess: both male and female member use equal amount of time on labor and home production

        #bounds and constraints
        bounds = ((0,24),(0,24),(0,24),(0,24))
        constraints = [{'type':'ineq','fun': lambda x: 24-x[0]-x[1]},{'type':'ineq','fun': lambda x: 24-x[2]-x[3]}]

        result = optimize.minimize(obj, intial_guess, bounds=bounds, constraints=constraints,  method='SLSQP', tol=1e-08)
        
        # Setting the solution equal to the solution namespace:
        opt.LM = sol.LM = result.x[0] 
        opt.HM = sol.HM = result.x[1]
        opt.LF = sol.LF = result.x[2]
        opt.HF = sol.HF = result.x[3]
        
    # Printing result
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')  
        return opt
    

    def solve_wF_vec(self,discrete=False, extension=False): 
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol

        for i,wF in enumerate(par.wF_vec):

            self.par.wF = wF

            #Solve model
            if discrete:
                results = self.solve_discrete(extension=extension)
            else:
                results = self.solve_continous(extension=extension)

            #Store results
            sol.LF_vec[i] = results.LF
            sol.HF_vec[i] = results.HF
            sol.LM_vec[i] = results.LM
            sol.HM_vec[i] = results.HM


    def plot(self,discrete=True, extension=False):

        #Create empty lists of log-ratios
        logHF_HM = np.zeros(5)
        logwF_wM = np.zeros(5)

        #solve model discretely
        if discrete:
            self.solve_wF_vec(discrete=True)
        else:
            self.solve_wF_vec(discrete=False)

        # create relevant variables
        for i in range(5):
            logHF_HM[i] = np.log(self.sol.HF_vec[i]/ self.sol.HM_vec[i])
            logwF_wM[i] = np.log(self.par.wF_vec[i])

        # create the figure
        fig = plt.figure()

        # plot
        ax = fig.add_subplot(1,1,1)

        ax.plot(logwF_wM,logHF_HM) 

        ax.set_title("Plot of " + r'$\log(\frac{H_F}{H_M})$' + " against " r'$\log(\frac{w_F}{w_M})$') # add title
        ax.set_xlabel(r'$\log(\frac{w_F}{w_M})$') # add x-label
        ax.set_ylabel(r'$\log(\frac{H_F}{H_M})$'); # add y-label

        plt.show()



    def run_regression(self):
        """ run regression """
        par = self.par
        sol = self.sol
        
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        
    def squared_dev(self, pars, extension=False): 
        """Specify squared deviation from data moments at given parameter
        values for alpha and sigma"""
        # Set parameters
    
        if extension:
            kappa, sigma = pars
            self.par.kappa = kappa
            self.par.sigma = sigma
            self.par.alpha = 0.5
            
        else:
            alpha, sigma = pars
            self.par.alpha = alpha
            self.par.sigma = sigma
        
        # Solve model for different values of w_F
        self.solve_wF_vec(discrete=False, extension=extension)


        self.run_regression()
        beta0 = self.sol.beta0
        beta1 = self.sol.beta1

        return ((self.par.beta0_target - beta0)**2 + (self.par.beta1_target - beta1)**2)
    
    def estimate(self, extension=False):
        """Estimate the values of alpha and sigma that minimizes
        the squared deviation function from target coefficients
        Note: In the extension of the model we estimate kappa instead of alpha"""

        # Initial guess and bounds
        if extension:
            bounds = ((-3.00,3.00),(0.0000001,4.0)) # bounds = (kappa_bounds, sigma_bounds)
            par_guess = [0.0,1.0] # [kappa_guess,sigma_guess]
        else:
            bounds = ((0.000001,1.0),(0.0000001,4.0)) # bounds = (alpha_bounds,sigma_bounds)
            par_guess = [0.5,1.0] # [alpha_guess,sigma_guess]
  

        # Optimize
        result = optimize.minimize(self.squared_dev, par_guess,bounds=bounds, method="nelder-mead")
        
        # Save results
        if extension:
            self.sol.kappa_hat = result.x[0]
            self.sol.sigma_hat = result.x[1]
        else:
            self.sol.alpha_hat = result.x[0]
            self.sol.sigma_hat = result.x[1]



    def plot_modelfit(self, N=20):
        alphas = np.linspace(0.90,0.99,N)
        sigmas = np.linspace(0.05,0.1,N)

        # Create empty space for function values
        func_vec = np.empty((N,N))

        # Compute function values at different values for alpha and sigma
        for i, alpha in enumerate(alphas):
            for j, sigma in enumerate(sigmas):
                func_vec[i,j] = self.squared_dev([alpha,sigma])

        # Create grid of alpha and sigma values 
        alpha_grid, sigma_grid = np.meshgrid(alphas, sigmas)

        # Create 3D plot
        fig = plt.figure(figsize=(10,8)) #Initiating figure
        gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1])  # Create a 1x2 grid with width ratios for the subplots
        ax = fig.add_subplot(gs[0], projection='3d') #Making the plot 3d
        surf = ax.plot_surface(alpha_grid, sigma_grid, func_vec, cmap='jet', alpha = 0.50) #Creating figure
        ax.scatter([self.sol.alpha_hat], [self.sol.sigma_hat], [(self.par.beta0_target-self.sol.beta0)**2 + (self.par.beta1_target-self.sol.beta1)**2]) #Plotting optimal solution
        ax.set_xlabel(r'$\alpha$') # X-label
        ax.set_ylabel(r'$\sigma$') # Y-label
        ax.set_zlabel('Function Value') #Z-label
        ax.set_title('Function Values for Different Alpha and Sigma Values') #Title
        
        # Add color bar
        cax = fig.add_subplot(gs[1])  # Add subplot for the color bar
        cbar = fig.colorbar(surf, cax=cax) 
        cbar.set_label('Function Value')  # Set label for the color bar
        plt.show() 




        