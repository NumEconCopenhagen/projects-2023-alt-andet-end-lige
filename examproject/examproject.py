# importing the used packages 
from types import SimpleNamespace
from scipy import optimize
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

#####################################################################################################
############################################# Problem 1 #############################################
#####################################################################################################

## Indsæt Nikos kode her


#####################################################################################################
############################################# Problem 2 #############################################
#####################################################################################################
# Setting basic, visual settings
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"-"})
plt.rcParams.update({'font.size': 10})

class LaborAdjustmentCosts():
    """Class for solving the labor adjustment costs model"""
    def __init__(self):
        """Creates the model"""
        # namespaces
        par = self.par = SimpleNamespace() # namespace for parameters
        sol = self.sol = SimpleNamespace() # namespace for solutions

        # baseline parameters
        par.eta = 0.5 
        par.w = 1.0 
        par.kappas = [1.0, 2.0]
        
        # Simulation parameters
        par.T = 120 # Planning horizon

        # solutions
        sol.lt = np.zeros(len(par.kappas))
        sol.epsilon = np.zeros(par.T)
        sol.K = np.nan # Optimal number of shock series
        sol.Delta_opt = np.nan


        # extended model parameters
        par.rho = 0.90
        par.iota = 0.01
        par.sigma_epsilon = 0.10
        par.R = (1+0.01)**(1/12)
        


    ############## Question 1 ################
    
    def profits(self, kappat, lt): # defining the profits function
        """Calculates profits without adjustment costs"""
        return kappat * lt**(1-self.par.eta) - self.par.w * lt #calculating profits given the formula
    
    def find_optimal_lt(self): # defining the function that finds the optimal lt for each value of kappa
        """Function that solves for the optimal labor"""
        for i,kappat in enumerate(self.par.kappas): # for loop to iterate over the different values of kappa 
            # Formulate objective function (negative of profit function)
            obj = lambda x: -self.profits(kappat, x)
            
            result = optimize.minimize_scalar(obj, bounds=(0, 2), method='bounded') #using the bounded method to find the optimal lt that maximizes profits
            
            self.sol.lt[i] = result.x #saving the optimal lt in the solution namespace
            
            print(f'For kappa = {kappat:.1f} the optimal labor supply is lt = {self.sol.lt[i]:.2f} which yields profits of {self.profits(kappat, self.sol.lt[i]):.2f} \n') #printing results
            
    ############## Question 2 and 3 ################                    
    def optimal_labor_rule(self,kappat): # defining the policy labor demand function as given in the problem
        """Policy labor demand function"""
        return ((1-self.par.eta) * kappat / self.par.w) ** (1/(self.par.eta))

    def profits_func(self,kappat,lt,lt_prev):
        """Profits function with adjustment costs"""
        return kappat * lt ** (1-self.par.eta) - self.par.w * lt - (lt != lt_prev) * self.par.iota


    def calc_H(self, Delta=0.0, K = 3500, do_print=False,extension=False):
        """Calculates the ex ante expected value of the salon"""
        h_values = np.zeros(K) #initializing h-values

        np.random.seed(1999) # set seed

        for k in range(K): #for loop to iterate over the different shocks
            epsilon = np.random.normal(-0.5 * self.par.sigma_epsilon**2, self.par.sigma_epsilon, size=self.par.T) # Initialize shock series
            
            #creating variables for values in initial period
            lt_prev = 0.0
            kappat_prev = 1.0
            h_k = 0
                    
            for t in range(self.par.T): # loop over each of the T periods in the time horizon
                #calculating the demand shock in period t
                kappat = np.exp(self.par.rho * np.log(kappat_prev) + epsilon[t])
                
                if extension:
                    if self.profits_func(kappat,self.optimal_labor_rule(kappat),lt_prev)>self.profits_func(kappat,lt_prev,lt_prev):
                        lt = self.optimal_labor_rule(kappat)
                    else:
                        lt = lt_prev
                else:
                    #calculating the labor supply in period t given the optimal labor supply function
                    if Delta==0: # used for question problem 2 question 2
                        lt = self.optimal_labor_rule(kappat)
                    else: # used for question problem 2 question 3
                        if np.abs(lt_prev - self.optimal_labor_rule(kappat)) > Delta:
                            lt = self.optimal_labor_rule(kappat)
                        else:
                            lt = lt_prev
                
                # Compute profits
                profits = self.profits_func(kappat,lt,lt_prev)

                #Now calculating the discounted value of the salon in time t
                h_k += self.par.R ** (-t) * profits
                
                #updating lt_prev and kappat_prev for the next iteration
                lt_prev = lt
                kappat_prev = kappat
                
            #saving the discounted value of the salon in the h_values vector for the k'th shock series
            h_values[k] = h_k
        
        # now we calculate the average value function H by taking the mean of the h_values vector
        H = np.mean(h_values)
        
        if do_print==True: 
            if extension: # printing results for the extension where Delta is not included
                print(f'For K={K} the value of H is {H:.3f}\n')
            else:
                if Delta==0: #printing results for the case where Delta=0
                    print(f'For K={K} the value of H is {H:.3f}\n')
                else: #printing results in other cases
                    print(f'For K={K} and Delta={Delta} the value of H is {H:.3f}\n')

        return H
    

    def estimate_K(self,tol=1e-2,do_print=True,do_plot=True):
        """Estimates optimal K"""
        H_range = np.empty(len(range(500,6500,500)))
        H_K = 0.0
        for i,K in enumerate(range(500,6500,500)):
            H_range[i] = self.calc_H(K=K)
            if np.abs(H_range[i]-H_range[i-1])<tol and H_K==0.0:
                self.sol.K = K #set optimal K 
                H_K = H_range[i] # store associated H in local
            else:
                continue
        
        if do_print:
            print(f'By comparing H for different values of K in the range between 500 and 6000 with a step size of 500, we choose K to be {self.sol.K}')

        if do_plot:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(1,1,1)
            ax.plot(range(500,6500,500),H_range, color='green')
            ax.scatter(self.sol.K, H_K , color='orange')
            #ax.annotate(f'Choice of K=({self.sol.K}',xy=(self.sol.K*0.8,H_K*1.2), xytext=(self.sol.K*0.8,H_K*1.2), textcoords='offset points')
            ax.set_xlabel("K")
            ax.set_ylabel("Ex ante expected value")
            ax.set_xlim([500,6000])
            ax.set_ylim([27.3,28.0])
            plt.grid(True)
            plt.show();            
        
    ############## Question 4 ################        

    def max_H(self,K=1000):
        """Function that maximizes H with respect to Delta"""
        Delta0 = 0.1 #initial guess on Delta
        
        # Formulate objective function as the negative of H as we maximize H using a minimizer
        obj = lambda x: - self.calc_H(Delta=x,K=K)

        result = optimize.minimize(obj, x0=Delta0, bounds=[(0.0, 1.0)], method='Nelder-mead')

        #save result
        self.sol.Delta_opt = result.x[0]

        # Print results
        print(f'H is maximized for Delta = {self.sol.Delta_opt:.3f}, implying H={-result.fun:.3f}')
        

    def plot_Delta(self,K=1000):
        """Function that plots H as a function of Delta"""
        H_Delta = np.empty(30)
        for i,Delta in enumerate(np.linspace(0,0.2,30)):
            self.par.Delta = Delta
            H_Delta[i] = self.calc_H(Delta=Delta,K=K)
        
        # Generate local variable of optimal H
        H_opt = self.calc_H(Delta=self.sol.Delta_opt, K=K)

        # Create figure
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.linspace(0,0.2,30),H_Delta, color='purple')
        ax.scatter(self.sol.Delta_opt, H_opt, color='red')
        #ax.annotate(f'(Delta,H)=({self.sol.Delta_opt:.3f},{H_opt:.3f})',xy=(self.sol.Delta_opt-0.01,H_opt), xytext=(self.sol.Delta_opt-0.01,H_opt), textcoords='offset points')
        ax.set_xlabel(r'$\Delta$')
        ax.set_ylabel("Ex ante expected value")
        ax.set_title("$H$ for different values of $\Delta$")
        ax.set_xlim([0,0.2])
        ax.set_ylim([27.5,29.0])
        plt.grid(True)
        plt.show();
   

#####################################################################################################
############################################# Problem 3 #############################################
#####################################################################################################

class Problem3():
    def __init__(self):
        """Create model"""
        par = self.par = SimpleNamespace() # define simplenamespace for parameters
        sol = self.sol = SimpleNamespace() # define simplenamespace for solutions

        #Define bounds
        par.bound_low = -600
        par.bound_high = 600

        # Define tolerance
        par.tol = 10**(-8)

        # Set iterations
        par.K_warm = 10 # number of warm-up iterations
        par.K_max = 1000 # maximum number of iterations

        # solution parameters
        sol.f_opt = np.nan
        sol.x_opt = np.nan
        sol.iterations = np.nan

    
    def allocate(self):
        """Formulate elements depending on parameters"""
        # call on namespace
        par = self.par
        sol = self.sol

        # empty arrays for x0s (for step 3.D) and associated function value
        x0_k = np.zeros((par.K_max,2))
        x_kstar = np.empty((par.K_max,2))

        # set seed and draw K random values of x_k in a uniform distribution in bound [-600,600]
        np.random.seed(999)
        xs = par.bound_low + 2*par.bound_high*np.random.uniform(size=(par.K_max,2))

        return x0_k,x_kstar,xs

    def griewank_(self,x1,x2):
        A = x1**2/4000 + x2**2/4000
        B = np.cos(x1/np.sqrt(1))*np.cos(x2/np.sqrt(2))
        return A-B+1
    
    def griewank(self,x):
        return self.griewank_(x[0],x[1])
    

    
    def global_optimizer(self, do_print=True, do_plot=False):
        """Implement the refined global optimizer with multi-start"""
        # Call on namespace
        par = self.par
        sol = self.sol
        # Call on allocate function
        x0_k,x_kstar,xs = self.allocate()

        # Formulate chi-function
        def chi(k):
            """Formulate chi-function"""
            return 0.5*2/(1+np.exp((k-self.par.K_warm)/100))
        
        # Initial values of optimal x and function value
        f_opt = np.inf
        x_opt = np.nan

        for k,x_k in enumerate(xs):
            if k<par.K_warm: # step 3.B
                # Set effective initial guess equal to x_k
                x0_k[k] = x_k

            else:
                # Set effective initial guess as specified in 3.D
                x0_k[k] = chi(k)*x_k+(1-chi(k))*x_opt

            # opimize    
            result = optimize.minimize(self.griewank,x0_k[k],method='BFGS',tol=par.tol)
            x_kstar[k,:] = result.x # assign optimized value to x_kstar
            f = result.fun # assign function value to x_kstar

            # print first 10 iterations or if better than seen yet
            if k<10 or f < f_opt:
                if f < f_opt: # step 3.F
                    f_opt = f
                    x_opt = x_kstar[k,:]   

                # print
                if do_print:
                    print(f'{k:4d}: Effective initial guess = ({x0_k[k][0]:7.2f},{x0_k[k][1]:7.2f})',end='')
                    print(f' -> converged at ({x_kstar[k][0]:7.2f},{x_kstar[k][1]:7.2f}) with f = {f:12.8f}')

                # Break loop if function value is below tolerance
                if f_opt < par.tol: 
                    sol.x_opt = x_opt
                    sol.f_opt = f_opt
                    sol.iterations = k
                    sol.x0_k = x0_k

                    break
                
        # best solution
        if do_print:
            print(f'\n best solution: \n x=({sol.x_opt[0]:.3f},{sol.x_opt[1]:.3f}) with f = {sol.f_opt:12.8f} and a convergence speed of {sol.iterations} iterations')

        if do_plot:
            self.plot()
        

    def plot(self):
        """Illustrate how the effective initial guess vary with the iteration counter"""
        par = self.par
        sol = self.sol

        # create figure
            # we use that the the iteration counter is the same as the index of the numpy array x0_k
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(1,1,1)
        ax.plot(range(len(sol.x0_k)), sol.x0_k[:,0], color='blue', label = r'x_1')
        ax.plot(range(len(sol.x0_k)), sol.x0_k[:,1], color='red', label = r'x_2')
        ax.legend()
        ax.set_xlabel(r'Iteration, $k$')
        ax.set_ylabel(r'Effective initial guesses, $x^{k0}$')
        ax.set_title("Effective initial guess plotted against number of iterations")
        ax.set_xlim([0,sol.iterations+50])
        ax.set_ylim([-600,600])
        plt.grid(True)
        plt.show()

    def estimate(self):
        """Estimate number of warm-up iterations maximizing convergence speed
        by finding the minimum number of iterations by looping through different
        number of warm-up iterations"""
        par = self.par
        sol = self.sol

        iterations_opt = np.inf
        K_warm_opt = np.nan

        for K_warm in range(1,40):
            par.K_warm = K_warm
            
            # Find global optimizer
            self.global_optimizer(do_print=False)

            if sol.iterations < iterations_opt:
                iterations_opt = sol.iterations
                K_warm_opt = K_warm
            
        print(f'The number of warm-up iterations maximizing speed of convergence is {K_warm_opt}.')
        print(f'This implies {iterations_opt} iterations before convergence to global optimum is obtained.')

        return K_warm_opt










        





        
        