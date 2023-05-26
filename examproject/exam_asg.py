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
        return kappat * lt**(1-self.par.eta) - self.par.w * lt #calculating profits given the formula
    
    def find_optimal_lt(self): # defining the function that finds the optimal lt for each value of kappa
        for i,kappat in enumerate(self.par.kappas): # for loop to iterate over the different values of kappa 
            # Formulate objective function (negative of profit function)
            obj = lambda x: -self.profits(kappat, x)
            
            result = optimize.minimize_scalar(obj, bounds=(0, 2), method='bounded') #using the bounded method to find the optimal lt that maximizes profits
            
            self.sol.lt[i] = result.x #saving the optimal lt in the solution namespace
            
            print(f'For kappa = {kappat:.1f} the optimal labor supply is lt = {self.sol.lt[i]:.2f} which yields profits of {self.profits(kappat, self.sol.lt[i]):.2f} \n') #printing results
            
    ############## Question 2 and 3 ################                    
    def optimal_labor_rule(self,kappat):
        return ((1-self.par.eta) * kappat / self.par.w) ** (1/(self.par.eta))

    def profits_func(self,kappat,lt,lt_prev):
        return kappat * lt ** (1-self.par.eta) - self.par.w * lt - (lt != lt_prev) * self.par.iota


    def calc_H(self, Delta=0.0, K = 3500, do_print=False,extension=False):
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
            if extension:
                print(f'For K={K} the value of H is {H:.3f}\n')
            else:
                if Delta==0: #printing results for the case where Delta=0
                    print(f'For K={K} the value of H is {H:.3f}\n')
                else: #printing results in other cases
                    print(f'For K={K} and Delta={Delta} the value of H is {H:.3f}\n')

        return H
    

    def estimate_K(self,tol=1e-2,do_print=True,do_plot=True):

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
        Delta0 = 0.1 #initial guess on Delta
        
        # Formulate objective function as the negative of H as we maximize H using a minimizer
        obj = lambda x: - self.calc_H(Delta=x,K=K)

        result = optimize.minimize(obj, x0=Delta0, bounds=[(0.0, 1.0)], method='Nelder-mead')

        #save result
        self.sol.Delta_opt = result.x[0]

        # Print results
        print(f'H is maximized for Delta = {self.sol.Delta_opt:.3f}, implying H={-result.fun:.3f}')
        

    def plot_Delta(self,K=1000):

        H_Delta = np.empty(50)
        for i,Delta in enumerate(np.linspace(0,0.2,50)):
            self.par.Delta = Delta
            H_Delta[i] = self.calc_H(Delta=Delta,K=K)
        
        # Generate local variable of optimal H
        H_opt = self.calc_H(Delta=self.sol.Delta_opt, K=K)

        # Create figure
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.linspace(0,0.2,50),H_Delta, color='purple')
        ax.scatter(self.sol.Delta_opt, H_opt, color='red')
        #ax.annotate(f'(Delta,H)=({self.sol.Delta_opt:.3f},{H_opt:.3f})',xy=(self.sol.Delta_opt-0.01,H_opt), xytext=(self.sol.Delta_opt-0.01,H_opt), textcoords='offset points')
        ax.set_xlabel(r'$\Delta$')
        ax.set_ylabel("Ex ante expected value")
        ax.set_xlim([0,0.2])
        ax.set_ylim([27.5,29.0])
        plt.grid(True)
        plt.show();
   


            





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
    
    
    
         
    
        
        