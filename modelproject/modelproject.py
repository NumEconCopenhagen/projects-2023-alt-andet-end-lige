from types import SimpleNamespace
from scipy import optimize
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

class PrincipalAgent():
    def __init__(self):
        """Create the model"""
        # namespaces
        par = self.par = SimpleNamespace() # namespace for parameters
        sol = self.sol = SimpleNamespace() #namespace for solutions
        ext = self.ext = SimpleNamespace() # namespace for extension with many agents

        # baseline parameters
        par.alpha = 3.0
        par.q = 0.5
        par.b_L = 2.5
        par.b_H = 1
        par.y_L = 100
        par.y_H = 200
        par.r_H = 70
        par.r_L = 30
        

        # baseline settings
        par.e_max = 30 #maximum years of education
        par.N = 100

        # number of agents in extended model
        ext.n = 10

        # solutions
        sol.w = np.nan
        sol.w_L = np.nan
        sol.w_H = np.nan
        sol.e_L = np.nan
        sol.e_H = np.nan




    ################ Adverse selection ################

    # profit function
    def profits_one(self, w):
        if w < self.par.r_H and w<self.par.r_L:
            return 0
        elif w < self.par.r_H:
            return (1-self.par.q)*(self.par.y_L-w)
        else:
            return self.par.q*(self.par.y_H - w) + (1-self.par.q)*(self.par.y_L-w)

    # Define objective function and constraints
    def objective_one(self, x):
        """Objective function to minimize"""
        return -self.profits_one(x)
    
    def ineq_IR_H_one(self, x):
        """Individual rationality constraint for high-productives"""
        return x-self.par.r_H
    
    def ineq_IR_L_one(self, x):
        """Individual rationality constraint for low-productives"""
        return x-self.par.r_L

    # Solve model
    def solve_one(self):
        """Solve model when firms can only offer a single contract for both worker types"""

        # call namespaces
        par = self.par
        sol = self.sol

        #setup
        bounds = [(0.0,np.inf)]
        IR_H ={'type': 'ineq', 'fun': self.ineq_IR_H_one} 
        IR_L ={'type': 'ineq', 'fun': self.ineq_IR_L_one}

        # call optimizer
        x0 = par.y_L # initial guess
        results = optimize.minimize(self.objective_one, x0, 
                                   method='SLSQP', 
                                   bounds=bounds, 
                                   constraints = [IR_H, IR_L])
        
        
        # compare profits 
        if self.profits_one(results.x[0])>=self.profits_one(self.par.r_L):
            if self.profits_one(results.x[0])>=0:
                sol.w = results.x[0]
            else:
                sol.w = 0
        else:
            if self.profits_one(self.par.r_L)>=0:
                sol.w = self.par.r_L
            else:
                sol.w = 0

                
                            
   

        
      
    ############### Firms can now condition on education level and hence offer two types of contracts #############
    def R_L(self,e):
        """Revenue from low-productive workers"""
        return self.par.y_L+self.par.alpha*e
    
    def R_H(self,e):
        """Revenue from low-productive workers"""
        return self.par.y_H+self.par.alpha*e  
    
    def f(self, e):
        "Increasing marginal utility cost from education"
        return 0.04*e**2.2
    
    def u_L(self,w, e):
        """"Utility function for low-productive worker"""
        return w-self.par.b_L*self.f(e)

    def u_H(self,w, e):
        """Utility function for high-productive worker"""
        return w-self.par.b_H*self.f(e)
    
    def profits(self, w_L, e_L, w_H, e_H):
        """Firm's profit function"""
        if self.u_H(w_H, e_H) < self.par.r_H and self.u_L(w_L,e_L)<self.par.r_L:
            return 0
        elif self.u_H(w_H,e_H) < self.par.r_H and self.u_L(w_L,e_L)>=self.par.r_L:
            return (1-self.par.q)*(self.R_L(e_L)-w_L)
        elif self.u_H(w_H,e_H) >= self.par.r_H and self.u_L(w_L,e_L)<self.par.r_L:
            return self.par.q*(self.R_H(e_H) - w_H)
        else:
            return self.par.q*(self.R_H(e_H) - w_H) + (1-self.par.q)*(self.R_L(e_L)-w_L)


    
    # Define objective and constraints
    def objective(self, x):
        """Objective function to minimize"""
        return -self.profits(x[0],x[1],x[2],x[3])
    
    def ineq_IR_H(self, x):
        """Individual rationality constraint for high-productives"""
        return self.u_H(x[2],x[3])-self.par.r_H
    
    def ineq_IR_L(self, x):
        """Individual rationality constraint for low-productives"""
        return self.u_L(x[0],x[1])-self.par.r_L

    def ineq_IC_H(self, x):
        """Incentive compatibility constraint for high-productives"""
        return self.u_H(x[2],x[3])-self.u_H(x[0],x[1])
    
    def ineq_IC_L(self, x):
        """Incentive compatibility constraint for low-productives"""
        return self.u_L(x[0],x[1])-self.u_L(x[2],x[3])
    


    # Solve model when firm observes education level
    def solve(self):
        """Solve model when firm can now offer two types of contracts"""
        par = self.par
        sol = self.sol

        #setup
        bounds = ((0.0,np.inf),(0.0,par.e_max),(0.0,np.inf),(0.0,par.e_max)) #you cannot take an infinitely long education
        IR_H ={'type': 'ineq', 'fun': self.ineq_IR_H} 
        IR_L ={'type': 'ineq', 'fun': self.ineq_IR_L}
        IC_H ={'type': 'ineq', 'fun': self.ineq_IC_H}  
        IC_L ={'type': 'ineq', 'fun': self.ineq_IC_L} 

        # call optimizer
        x0 = (par.y_L, 10.0, par.y_H, 15.0) # initial guess

        # Optimal contract when we design one contracts for each of the worker types
        results = optimize.minimize(self.objective, x0, 
                                   method='SLSQP', 
                                   bounds=bounds, 
                                   constraints = [IR_H, IR_L, IC_H, IC_L])
        
        
        
        ####################################################################################################        
        ####### Find optimal contracts on the boundary where principal only offers a single contract #######
        ####################################################################################################

        # Optimal contract when principal offers a single contract that only low productivity workers accepts
        results_alt1 = optimize.minimize(self.objective, x0, 
                                   method='SLSQP', 
                                   bounds=bounds, 
                                   constraints = [IR_L, {'type': 'eq', 'fun': lambda x: x[2]}, {'type': 'eq', 'fun': lambda x: x[3]}])
        
        # Optimal contract when principal offers a single contract that both high and low productivity workers accepts 
            # this is the case when both outside options and productivity levels are very close to each other for the worker types
        results_alt2 = optimize.minimize(self.objective, x0, 
                                   method='SLSQP', 
                                   bounds=bounds, 
                                   constraints = [IR_H, IR_L, {'type': 'eq', 'fun': lambda x: x[2]-x[0]}, 
                                                  {'type': 'eq', 'fun': lambda x: x[3]-x[1]}])


        # Optimal contract when principal offers a single contract that only high productivity workers are willing to accept
        results_alt3 = optimize.minimize(self.objective, x0, 
                                   method='SLSQP', 
                                   bounds=bounds, 
                                   constraints = [IR_H, {'type': 'eq', 'fun': lambda x: x[0]}, 
                                                  {'type': 'eq', 'fun': lambda x: x[1]},
                                                  {'type': 'ineq', 'fun': lambda x: self.par.r_L-self.u_L(x[2],x[3])}])
        

        #########################################################################################################
        ###### Compare profits for different solutions and choose the contract prodiving the highest profit #####
        #########################################################################################################

        # Calculate profit for best alternative to the inner solution
        best_alt = max(self.profits(results_alt1.x[0], results_alt1.x[1],results_alt1.x[2],results_alt1.x[3]), 
                    self.profits(results_alt2.x[0], results_alt2.x[1],results_alt2.x[2],results_alt2.x[3]),
                    self.profits(results_alt3.x[0], results_alt3.x[1],results_alt3.x[2],results_alt3.x[3]))    

        # If inner solution implies positive and higher profits than best alternative -> set inner solution as solution
        if self.profits(results.x[0], results.x[1],results.x[2],results.x[3])>=best_alt:
            if self.profits(results.x[0], results.x[1],results.x[2],results.x[3])>0:
                sol.w_L = results.x[0]
                sol.e_L = results.x[1]
                sol.w_H = results.x[2]
                sol.e_H = results.x[3] 
            else:
                sol.w_L = 0.0
                sol.e_L = 0.0
                sol.w_H = 0.0
                sol.e_H = 0.0

        else: # if best alternative implies positive and higher profits than inner solution -> set best alternative as solution  
            if best_alt>0:
                if best_alt == self.profits(results_alt1.x[0], results_alt1.x[1],results_alt1.x[2],results_alt1.x[3]):
                    sol.w_L = results_alt1.x[0]
                    sol.e_L = results_alt1.x[1]
                    sol.w_H = results_alt1.x[2]
                    sol.e_H = results_alt1.x[3] 
                elif best_alt == self.profits(results_alt2.x[0], results_alt2.x[1],results_alt2.x[2],results_alt2.x[3]):
                    sol.w_L = results_alt2.x[0]
                    sol.e_L = results_alt2.x[1]
                    sol.w_H = results_alt2.x[2]
                    sol.e_H = results_alt2.x[3]  
                elif best_alt == self.profits(results_alt3.x[0], results_alt3.x[1],results_alt3.x[2],results_alt3.x[3]):
                    sol.w_L = results_alt3.x[0]
                    sol.e_L = results_alt3.x[1]
                    sol.w_H = results_alt3.x[2]
                    sol.e_H = results_alt3.x[3]

            else:
                sol.w_L = 0.0
                sol.e_L = 0.0
                sol.w_H = 0.0
                sol.e_H = 0.0




    ########## Plot solutions ############
    
    # find indifference curves through optimum
    def find_indifference_curves(self):

        # Utility in optimum
        uH = self.u_H(self.sol.w_H, self.sol.e_H)
        uL = self.u_L(self.sol.w_L, self.sol.e_L)

        # allocate numpy arrays
        self.wH_vec = np.empty(self.par.N)
        self.wL_vec = np.empty(self.par.N)
        self.e_vec = np.linspace(1e-8,self.par.e_max,self.par.N)

        # Loop through e_vec and find each w on same indifference curve as optimum for each type of worker
        for i,e in enumerate(self.e_vec):
            
            def obj_H(w):
                return self.u_H(w,e)-uH
            def obj_L(w):
                return self.u_L(w,e)-uL
            
            sol_H = optimize.root(obj_H,0)
            sol_L = optimize.root(obj_L,0)

            if self.par.q == 0: # if only low-productives do not create indifference curve for high-productives
                self.wH_vec[i] = np.nan
            else:
                self.wH_vec[i] = sol_H.x[0]
    
            if self.par.q == 1: # if only high-productive workers do not create indifference curve for low-productives
                self.wL_vec[i] = np.nan
            else:
                self.wL_vec[i] = sol_L.x[0]

    def find_isoprofit_curves(self):
        # Profit in optimum that each of the worker types provides for the firm
        pi_H = self.R_H(self.sol.e_H)-self.sol.w_H
        pi_L = self.R_L(self.sol.e_L)-self.sol.w_L

        # Allocate numpy arrays 
        self.wH_iso = np.empty(self.par.N)
        self.wL_iso = np.empty(self.par.N)

        # Loop through e_vec and find each w on same isoprofit curve as optimum for each type of worker
        for i,e in enumerate(self.e_vec):
            
            def obj_H(w):
                return self.R_H(e)-w-pi_H
            def obj_L(w):
                return self.R_L(e)-w-pi_L
            
            sol_H = optimize.root(obj_H,0)
            sol_L = optimize.root(obj_L,0)

            if self.par.q == 0: # if only low-productives do not create isoprofit line for high-productives
                self.wH_iso[i] = np.nan
            else:
                self.wH_iso[i] = sol_H.x[0]
            
            if self.par.q == 1: # if only high-productive workers do not create isoprofit line for low-productives
                self.wL_iso[i] = np.nan
            else:
                self.wL_iso[i] = sol_L.x[0]



    def plot_solutions(self,ax):

        if self.par.q == 1:
            pass
        else:
            ax.plot(self.sol.e_L,self.sol.w_L, 'ro') #low-produtvives
            ax.text(self.sol.e_L,self.sol.w_L/1.4,f'$L^*$')

        if self.par.q == 0:
            pass
        else:
            ax.plot(self.sol.e_H,self.sol.w_H, 'ro') #high-productives
            ax.text(self.sol.e_H,self.sol.w_H*1.08,f'$H^*$')

    def plot_indifference_curves(self,ax):
        
        # Plot for low-productives
        ax.plot(self.e_vec,self.wL_vec, color="black")

        # Label for low-productives
        x, y = self.e_vec[-1], self.wL_vec[-1]
        ax.annotate("L", xy=(x, y), xytext=(5, -5), textcoords='offset points')

        # Plot for high-productives
        ax.plot(self.e_vec,self.wH_vec, color="darkred")

        # Label for high-productives
        x, y = self.e_vec[-1], self.wH_vec[-1]
        ax.annotate("H", xy=(x, y), xytext=(5, -5), textcoords='offset points')

    def plot_isoprofit_curves(self,ax):
        # Plot for low-productives
        ax.plot(self.e_vec,self.wL_iso, color='grey', linestyle='dashed')

        # Label for low-productives
        x, y = self.e_vec[-1], self.wL_iso[-1]
        ax.annotate("Isoprofit_L", xy=(x, y), xytext=(5, -5), textcoords='offset points')

        # Plot for high-productives
        ax.plot(self.e_vec,self.wH_iso, color='grey', linestyle='dashed')

        # Label for high-productives
        x, y = self.e_vec[-1], self.wH_iso[-1]
        ax.annotate("Isoprofit_H", xy=(x, y), xytext=(5, -5), textcoords='offset points')   

    def plot_details(self,ax):
        ax.set_xlabel('$e$')
        ax.set_ylabel('$w$')
                
        ax.set_xlim([0,self.par.e_max+7])
        ax.set_ylim([0,250])

        ax.grid(ls='--',lw=1)


    def plot_everything(self):
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(1,1,1)

        self.plot_indifference_curves(ax)
        self.plot_isoprofit_curves(ax)
        self.plot_solutions(ax)
        self.plot_details(ax)




#################################################################
################ n different types of workers ###################
#################################################################

    def setup_many(self):
        """Setup for model with many agents"""

        # Call on namespaces
        par = self.par
        sol = self.sol
        ext = self.ext

        # Allocate arrays for solutions
        sol.w_vec = np.zeros(ext.n) # optimal wages in extended model
        sol.e_vec = np.zeros(ext.n) # optimal education levels in extended model

        # Draw linear spaced values of productivity levels in the interval from 10 to 350
        ext.y_vec = np.linspace(10.0,350.0,num=ext.n)

        # Draw linear spaced values of disutility from education between 3.5 and 1.0. 
            # We assume that disutility from education and worker's productivity level are negatively correlated,
            # thus we create array in descending order
        ext.b_vec = np.linspace(3.5,1.0,num=ext.n)


        # For simplicity we assume that the outside option is smaller than the productivity level and that it increases in the productivity level
        ext.r_vec = ext.y_vec*0.4

        # Find shares of each worker type
        F = norm(loc=np.mean(ext.y_vec),scale=100) # distribution of workers in the population is normally distributed around y=180
        ext.q_vec = F.pdf(ext.y_vec) # extract probabilities for each of the productivity levels in y_vec
        ext.q_vec /= np.sum(ext.q_vec) # rescale probabilities assuming that values in y_vec are the only possible worker types

    def _plot_distribution(self,n):
        """Plot distribution of workers in population"""
        ext = self.ext

        self.ext.n = n # types of workers
        self.setup_many() # update arrays in setup

        # Get draws from distribution for a population of 10,000 workers
        K = np.random.choice(ext.y_vec,size=10000,p=ext.q_vec)

        # Plot distribution
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1,1,1)
        ax.hist(K,bins=100,density=True,alpha=0.5,color='purple')

        # Plot details
        ax.set_xlim([-30.0,380.0])
        ax.set_xlabel('$y$')
        ax.set_ylabel('Density');
    
    def plot_distribution(self):
        "Interactive plot of distribution of workers"
        ext = self.ext
        widgets.interact(self._plot_distribution,
        n=widgets.IntSlider(description="Types of workers", min=1, max=100, step=1, value=10)
        );


    def u(self,b,w,e):
        """Utility function"""
        return w-b*self.f(e)
    
    def profits_many(self, *args):
        """Firm's profit function"""
        profits = 0.0
        
        for i in range(self.ext.n):
            if self.u(self.ext.b_vec[i],args[i],args[i+self.ext.n]) >= self.ext.r_vec[i]:
                pi_i = self.ext.q_vec[i]*(self.ext.y_vec[i]+self.par.alpha*args[i+self.ext.n]-args[i])
                profits += pi_i
            else:
                continue

        return profits
    

    # Define objective function
    def objective_many(self, x):
        """Objective function to minimize"""
        return -self.profits_many(*x)



    def constraints_many(self,x):
        """ Define IR- and IC constraints"""
        ext = self.ext

      # define constraints
        constraints = [] # construct empty list of constraints
        # For each type of worker, we now append the IR constraint of the worker 
                # and all the n-1 IC constraints of this worker to the list of constraints
        for i in range(ext.n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: self.u(ext.b_vec[i], x[i], x[i+ext.n]) - ext.r_vec[i], 'jac': None}) # IR_i constraint
                                                            # note that x[0][i] is worker i's wage and x[1][i] is his education level

            for j in range(ext.n):
                if j == i:
                    continue
                else:
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: self.u(ext.b_vec[i], x[i], x[i+ext.n]) - self.u(ext.b_vec[i], x[j], x[j+ext.n]), 'jac': None}) # IC_i constraints
        
        return constraints
    

    def solve_many(self):
        """Solve model when firm offers a contract for each of the n types of workers """

        par = self.par
        sol = self.sol
        ext = self.ext

        # Call on setup_many function such that arrays are updated when we change number of agents in the model.
        self.setup_many()

        # bounds for solution variables
        bounds_w = [(0.0, np.inf) for _ in range(ext.n)] # wage must be non-negative
        bounds_e = [(0.0, par.e_max) for _ in range(ext.n)] # education is non-negative and there is a limit on how high education you can take
        bounds = bounds_w + bounds_e 

        
        # The initial guess must now be a list of 2*n elements, 
                # where the first n elements are guesses on w_vec and the last n elements are guesses on e_vec
        w0_vec = ext.y_vec[:] # copy of y_vec
        e0_vec = np.linspace(0.0,25.0,num=ext.n)
        x0 = list(np.concatenate((w0_vec,e0_vec)))

        # define constraints
        constraints = self.constraints_many(x0)

        # Optimal contract when we design contracts accepted by both workers
        results = optimize.minimize(self.objective_many, x0, 
                                   method='SLSQP', 
                                   bounds = bounds,
                                   constraints = constraints)
                
        # save results
        for i in range(ext.n):
            sol.w_vec[i] = results.x[i]
            sol.e_vec[i] = results.x[i+ext.n]
        









 





