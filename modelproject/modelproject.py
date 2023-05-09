from types import SimpleNamespace
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

class PrincipalAgent():
    def __init__(self):
        """Create the model"""
        # namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # baseline parameters
        par.alpha = 0.5
        par.q = 0.5
        par.b_L = 3
        par.b_H = 1
        par.y_L = 100
        par.y_H = 200
        par.r_H = 35
        par.r_L = 25

        # baseline settings
        par.e_max = 20
        par.N = 100

        # solutions
        sol.w_L = np.nan
        sol.w_H = np.nan
        sol.e_L = np.nan
        sol.e_H = np.nan

        

    def R_L(self,e):
        """Revenue from low-productive workers"""
        return self.par.y_L+self.par.alpha*e
    
    def R_H(self,e):
        """Revenue from low-productive workers"""
        return self.par.y_H+self.par.alpha*e  

    def profits(self, w_L, e_L, w_H, e_H):
        """Firm's profit function"""
        return self.par.q*(self.R_H(e_H) - w_H) + (1-self.par.q)*(self.R_L(e_L)-w_L)
    
    def f(self, e):
        "Increasing marginal utility cost from education"
        return 0.02*e**2.2
    
    def u_L(self,w, e):
        """"Utility function for low-productive worker"""
        return w-self.par.b_L*self.f(e)

    def u_H(self,w, e):
        """Utility function for high-productive worker"""
        return w-self.par.b_H*self.f(e)
    


    
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
    

    # Solve model

    def solve(self):

        par = self.par
        sol = self.sol

        #setup
        bounds = ((0.0,np.inf),(0.0,par.e_max),(0.0,np.inf),(0.0,par.e_max)) #you cannot take an infinitely long education
        IR_H ={'type': 'ineq', 'fun': self.ineq_IR_H} 
        IR_L ={'type': 'ineq', 'fun': self.ineq_IR_L}
        IC_H ={'type': 'ineq', 'fun': self.ineq_IC_H}  
        IC_L ={'type': 'ineq', 'fun': self.ineq_IC_L} 

        # call optimizer
        x0 = (par.y_L, 10.0, par.y_H, 15.0)
        results = optimize.minimize(self.objective, x0, 
                                   method='SLSQP', 
                                   bounds=bounds, 
                                   constraints = [IR_H, IR_L, IC_H, IC_L])
        
        # save
        sol.w_L = results.x[0]
        sol.e_L = results.x[1]
        sol.w_H = results.x[2]
        sol.e_H = results.x[3]


    ########## Plot solutions ############
    
    # find indifference curves through optimum
    def find_indifference_curves(self):
        
        #allocate memory
        self.e_vecs = []
        self.w_vecs = []
        self.us = []

        # Utility in optimum
        uH = self.u_H(self.sol.w_H, self.sol.e_H)
        uL = self.u_L(self.sol.w_H, self.sol.e_H)

        # allocate numpy arrays
        self.wH_vec = np.empty(self.par.N)
        self.wL_vec = np.empty(self.par.N)
        self.e_vec = np.linspace(1e-8,self.par.e_max,self.par.N)

        # Loop through e and find each w on same indifference curve as optimum
        for i,e in enumerate(self.e_vec):

            def obj_H(w):
                return self.u_H(w,e)-uH
            def obj_L(w):
                return self.u_L(w,e)-uL
            
            sol_H = optimize.root(obj_H,0)
            self.wH_vec[i] = sol_H.x[0]

            sol_L = optimize.root(obj_L,0)
            self.wL_vec[i] = sol_L.x[0]



    def plot_solutions(self,ax):

        ax.plot(self.sol.e_L,self.sol.w_L, 'ro') #low-produtvives
        ax.plot(self.sol.e_H,self.sol.w_H, 'ro') #high-productives

    def plot_indifference_curves(self,ax):
        
        # Plot for low-productives
        ax.plot(self.e_vec,self.wL_vec)

        # Label for low-productives
        x, y = self.e_vec[-1], self.wL_vec[-1]
        ax.annotate("L", xy=(x, y), xytext=(5, -5), textcoords='offset points')

        # Plot for high-productives
        ax.plot(self.e_vec,self.wH_vec)

        # Label for high-productives
        x, y = self.e_vec[-1], self.wH_vec[-1]
        ax.annotate("H", xy=(x, y), xytext=(5, 5), textcoords='offset points')

    def plot_details(self,ax):
        ax.set_xlabel('$e$')
        ax.set_ylabel('$w$')
                
        ax.set_xlim([0,self.par.e_max+2])
        ax.set_ylim([0,70])

        ax.grid(ls='--',lw=1)


    def plot_everything(self):
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(1,1,1)

        self.plot_indifference_curves(ax)
        #self.plot_budgetset(ax)
        self.plot_solutions(ax)
        self.plot_details(ax)











    





 









def solve_ss(alpha, c):
    """ Example function. Solve for steady state k. 

    Args:
        c (float): costs
        alpha (float): parameter

    Returns:
        result (RootResults): the solution represented as a RootResults object.

    """ 
    
    # a. Objective function, depends on k (endogenous) and c (exogenous).
    f = lambda k: k**alpha - c
    obj = lambda kss: kss - f(kss)

    #. b. call root finder to find kss.
    result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
    
    return result



