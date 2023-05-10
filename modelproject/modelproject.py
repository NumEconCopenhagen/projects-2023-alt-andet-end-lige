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
        par.alpha = 3.0
        par.q = 0.5
        par.b_L = 2.5
        par.b_H = 1
        par.y_L = 100
        par.y_H = 200
        par.r_H = 70
        par.r_L = 30
        par.rho = 2.0
        par.delta = 0.0

        # baseline settings
        par.e_max = 30
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
        return 0.04*e**2.2
    
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



    ##################### Introducing risk ##########################
    
    # Workers are risk averse
    def u_L_alt(self,w, e):
        """"Utility function for low-productive worker"""
        return w**(1-self.par.rho)/(1-self.par.rho)-self.par.b_L*self.f(e)

    def u_H_alt(self,w, e):
        """Utility function for high-productive worker"""
        return w**(1-self.par.rho)/(1-self.par.rho)-self.par.b_H*self.f(e)



    def R_L_alt(self,e):
        """Revenue from low-productive workers"""
        return self.par.y_L+(self.par.alpha+self.par.delta)*e
    
    def R_H_alt(self,e):
        """Revenue from low-productive workers"""
        return self.par.y_H+(self.par.alpha+self.par.delta)*e  

    def profits_alt(self, w_L, e_L, w_H, e_H):
        """Firm's profit function"""
        return self.par.q*(self.R_H_alt(e_H) - w_H) + (1-self.par.q)*(self.R_L_alt(e_L)-w_L)
    
    def f(self, e):
        "Increasing marginal utility cost from education"
        return 0.04*e**2.2
    
    
    def solve_principal_one(self):

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











    





 





