from types import SimpleNamespace
from scipy import optimize
import numpy as np

class PrincipalAgent():
    def _init_(self):
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
        
    def u_L(self,w, e):
        """"Utility function for low-productive worker"""
        return w-self.par.b_L*e

    def u_H(self,w, e):
        """Utility function for high-productive worker"""
        return w-self.par.b_H*e
    
    def R_L(self,e):
        """Revenue from low-productive workers"""
        return self.par.y_L+self.par.alpha*e
    
    def R_H(self,e):
        """Revenue from low-productive workers"""
        return self.par.y_H+self.par.alpha*e  

    def profits(self, w_L, e_L, w_H, e_H):
        """Firm's profit function"""
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
        return self.u_H(x[0],x[1])-self.par.r_L

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
        bounds = ((0.0,np.inf),(0.0,25.0)(0.0,np.inf),(0.0,25.0)) #you cannot take an infinitely long education
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



