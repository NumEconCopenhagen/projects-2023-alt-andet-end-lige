from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

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










        




