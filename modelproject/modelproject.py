from scipy import optimize

def Rl(yl, el, alpha=0.5):
    return yl + alpha*el
    
def Rh(yh, eh, alpha=0.5):
    return yh + alpha*eh
    
def ul(w, bl, el):
    return w-bl*el

def uh(w, bh, eh):
    return w-bh*eh

def revenue_max(q, yh, alpha, eh, wh, yl, el, wl):
    return q*(yh + alpha*eh - wh) + (1-q)*(yl+alpha*el-wl)


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



