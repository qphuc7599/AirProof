"""Bound-preserving numerical refinement; objective and KKT tolerance unchanged."""
import numpy as np
from scipy.optimize import minimize

def refine_to_kkt(objective,x,lower,upper,*,tolerance=1e-7,max_iterations=1000,restarts=2):
    x=np.asarray(x,float).copy();lower=np.broadcast_to(lower,x.shape);upper=np.broadcast_to(upper,x.shape)
    history=[]
    for attempt in range(restarts+1):
        value,gradient=objective(x)
        residual=float(np.max(np.abs(x-np.clip(x-gradient,lower,upper))))
        history.append({'attempt':attempt,'objective':float(value),'kkt':residual})
        if residual<=10*tolerance or attempt==restarts:break
        result=minimize(objective,x,jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),
                        options={'gtol':tolerance,'ftol':0.,'maxiter':max_iterations,'maxls':80,'maxcor':20})
        if objective(result.x)[0]>value+1e-12*max(1,abs(value)):break
        x=result.x
    return x,history
