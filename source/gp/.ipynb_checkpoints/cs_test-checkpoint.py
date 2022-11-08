import numpy as np
from gp import KernelFunction, GPR
import casadi as cs

if __name__ == "__main__":
    kernel = KernelFunction(np.eye(1),1)

    x = cs.SX.sym('x',3,1) 
    y = cs.SX.sym('y',3,1) 
    
    output = kernel(x,y)
    print(output)

    model = GPR(x, y, KernelFunction, np.array([1,1,1]))

    
    #k = kernel(1,0)
    #print(k)