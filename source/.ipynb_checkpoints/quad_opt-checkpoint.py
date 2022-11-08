import numpy as np
import casadi as cs
import os
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

from quad import Quadrotor3D
from utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q, quaternion_inverse

import pdb

class quad_optimizer:
    def __init__(self, quad, t_horizon=1, n_nodes=100):
        
        self.n_nodes = n_nodes
        self.t_horizon = t_horizon
        #self.optimization_dt = optimization_dt

        self.optimization_dt = self.t_horizon/self.n_nodes
        #self.t_horizon = self.n_nodes*self.optimization_dt # look-ahead time
        
        self.quad = quad # quad is needed to create the casadi model using quad parameters
        self.dynamics = self.setup_casadi_model()
        
        self.x_dot = cs.MX.sym('x_dot', self.dynamics(x=self.x, u=self.u)['x_dot'].shape)  # x_dot has the same dimensions as the dynamics function output

        #print(dynamics(x=x, u=u)['x_dot'])

        #optimization_dt = 5e-2 # dt between predicted states
        #n_nodes = 100 # number of predicted states
        

        #self.discrete_dynamics = 

        self.terminal_cost = 1

        self.acados_model = AcadosModel()
        self.acados_model.name = 'quad_OCP'
        self.acados_model.x = self.x
        self.acados_model.u = self.u
        #acados_model.x_dot = x_dot
        
        #breakpoint()
        self.acados_model.f_expl_expr = self.dynamics(x=self.x, u=self.u)['x_dot'] # x_dot = f
        self.acados_model.f_impl_expr = self.x_dot - self.dynamics(x=self.x, u=self.u)['x_dot'] # 0 = f - x_dot

        #print(acados_model)
        

        self.acados_ocp = AcadosOcp()

        self.acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        self.acados_ocp.acados_include_path = self.acados_source_path + '/include'
        self.acados_ocp.acados_lib_path = self.acados_source_path + '/lib'

        self.acados_ocp.model = self.acados_model
        self.acados_ocp.dims.N = self.n_nodes # prediction horizon
        self.acados_ocp.solver_options.tf = self.t_horizon # look ahead time

        self.acados_ocp.cost.cost_type = 'LINEAR_LS' # weigths times states (as opposed to a nonlinear relationship)
        self.acados_ocp.cost.cost_type_e = 'LINEAR_LS' # end state cost


        self.nx = self.acados_model.x.size()[0]
        self.nu = self.acados_model.u.size()[0]
        self.ny = self.nx + self.nu # y is x and u concatenated for compactness of the loss function

        ## Optimization costs
        self.acados_ocp.cost.Vx = np.zeros((self.ny, self.nx)) # raise the dim of x to the dim of y
        self.acados_ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx) # weight only x
        self.acados_ocp.cost.Vx_e = np.eye(self.nx) # end x cost

        self.acados_ocp.cost.Vu = np.zeros((self.ny, self.nu)) # raise the dim of u to the dim of y
        self.acados_ocp.cost.Vu[-self.nu:, -self.nu:] = np.eye(self.nu) # weight only u

        # x cost (dim 12)
        #Original
        #q_cost = np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # euler angles
        #q_cost = np.array([0, 1000, 1, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # euler angles
        
        q_cost = np.array([10, 10, 10] +  [0.1, 0.1, 0.1] + [0.05, 0.05, 0.05] + [0.05, 0.05, 0.05])
        # add one more weigth to account for the 4 quaternions instead of 3 EA
        q_diagonal = np.concatenate((q_cost[:3], np.mean(q_cost[3:6])[np.newaxis], q_cost[3:]))
        r_cost = np.array([0.1, 0.1, 0.1, 0.1]) # u cost (dim 4)
        #r_cost = np.array([1000, 1000, 1000, 1000]) # u cost (dim 4)
        self.acados_ocp.cost.W = np.diag(np.concatenate((q_diagonal, r_cost))) # error costs
        self.acados_ocp.cost.W_e = np.diag(q_diagonal) * self.terminal_cost # end error cost


        # reference trajectory (will be overwritten later, this is just for dimensions)
        x_ref = np.zeros(self.nx)
        self.acados_ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
        #print(acados_ocp.cost.yref.shape)
        self.acados_ocp.cost.yref_e = x_ref # end node reference

        self.acados_ocp.constraints.x0 = x_ref

        # u constraints
        self.acados_ocp.constraints.lbu = np.array([0] * self.nu)
        self.acados_ocp.constraints.ubu = np.array([1] * self.nu)
        self.acados_ocp.constraints.idxbu = np.array([0,1,2,3]) # Indices of bounds on u 

        ## Solver options
        self.acados_ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        self.acados_ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.acados_ocp.solver_options.integrator_type = 'ERK'
        self.acados_ocp.solver_options.print_level = 0
        self.acados_ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # Compile acados OCP solver if necessary
        json_file = '_acados_ocp.json'
        
        self.acados_ocp_solver = AcadosOcpSolver(self.acados_ocp, json_file=json_file)
        
    def setup_casadi_model(self):
        
        self.p = cs.MX.sym('p',3) # Position
        self.q = cs.MX.sym('q',4) # Quaternion

        self.v = cs.MX.sym('v',3) # Velocity
        self.r = cs.MX.sym('w',3) # Angle rate

        self.x = cs.vertcat(self.p,self.q,self.v,self.r) # the complete state

        # Control variables
        u1 = cs.MX.sym('u1')
        u2 = cs.MX.sym('u2')
        u3 = cs.MX.sym('u3')
        u4 = cs.MX.sym('u4')

        self.u = cs.vertcat(u1,u2,u3,u4) # complete control

        # d position 
        f_p = self.v # position dynamicss

        # d quaternion
        f_q = 1.0 / 2.0 * cs.mtimes(skew_symmetric(self.r),self.q) # quaternion dynamics

        # d velocity
        f_thrust = self.u * self.quad.max_thrust
        a_thrust = cs.vertcat(0, 0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / self.quad.mass
        self.g = cs.vertcat(0,0,9.81)
        f_v = v_dot_q(a_thrust, self.q) - self.g # velocity dynamics

        # d rate
        y_f = cs.MX(self.quad.y_f)
        x_f = cs.MX(self.quad.x_f)
        c_f = cs.MX(self.quad.z_l_tau)
        # rate dynamics
        f_r =  cs.vertcat(
            (cs.mtimes(f_thrust.T, y_f) + (self.quad.J[1] - self.quad.J[2]) * self.r[1] * self.r[2]) / self.quad.J[0],
            (-cs.mtimes(f_thrust.T, x_f) + (self.quad.J[2] - self.quad.J[0]) * self.r[2] * self.r[0]) / self.quad.J[1],
            (cs.mtimes(f_thrust.T, c_f) + (self.quad.J[0] - self.quad.J[1]) * self.r[0] * self.r[1]) / self.quad.J[2])


        # concatenated dynamics
        x_dot = cs.vertcat(f_p, f_q, f_v, f_r)
        
        # Casadi function for dynamics 
        return cs.Function('x_dot', [self.x,self.u], [x_dot], ['x','u'], ['x_dot'])

    
    # this does not do anything, I want to use the acados ocp solver to make predictions, not the internal quad object, that exists only to get the model parameters
    def set_quad_state(self, x):
        self.quad.set_state(x)
        
    def set_reference_state(self, x_target=None, u_target=None):
        if u_target is None:
            u_target = np.ones((self.nu,))*0.16 # hover
        if x_target is None:
            x_target = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0])


        self.yref = np.empty((self.n_nodes, self.ny)) # prepare memory, N x ny 
        for j in range(self.n_nodes):
            self.yref[j,:] = np.concatenate((x_target, u_target)) # load desired trajectory into yref
            #print(self.yref[j,:])
            self.acados_ocp_solver.set(j, "yref", self.yref[j,:]) # supply desired trajectory to ocp

        # end point of the trajectory has no desired u 
        self.yref_N = x_target
        self.acados_ocp_solver.set(self.n_nodes, "yref", self.yref_N) # dimension nx
        #############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return self.yref, self.yref_N


    def set_reference_trajectory(self, x_trajectory=None, u_trajectory=None):
	    
	    self.yref = np.empty((self.n_nodes,self.ny)) # prepare memory, N x ny 
	    for j in range(self.n_nodes):
	        #print(j)
	        self.yref[j,:] = np.concatenate((x_trajectory[j,:], u_trajectory[j,:])) # load desired trajectory into yref
	        #print(yref[j,:])
	        self.acados_ocp_solver.set(j, "yref", self.yref[j,:]) # supply desired trajectory to ocp

	    # end point of the trajectory has no desired u 
	    self.yref_N = x_trajectory[-1,:]
	    self.acados_ocp_solver.set(self.n_nodes, "yref", self.yref_N) # dimension nx

	    return self.yref, self.yref_N
	    

    
    def run_optimization(self, x_init=None):

        if x_init is None:
            x_init = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0]) 

        # set initial conditions
        self.acados_ocp_solver.set(0, 'lbx', x_init) # not sure what these two lines do. Set the lower and upper bound for x to the same value?
        self.acados_ocp_solver.set(0, 'ubx', x_init)
        #self.acados_ocp_solver.set(0, 'x', x_init)

        # Solve OCP
        self.acados_ocp_solver.solve()
        self.acados_ocp_solver.solve()
        #self.acados_ocp_solver.solve()

        #print(x_init)

        # preallocate memory
        w_opt_acados = np.ndarray((self.n_nodes, self.nu)) 
        x_opt_acados = np.ndarray((self.n_nodes + 1, self.nx))
        x_opt_acados[0, :] = self.acados_ocp_solver.get(0, "x") # should be x_init?

        # write down the optimal solution
        for i in range(self.n_nodes):
            w_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u") # optimal control
            x_opt_acados[i + 1, :] = self.acados_ocp_solver.get(i + 1, "x") # state under optimal control

        #w_opt_acados = np.reshape(w_opt_acados, (-1))
        return x_opt_acados, w_opt_acados


    def discrete_dynamics(self, x, u, dt, body_frame=False):
        # Fixed step Runge-Kutta 4 integrator

    
        k1 = self.dynamics(x=x, u=u)['x_dot']
        k2 = self.dynamics(x=x + dt / 2 * k1, u=u)['x_dot']
        k3 = self.dynamics(x=x + dt / 2 * k2, u=u)['x_dot']
        k4 = self.dynamics(x=x + dt * k3, u=u)['x_dot']
        x_out = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        #print(x_out)
        if not body_frame:
            return x_out
        else:
            # velocity is transformed to bodyframe
            v_b = v_dot_q(x_out[7:10], quaternion_inverse(x_out[3:7]))
            #print(x_out)
            x_out = np.array([x_out[0], x_out[1], x_out[2], x_out[3], x_out[4], x_out[5], x_out[6],
                    v_b[0], v_b[1], v_b[2], x_out[10], x_out[11], x_out[12]])
            #print(x_out)
            return x_out
       




    @staticmethod
    def square_trajectory(n=10, dt=0.1):
    	# Calculate a square trajectory, static method
	    #dt = optimization_dt
	    v_max = 10
	    
	    nx = 13
	    t_section = np.arange(0,n*dt/4,dt)
	    p0 = np.array([0,0,0])
	    v = np.array([v_max,0,0])
	    p_target = p0[np.newaxis,:] + v*t_section[:,np.newaxis]
	    
	    p0 = p_target[-1,:]
	    v = np.array([0,v_max,0])
	    p_target = np.concatenate((p_target, p0[np.newaxis,:] + v*t_section[:,np.newaxis]))
	    
	    p0 = p_target[-1,:]
	    v = np.array([-v_max,0,0])
	    p_target = np.concatenate((p_target, p0[np.newaxis,:] + v*t_section[:,np.newaxis]))
	    
	    p0 = p_target[-1,:]
	    v = np.array([0,-v_max,0])
	    p_target = np.concatenate((p_target, p0[np.newaxis,:] + v*t_section[:,np.newaxis]))    
	    
	    x_target = np.zeros((p_target.shape[0], nx))
	    x_target[:,3] = 1
	    x_target[:,0:3] = p_target
	    return x_target


'''
def linearized_quad_dynamics():
    """
    Jacobian J matrix of the linearized dynamics specified in the function quad_dynamics. J[i, j] corresponds to
    the partial derivative of f_i(x) wrt x(j).

    :return: a CasADi symbolic function that calculates the 13 x 13 Jacobian matrix of the linearized simplified
    quadrotor dynamics
    """

    jac = cs.MX(state_dim, state_dim)

    # Position derivatives
    jac[0:3, 7:10] = cs.diag(cs.MX.ones(3))

    # Angle derivatives
    jac[3:7, 3:7] = skew_symmetric(r) / 2
    jac[3, 10:] = 1 / 2 * cs.horzcat(-q[1], -q[2], -q[3])
    jac[4, 10:] = 1 / 2 * cs.horzcat(q[0], -q[3], q[2])
    jac[5, 10:] = 1 / 2 * cs.horzcat(q[3], q[0], -q[1])
    jac[6, 10:] = 1 / 2 * cs.horzcat(-q[2], q[1], q[0])

    # Velocity derivatives
    a_u = (u[0] + u[1] + u[2] + u[3]) * quad.max_thrust / quad.mass
    jac[7, 3:7] = 2 * cs.horzcat(a_u * q[2], a_u * q[3], a_u * q[0], a_u * q[1])
    jac[8, 3:7] = 2 * cs.horzcat(-a_u * q[1], -a_u * q[0], a_u * q[3], a_u * q[2])
    jac[9, 3:7] = 2 * cs.horzcat(0, -2 * a_u * q[1], -2 * a_u * q[1], 0)

    # Rate derivatives
    jac[10, 10:] = (quad.J[1] - quad.J[2]) / quad.J[0] * cs.horzcat(0, r[2], r[1])
    jac[11, 10:] = (quad.J[2] - quad.J[0]) / quad.J[1] * cs.horzcat(r[2], 0, r[0])
    jac[12, 10:] = (quad.J[0] - quad.J[1]) / quad.J[2] * cs.horzcat(r[1], r[0], 0)

    return cs.Function('J', [x, u], [jac])
'''