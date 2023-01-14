 # 
 # This file is part of the mpc_quad_ros distribution (https://github.com/smidmatej/mpc_quad_ros).
 # Copyright (c) 2023 Smid Matej.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #


import numpy as np
import casadi as cs
import os
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

from quad import Quadrotor3D
from utils.utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q, quaternion_inverse

import pdb

class quad_optimizer:
    def __init__(self, quad, t_horizon=1, n_nodes=100, gpe=None):
        
        self.n_nodes = n_nodes
        self.t_horizon = t_horizon
        self.gpe = gpe

        self.optimization_dt = self.t_horizon/self.n_nodes
        self.terminal_cost = 1

        self.quad = quad # quad is needed to create the casadi model using quad parameters
        self.dynamics = self.setup_casadi_model()
       
        self.x_dot = cs.MX.sym('x_dot', self.dynamics(x=self.x, u=self.u)['f'].shape)  # x_dot has the same dimensions as the dynamics function output


        self.acados_model = AcadosModel()
        self.acados_model.name = 'quad_OCP'
        self.acados_model.x = self.x
        self.acados_model.u = self.u
        self.acados_model.p = self.params
        
        if self.gpe.type == 'RGP':
            self.acados_model.f_expl_expr = self.dynamics(x=self.x, u=self.u, p=self.params)['f'] # x_dot = f
            self.acados_model.f_impl_expr = self.x_dot - self.dynamics(x=self.x, u=self.u, p=self.params)['f'] # 0 = f - x_dot
        else:
            self.acados_model.f_expl_expr = self.dynamics(x=self.x, u=self.u)['f']
            self.acados_model.f_impl_expr = self.x_dot - self.dynamics(x=self.x, u=self.u)['f']

        '''
        if self.gpe is not None:
            self.acados_model.f_expl_expr = self.dynamics(x=self.x, u=self.u, p=self.params)['f'] # x_dot = f
            self.acados_model.f_impl_expr = self.x_dot - self.dynamics(x=self.x, u=self.u, p=self.params)['f'] # 0 = f - x_dot
        else:
            self.acados_model.f_expl_expr = self.dynamics(x=self.x, u=self.u)['f']
            self.acados_model.f_impl_expr = self.x_dot - self.dynamics(x=self.x, u=self.u)['f']
        '''

        

        self.nx = self.acados_model.x.size()[0]
        self.nu = self.acados_model.u.size()[0]
        if self.params is None:
            self.np = 0
        else:
            self.np = self.acados_model.p.size()[0]
        self.ny = self.nx + self.nu # y is x and u concatenated for compactness of the loss function

        print('Before change')
        print(self.acados_model.f_impl_expr)

        '''
        for i in range(3):
            self.gpe.gp[i].z_train = np.zeros((self.gpe.gp[i].n_train,))
            self.gpe.gp[i].y_train = np.zeros((self.gpe.gp[i].n_train,))
        
        #self.gpe.gp[0].z_train = np.zeros((self.gpe.gp[0].n_train,))
        print('After change')
        print(self.acados_model.f_impl_expr)
        '''

        #self.acados_model.set_param('gain', 1)
        print('After change')
        print(self.acados_model.f_impl_expr)
        self.acados_ocp = AcadosOcp()

        self.acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        self.acados_ocp.acados_include_path = self.acados_source_path + '/include'
        self.acados_ocp.acados_lib_path = self.acados_source_path + '/lib'

        self.acados_ocp.model = self.acados_model
        self.acados_ocp.dims.N = self.n_nodes # prediction horizon
        self.acados_ocp.solver_options.tf = self.t_horizon # look ahead time
        
        if self.gpe is not None:
            self.acados_ocp.parameter_values = np.array([1.0, 1.0]) # Default parameter values

        self.acados_ocp.cost.cost_type = 'LINEAR_LS' # weigths times states (as opposed to a nonlinear relationship)
        self.acados_ocp.cost.cost_type_e = 'LINEAR_LS' # end state cost



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
        
        q_cost = np.array([10, 10, 10] +  [0.1, 0.1, 0.1] + [0.05, 0.05, 0.05] + [0.05, 0.05, 0.05]) # ORIGINAL

        #q_cost = np.array([100, 100, 100] + [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0] + [0.05, 0.05, 0.05]) # MPC doesnt care about linear or angular velocity
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
        
        #self.acados_ocp.model.set('', self.t_step)

        print(f'Optimizer MPC lookahead time={self.t_horizon}s, Number of timesteps in prediction horizon={self.n_nodes}')


        
    def setup_casadi_model(self):
        """
        Creates the appropriate symbolic variables as class fields and returns the dynamic equation of the quad model 
        """
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

        self.params = None

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
        f_dyn = cs.vertcat(f_p, f_q, f_v, f_r)

        if self.gpe is not None:

            self.params = cs.vertcat(cs.MX.sym('gain1'), cs.MX.sym('gain2'))
            self.params = cs.MX.sym('p_y', len(self.gpe.gp), self.gpe.gp[0].X.shape[0])

            p_y = [None]*len(self.gpe.gp)
            for d in range(len(self.gpe.gp)):
                p_y[d] = cs.MX.sym('p_x', 1, self.gpe.gp[d].X.shape[0])
            
            # Transform to body frame because thats what the gpe were trained on
            v_body = v_dot_q(self.v, quaternion_inverse(self.q))
            breakpoint()
            v_body = v_body.tolist()

            # Symbolic prediction
            if self.gpe.type == "RGP":
                gp_means = self.gpe.predict_at_y(v_body, p_).T
            
            gp_means = self.gpe.predict(v_body).T

            # Transform prediction back to world frame because thats what the simulator uses
            gp_means = v_dot_q(gp_means, self.q)
            
            # 13 x 3 matrix that maps the 3 gpe means to dvx, dvy, dvz 
            B_x = np.concatenate([np.zeros((3,3)), np.zeros((4,3)), np.diag([1,1,1]), np.zeros((3,3))], axis=0)
            f_augment = cs.mtimes(B_x, gp_means)

            
            # Dynamics correction using learned GP
            f_corrected = self.params[0] * f_dyn + self.params[1] * f_augment


            # Dynamics corrected using the gpe 
            return cs.Function('x_dot', [self.x, self.u, self.params], [f_corrected], ['x', 'u', 'p'], ['f'])

        # Dynamics w/o the gpe augmentation
        return cs.Function('x_dot', [self.x,self.u], [f_dyn], ['x','u'], ['f'])

    
    def set_quad_state(self, x):
        # this does not do anything, 
        # I want to use the acados ocp solver to make predictions, not the internal quad object, that exists only to get the model parameters
   
        self.quad.set_state(x)
        
    def set_reference_state(self, x_target=None, u_target=None):
        """
        Sets the state x_target as the reference state over the whole optimization horizon (quad_opt.n_nodes)
        :param: x_target: np.array of shape 1 x nx
        """
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


    def set_reference_trajectory(self, x_trajectory, u_trajectory=None):
        """
        Passes x_trajectory to acados_ocp_solver. Acados_ocp_solver then tries to find the optimal control
        :param: x_trajectory: x_trajectory has to have the same length as quad_opt.n_node
        :param: u_trajectory: u_trajectory has to have the same length as quad_opt.n_node
        :param: oversampling: if x_trajectory has smaller sampling time then quad_opt.optimization_time, then undersample the signal to match the sampling time
        """
        #breakpoint()
        if u_trajectory is None:
            u_trajectory = np.ones((self.n_nodes, 4))*0.16 # hover

        self.yref = np.empty((self.n_nodes, self.ny)) # prepare memory, N x ny 
        for j in range(self.n_nodes):
            #print(j)
            self.yref[j,:] = np.concatenate((x_trajectory[j,:], u_trajectory[j,:])) # load desired trajectory into yref
            #print(yref[j,:])
            self.acados_ocp_solver.set(j, "yref", self.yref[j,:]) # supply desired trajectory to ocp

        # end point of the trajectory has no desired u 
        self.yref_N = x_trajectory[-1,:]
        self.acados_ocp_solver.set(self.n_nodes, "yref", self.yref_N) # dimension nx

        return self.yref, self.yref_N


    
    def run_optimization(self, x_init):
        """
        Performs the acados_ocp_solver.solve() method with the current parameters and returns the solution
        """
        if x_init is None:
            raise ValueError("x_init has to be set before running the optimization")

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
        return x_opt_acados, w_opt_acados, self.acados_ocp_solver.get_stats('time_tot'), self.acados_ocp_solver.get_cost()


    def discrete_dynamics(self, x, u, dt, body_frame=False):
        # 
        """
        Fixed step Runge-Kutta 4 of the dynamic equation in self.dynamics
        :param: x: State 1 x nx to discretize around
        :param: u: Conrol 1 x nu to discretize around
        :param: dt: Time step for dicretization
        """
    
        k1 = self.dynamics(x=x, u=u)['f']
        k2 = self.dynamics(x=x + dt / 2 * k1, u=u)['f']
        k3 = self.dynamics(x=x + dt / 2 * k2, u=u)['f']
        k4 = self.dynamics(x=x + dt * k3, u=u)['f']
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