import numpy as np
from six.moves import cPickle as pickle #for performance




data = dict()
outputs = np.zeros([1, 2],dtype=np.float32)
data['x'] = np.zeros([1, 2],dtype=np.float32)
data['dx'] = np.zeros([1, 2],dtype=np.float32)
rng = np.random.default_rng(0) # seed

# initial condition
x0 = np.array([np.pi/2, 0])
# time points

# solve ODE
y = solve_ivp(derivs, t_span, x0, t_eval=t_eval)
y_sol = y.y.T

# derivatives
dy = np.asarray([derivs(0, y) for y in y_sol])

data['x'] = y_sol
data['dx'] = dy

save_dict(data, 'data/pendulumTrajectory.pkl') 