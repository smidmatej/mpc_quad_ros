import numpy as np
from six.moves import cPickle as pickle #for performance



def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)
        



def save_trajectories_as_dict(state, u, predicted_state, aero_drag, t, dt, filename='data/trajectory.pkl'):
	"""
	Saves 2 states into a dict to calculate the acceleration error
	a_error_k = (v_k+1 - v^star_k+1) / dt
	"""
	data = dict()

	# measured state
	data['p'] = state[:,0:3]
	data['q'] = state[:,3:7]
	data['v'] = state[:,7:10]
	data['w'] = state[:,10:13]
	data['u'] = u
	data['aero_drag'] = aero_drag

	# predicted state
	data['p_pred'] = predicted_state[:,0:3]
	data['q_pred'] = predicted_state[:,3:7]
	data['v_pred'] = predicted_state[:,7:10]
	data['w_pred'] = predicted_state[:,10:13]

	# need the dt to calculate a_error
	data['dt'] = dt
	data['t'] = t

	save_dict(data, filename) 