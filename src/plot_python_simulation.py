from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from gp.data_loader import data_loader
from utils.utils import v_dot_q
import matplotlib as style
from matplotlib import gridspec
import matplotlib.colors as colors
from tqdm import tqdm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import os
from utils.save_dataset import load_dict


def main():

    plt.style.use('fast')
    sns.set_style("whitegrid")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    trajectory_filename = os.path.join(dir_path, '..', 'outputs', 'python_simulation', 'data', 'executed_trajectory.pkl')
    result_plot_filename = os.path.join(dir_path, '..', 'outputs', 'python_simulation', 'img', 'executed_trajectory.pdf')



    data_dict = load_dict(trajectory_filename)


    # Color scheme convert from [0,255] to [0,1]
    cs = [[x/256 for x in (8, 65, 92)], \
            [x/256 for x in (204, 41, 54)], \
            [x/256 for x in (118, 148, 159)], \
            [x/256 for x in (232, 197, 71)]] 



    
    fig = plt.figure(figsize=(10,6), dpi=100)
    plt.subplot(241)
    plt.plot(t, x_sim[:,0], 'r', linewidth=0.8)
    plt.plot(t, x_sim[:,1], 'g', linewidth=0.8)
    plt.plot(t, x_sim[:,2], 'b', linewidth=0.8)
    plt.plot(t, yref_sim[:,0], 'r--', linewidth=0.8)
    plt.plot(t, yref_sim[:,1], 'g--', linewidth=0.8)
    plt.plot(t, yref_sim[:,2], 'b--', linewidth=0.8)
    plt.title('Position xyz')


    plt.subplot(242)
    plt.plot(t, x_sim[:,3], 'r', linewidth=0.8)
    plt.plot(t, x_sim[:,4], 'g', linewidth=0.8)
    plt.plot(t, x_sim[:,5], 'b', linewidth=0.8)
    plt.plot(t, x_sim[:,6], 'c', linewidth=0.8)
    plt.plot(t, yref_sim[:,3], 'r--', linewidth=1)
    plt.plot(t, yref_sim[:,4], 'g--', linewidth=1)
    plt.plot(t, yref_sim[:,5], 'b--', linewidth=1)
    plt.plot(t, yref_sim[:,6], 'c--', linewidth=1)
    plt.title('Quaternion q')

    plt.subplot(243)
    plt.plot(t, x_sim[:,7], 'r', linewidth=0.8)
    plt.plot(t, x_sim[:,8], 'g', linewidth=0.8)
    plt.plot(t, x_sim[:,9], 'b', linewidth=0.8)
    plt.plot(t, np.linalg.norm(x_sim[:,7:10], axis=1), 'c', linewidth=0.8, label='vmax')
    plt.plot(t, -np.linalg.norm(x_sim[:,7:10], axis=1), 'c', linewidth=0.8, label='vmax')
    plt.plot(t, yref_sim[:,7], 'r--', linewidth=0.8)
    plt.plot(t, yref_sim[:,8], 'g--', linewidth=0.8)
    plt.plot(t, yref_sim[:,9], 'b--', linewidth=0.8)
    plt.plot(t, np.linalg.norm(yref_sim[:,7:10], axis=1), 'c--', linewidth=0.8, label='vmax_ref')
    plt.plot(t, -np.linalg.norm(yref_sim[:,7:10], axis=1), 'c--', linewidth=0.8, label='vmax_ref')
    plt.plot(t, np.repeat(v_max, repeats=len(t)), 'k--', linewidth=0.8, label='vmax')
    plt.plot(t, -np.repeat(v_max, repeats=len(t)), 'k--', linewidth=0.8, label='vmax')


    plt.title('Velocity xyz')

    plt.subplot(244)
    plt.plot(t, x_sim[:,10], 'r', linewidth=0.8)
    plt.plot(t, x_sim[:,11], 'g', linewidth=0.8)
    plt.plot(t, x_sim[:,12], 'b', linewidth=0.8)
    plt.plot(t, yref_sim[:,10], 'r--', linewidth=0.8)
    plt.plot(t, yref_sim[:,11], 'g--', linewidth=0.8)
    plt.plot(t, yref_sim[:,12], 'b--', linewidth=0.8)
    plt.title('Angle rate xyz')

    plt.subplot(245)
    plt.plot(t, u_sim[:,0], 'r', linewidth=0.8)
    plt.plot(t, u_sim[:,1], 'g', linewidth=0.8)
    plt.plot(t, u_sim[:,2], 'b', linewidth=0.8)
    plt.plot(t, u_sim[:,3], 'c', linewidth=0.8)
    plt.ylim([0,1.2])
    plt.title('Control u')

    plt.subplot(246)
    plt.plot(solution_times, linewidth=0.8)
    #plt.plot([quad_opt.optimization_dt]*len(solution_times))
    plt.title(f'Total optimization time: {np.sum(solution_times).round(2)}s')
    #plt.legend(('MPC solution time', 'quad_opt.optimization_dt'))
    

    plt.subplot(247)
    plt.plot(cost_solutions, linewidth=0.8)
    #plt.plot([quad_opt.optimization_dt]*len(solution_times))
    plt.title('Cost of solution')

    plt.subplot(248)
    plt.plot(t, aero_drag_sim[:,0], 'r', linewidth=0.8)
    plt.plot(t, aero_drag_sim[:,1], 'g', linewidth=0.8)
    plt.plot(t, aero_drag_sim[:,2], 'b', linewidth=0.8)
    #plt.plot(rmse_pos, linewidth=0.8)
    #plt.plot([quad_opt.optimization_dt]*len(solution_times))
    #plt.title('Position RMSE')
    #plt.legend(('MPC solution time', 'quad_opt.optimization_dt'))
    
    
    
    plt.tight_layout()

    
    plt.savefig(args.plot_output, format="pdf", bbox_inches="tight")
    print(f'Saved generated figure to {args.plot_output}')






    plt.show()

if __name__ == '__main__':
    main()