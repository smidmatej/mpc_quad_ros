
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.gridspec as gridspec


from utils.save_dataset import load_dict
import os

def main ():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    trajectory_filename = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'data', 'simulated_trajectory.pkl')
    result_plot_filename = os.path.join(dir_path, '..', 'outputs', 'gazebo_simulation', 'img', 'executed_trajectory.pdf')


    data_dict = load_dict(trajectory_filename)


    print(f'Data dict keys: {data_dict.keys()}')
    v_norm = np.linalg.norm(data_dict['x_odom'][:,7:10], axis=1)
    v_ref_norm = np.linalg.norm(data_dict['x_ref'][:,7:10], axis=1)

    rms_pos_ref = np.sqrt(np.mean((data_dict['x_odom'][:,0:3] - data_dict['x_ref'][:,0:3])**2, axis=1))
    rms_vel_ref = np.sqrt(np.mean((data_dict['x_odom'][:,7:10] - data_dict['x_ref'][:,7:10])**2, axis=1))
    rms_quat_ref = np.sqrt(np.mean((data_dict['x_odom'][:,3:7] - data_dict['x_ref'][:,3:7])**2, axis=1))
    rms_rate_ref = np.sqrt(np.mean((data_dict['x_odom'][:,10:13] - data_dict['x_ref'][:,10:13])**2, axis=1))

    rms_total_ref = np.sqrt(np.mean((data_dict['x_odom'] - data_dict['x_ref'])**2, axis=1))


    # Color scheme convert from [0,255] to [0,1]
    cs_u = [[x/256 for x in (8, 65, 92)], \
            [x/256 for x in (204, 41, 54)], \
            [x/256 for x in (118, 148, 159)], \
            [x/256 for x in (232, 197, 71)]] 

    cs_rgb = [[x/256 for x in (205, 70, 49)], \
              [x/256 for x in (105, 220, 158)], \
              [x/256 for x in (102, 16, 242)], \
              [x/256 for x in (7, 59, 58)]]







    plt.style.use('fast')
    sns.set_style("whitegrid")
    

    fig = plt.figure(figsize=(20, 10))
    #gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1, 1])
    gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[0, 3])
    ax5 = plt.subplot(gs[1, 0])
    ax6 = plt.subplot(gs[1, 1])
    ax7 = plt.subplot(gs[1, 2])
    ax8 = plt.subplot(gs[1, 3])
    ax9 = plt.subplot(gs[2, 0])
    ax10 = plt.subplot(gs[2, 1])
    ax11 = plt.subplot(gs[2, 2])
    ax12 = plt.subplot(gs[2, 3])
    



    ax1.plot(data_dict['t_odom'], data_dict['x_odom'][:,0], label='x', color=cs_rgb[0])
    ax1.plot(data_dict['t_odom'], data_dict['x_odom'][:,1], label='y', color=cs_rgb[1])
    ax1.plot(data_dict['t_odom'], data_dict['x_odom'][:,2], label='z', color=cs_rgb[2])
    ax1.plot(data_dict['t_odom'], data_dict['x_ref'][:,0], label='x_ref', color=cs_rgb[0], linestyle='dashed')
    ax1.plot(data_dict['t_odom'], data_dict['x_ref'][:,1], label='y_ref', color=cs_rgb[1], linestyle='dashed')
    ax1.plot(data_dict['t_odom'], data_dict['x_ref'][:,2], label='z_ref', color=cs_rgb[2], linestyle='dashed')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [m]')
    ax1.set_title('Position')

    ax2.plot(data_dict['t_odom'], data_dict['x_odom'][:,3], label='qw', color=cs_rgb[0])
    ax2.plot(data_dict['t_odom'], data_dict['x_odom'][:,4], label='qx', color=cs_rgb[1])
    ax2.plot(data_dict['t_odom'], data_dict['x_odom'][:,5], label='qy', color=cs_rgb[2])
    ax2.plot(data_dict['t_odom'], data_dict['x_odom'][:,6], label='qz', color=cs_rgb[3])
    ax2.plot(data_dict['t_odom'], data_dict['x_ref'][:,3], label='qw_ref', color=cs_rgb[0], linestyle='dashed')
    ax2.plot(data_dict['t_odom'], data_dict['x_ref'][:,4], label='qx_ref', color=cs_rgb[1], linestyle='dashed')
    ax2.plot(data_dict['t_odom'], data_dict['x_ref'][:,5], label='qy_ref', color=cs_rgb[2], linestyle='dashed')
    ax2.plot(data_dict['t_odom'], data_dict['x_ref'][:,6], label='qz_ref', color=cs_rgb[3], linestyle='dashed')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Quaternion')
    ax2.set_title('Orientation')

    ax3.plot(data_dict['t_odom'], data_dict['x_odom'][:,7], label='vx', color=cs_rgb[0])
    ax3.plot(data_dict['t_odom'], data_dict['x_odom'][:,8], label='vy', color=cs_rgb[1])
    ax3.plot(data_dict['t_odom'], data_dict['x_odom'][:,9], label='vz', color=cs_rgb[2])
    ax3.plot(data_dict['t_odom'], data_dict['x_ref'][:,7], label='vx_ref', color=cs_rgb[0], linestyle='dashed')
    ax3.plot(data_dict['t_odom'], data_dict['x_ref'][:,8], label='vy_ref', color=cs_rgb[1], linestyle='dashed')
    ax3.plot(data_dict['t_odom'], data_dict['x_ref'][:,9], label='vz_ref', color=cs_rgb[2], linestyle='dashed')

    ax3.plot(data_dict['t_odom'], v_norm, label='v_norm', color=cs_rgb[3])
    ax3.plot(data_dict['t_odom'], v_ref_norm, label='v_ref_norm', color=cs_rgb[3], linestyle='dashed')
    ax3.set_title('Velocity')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity [m/s]')

    ax4.plot(data_dict['t_odom'], data_dict['x_odom'][:,10], label='wx', color=cs_rgb[0])
    ax4.plot(data_dict['t_odom'], data_dict['x_odom'][:,11], label='wy', color=cs_rgb[1])
    ax4.plot(data_dict['t_odom'], data_dict['x_odom'][:,12], label='wz', color=cs_rgb[2])
    ax4.plot(data_dict['t_odom'], data_dict['x_ref'][:,10], label='wx_ref', color=cs_rgb[0], linestyle='dashed')
    ax4.plot(data_dict['t_odom'], data_dict['x_ref'][:,11], label='wy_ref', color=cs_rgb[1], linestyle='dashed')
    ax4.plot(data_dict['t_odom'], data_dict['x_ref'][:,12], label='wz_ref', color=cs_rgb[2], linestyle='dashed')
    ax4.set_title('Angular Velocity')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Angular Velocity [rad/s]')


    ax5.plot(data_dict['t_odom'], rms_pos_ref, label='rms_pos_ref', color=cs_rgb[0])
    ax5.set_title('RMS Position Error')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('RMS Position Error [m]')

    ax6.plot(data_dict['t_odom'], rms_quat_ref, label='rms_quat_ref', color=cs_rgb[0])
    ax6.set_title('RMS Quaternion Error')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('RMS Quaternion Error [m/s]')

    ax7.plot(data_dict['t_odom'], rms_vel_ref, label='rms_vel_ref', color=cs_rgb[0])
    ax7.set_title('RMS Velocity Error')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('RMS Velocity Error [m/s]')

    ax8.plot(data_dict['t_odom'], rms_rate_ref, label='rms_rate_ref', color=cs_rgb[0])
    ax8.set_title('RMS Angular Velocity Error')
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('RMS Angular Velocity Error [rad/s]')


    ax9.plot(data_dict['t_odom'], data_dict['w_odom'][:,0], label='u1', color=cs_u[0])
    ax9.plot(data_dict['t_odom'], data_dict['w_odom'][:,1], label='u2', color=cs_u[1])
    ax9.plot(data_dict['t_odom'], data_dict['w_odom'][:,2], label='u3', color=cs_u[2])
    ax9.plot(data_dict['t_odom'], data_dict['w_odom'][:,3], label='u4', color=cs_u[3])
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Control Input')
    ax9.set_title('Control Input')


    print(type(data_dict['t_cpu'][:]))
    ax10.plot(data_dict['t_cpu'][:]*1e3, label='t_cpu', color=cs_rgb[0])
    ax10.set_xlabel('Time [s]')
    ax10.set_ylabel('CPU Time [ms]')
    ax10.set_title('MPC CPU Time')

    ax11.plot(data_dict['cost_solution'][:], label='solution_cost', color=cs_rgb[0])
    ax11.set_xlabel('Time [s]')
    ax11.set_ylabel('Solution Cost')
    ax11.set_title('Solution Cost')

    
    

    plt.tight_layout()

    plt.show()



    plt.savefig(result_plot_filename, format="pdf", bbox_inches="tight")



    
if __name__ == '__main__':
    main()
