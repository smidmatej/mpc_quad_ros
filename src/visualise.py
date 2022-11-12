
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.gridspec as gridspec


from source.utils.save_dataset import load_dict

def main ():
    data_dict = load_dict('outputs/log.pkl')


    v_norm = np.linalg.norm(data_dict['odom'][:,7:10], axis=1)
    v_ref_norm = np.linalg.norm(data_dict['x_ref'][:,7:10], axis=1)

    rms_pos_ref = np.sqrt(np.mean((data_dict['odom'][:,0:3] - data_dict['x_ref'][:,0:3])**2, axis=1))
    rms_vel_ref = np.sqrt(np.mean((data_dict['odom'][:,7:10] - data_dict['x_ref'][:,7:10])**2, axis=1))
    rms_quat_ref = np.sqrt(np.mean((data_dict['odom'][:,3:7] - data_dict['x_ref'][:,3:7])**2, axis=1))
    rms_rate_ref = np.sqrt(np.mean((data_dict['odom'][:,10:13] - data_dict['x_ref'][:,10:13])**2, axis=1))

    rms_total_ref = np.sqrt(np.mean((data_dict['odom'] - data_dict['x_ref'])**2, axis=1))


    # Color scheme convert from [0,255] to [0,1]
    cs_u = [[x/256 for x in (8, 65, 92)], \
            [x/256 for x in (204, 41, 54)], \
            [x/256 for x in (118, 148, 159)], \
            [x/256 for x in (232, 197, 71)]] 

    cs_rgb = [[x/256 for x in (205, 70, 49, 1)], \
              [x/256 for x in (112, 174, 110, 1)], \
              [x/256 for x in (13, 59, 102, 1)], \
              [x/256 for x in (40, 0, 3, 1)]]







    plt.style.use('fast')
    sns.set_style("whitegrid")
    

    fig = plt.figure(figsize=(10, 10))
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
    


    ax1.plot(data_dict['t'], data_dict['odom'][:,0], label='x', color=cs_rgb[0])
    ax1.plot(data_dict['t'], data_dict['odom'][:,1], label='y', color=cs_rgb[1])
    ax1.plot(data_dict['t'], data_dict['odom'][:,2], label='z', color=cs_rgb[2])
    ax1.plot(data_dict['t'], data_dict['x_ref'][:,0], label='x_ref', color=cs_rgb[0], linestyle='dashed')
    ax1.plot(data_dict['t'], data_dict['x_ref'][:,1], label='y_ref', color=cs_rgb[1], linestyle='dashed')
    ax1.plot(data_dict['t'], data_dict['x_ref'][:,2], label='z_ref', color=cs_rgb[2], linestyle='dashed')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [m]')
    ax1.set_title('Position')

    ax2.plot(data_dict['t'], data_dict['odom'][:,3], label='qw', color='red')
    ax2.plot(data_dict['t'], data_dict['odom'][:,4], label='qx', color='green')
    ax2.plot(data_dict['t'], data_dict['odom'][:,5], label='qy', color='blue')
    ax2.plot(data_dict['t'], data_dict['odom'][:,6], label='qz', color='orange')
    ax2.plot(data_dict['t'], data_dict['x_ref'][:,3], label='qw_ref', color='red', linestyle='dashed')
    ax2.plot(data_dict['t'], data_dict['x_ref'][:,4], label='qx_ref', color='green', linestyle='dashed')
    ax2.plot(data_dict['t'], data_dict['x_ref'][:,5], label='qy_ref', color='blue', linestyle='dashed')
    ax2.plot(data_dict['t'], data_dict['x_ref'][:,6], label='qz_ref', color='orange', linestyle='dashed')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Quaternion')
    ax2.set_title('Orientation')

    ax3.plot(data_dict['t'], data_dict['odom'][:,7], label='vx', color='red')
    ax3.plot(data_dict['t'], data_dict['odom'][:,8], label='vy', color='green')
    ax3.plot(data_dict['t'], data_dict['odom'][:,9], label='vz', color='blue')
    ax3.plot(data_dict['t'], data_dict['x_ref'][:,7], label='vx_ref', color='red', linestyle='dashed')
    ax3.plot(data_dict['t'], data_dict['x_ref'][:,8], label='vy_ref', color='green', linestyle='dashed')
    ax3.plot(data_dict['t'], data_dict['x_ref'][:,9], label='vz_ref', color='blue', linestyle='dashed')

    ax3.plot(data_dict['t'], v_norm, label='v_norm', color='black')
    ax3.plot(data_dict['t'], v_ref_norm, label='v_ref_norm', color='black', linestyle='dashed')
    ax3.set_title('Velocity')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity [m/s]')

    ax4.plot(data_dict['t'], data_dict['odom'][:,10], label='wx', color='red')
    ax4.plot(data_dict['t'], data_dict['odom'][:,11], label='wy', color='green')
    ax4.plot(data_dict['t'], data_dict['odom'][:,12], label='wz', color='blue')
    ax4.plot(data_dict['t'], data_dict['x_ref'][:,10], label='wx_ref', color='red', linestyle='dashed')
    ax4.plot(data_dict['t'], data_dict['x_ref'][:,11], label='wy_ref', color='green', linestyle='dashed')
    ax4.plot(data_dict['t'], data_dict['x_ref'][:,12], label='wz_ref', color='blue', linestyle='dashed')
    ax4.set_title('Angular Velocity')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Angular Velocity [rad/s]')


    ax5.plot(data_dict['t'], rms_pos_ref, label='rms_pos_ref')
    ax5.set_title('RMS Position Error')

    ax9 = plt.subplot(gs[0, 0])
    ax9 = plt.subplot(gs[0, 1])
    ax9 = plt.subplot(gs[1, 0])
    ax9 = plt.subplot(gs[1, 1])
    ax9.plot(data_dict['t'], data_dict['w'][:,0], label='u1', color='red')
    ax9.plot(data_dict['t'], data_dict['w'][:,1], label='u2', color='green')
    ax9.plot(data_dict['t'], data_dict['w'][:,2], label='u3', color='blue')
    ax9.plot(data_dict['t'], data_dict['w'][:,3], label='u4', color='orange')
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Control Input')
    ax9.set_title('Control Input')
    
    


    plt.show()

    plt.tight_layout()

    plot_filename = "outputs/trajectory_tracking.pdf"
    plt.savefig(plot_filename, format="pdf", bbox_inches="tight")



    
if __name__ == '__main__':
    main()
