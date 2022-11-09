from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np

import matplotlib.pyplot as plt

from utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q




 

def main():
    file = open("mpc_log.p",'rb')
    [x, u, quad] = pickle.load(file)

    '''
    real_time_artists = initialize_drone_plotter(n_props=50, quad_rad=quad.length,
                                                 world_rad=3)

    fig, ax, artists, background, world_rad = real_time_artists
    '''

    '''
    drone_sketch_artist = artists["drone"] if "drone" in artists.keys() else []
    drone_sketch_artist_x_motor = artists["drone_x"] if "drone_x" in artists.keys() else[]

    drone_art = draw_drone(x[0:3,0], x[3:7,0], quad.x_f, quad.y_f)

    drone_sketch_artist_x_motor.set_data(drone_art[0][0], drone_art[1][0])
    drone_sketch_artist_x_motor.set_3d_properties(1)
    '''
    
    '''
    drone_sketch_artist.set_data(drone_art[0], drone_art[1])
    drone_sketch_artist.set_3d_properties(drone_art[2])
    ax.draw_artist(drone_sketch_artist)
    ax.draw_artist(drone_sketch_artist_x_motor)
    '''
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    line, = [ax.plot([0,0],[0,1],[0,1])[0]]


    #point.set_data([0,0],[0,0])


    ani=FuncAnimation(fig, lambda i :line.set_data([i,0],[0,0]))
    plt.show()
    plt.draw()

    '''
    

    pos = x[::1,0:3]


    def update_lines(num, pos, lines):
        for line, pos in zip(lines, [pos]):
            # NOTE: there is no .set_data() for 3 dim data...

            sketch = pos[num-10:num,:]
            print(sketch)
            line.set_data(sketch[:,:2].T)
            line.set_3d_properties(sketch[:, 2])
        return lines
    


    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")


    # Create lines initially without data
    lines = [ax.plot([], [], [])[0]]

    ax.plot(pos[:,0], pos[:,1], pos[:,2], '--')

    ax.set_xlim([min(x[:,0]), max(x[:,0])])
    ax.set_ylim([min(x[:,1]), max(x[:,1])])
    ax.set_zlim([min(x[:,2]), max(x[:,2])])


    #ax.set_xlim([0, 1])
    #ax.set_ylim([0, 1])
    #ax.set_zlim([0, 1])

    # Creating the Animation object
    ani = FuncAnimation(
        fig, update_lines, x.shape[0], fargs=(pos, lines), interval=0.01)

    plt.show()
    

    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #plt.plot(x[:,0], x[:,1], x[:,2])

    line, = ax.plot(x[0:1,0], x[0:1,1], x[0:1,2])

    quad = ax.scatter([], [], [])


    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    ax.set_zlim([-10,10])
    def update_points(idx, x, line, quad):
        
        # update properties
        line.set_data(x[:idx,0], x[:idx,1])
        line.set_3d_properties(x[:idx,2])
        
        
        offsets = (x[idx,0], x[idx,1], x[idx,2])
        print(offsets)
        plt.plot(x[idx,0], x[idx,1], x[idx,2])
        print(ax.collections[-1].remove())
        
        #quad._offsets3d = offsets

        
        
    ani=animation.FuncAnimation(fig, update_points, fargs=(x, line, quad))
    #ani.save(r'AnimationNew.mp4')
    plt.show()
    '''


def initialize_drone_plotter(world_rad, quad_rad, n_props, full_traj=None):

    fig = plt.figure(figsize=(10, 10), dpi=96)
    fig.show()

    mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())

    ax = fig.add_subplot(111, projection='3d')

    if full_traj is not None:
        ax.plot(full_traj[:, 0], full_traj[:, 1], full_traj[:, 2], '--', color='tab:blue', alpha=0.5)
        ax.set_xlim([ax.get_xlim()[0] - 2 * quad_rad, ax.get_xlim()[1] + 2 * quad_rad])
        ax.set_ylim([ax.get_ylim()[0] - 2 * quad_rad, ax.get_ylim()[1] + 2 * quad_rad])
        ax.set_zlim([ax.get_zlim()[0] - 2 * quad_rad, ax.get_zlim()[1] + 2 * quad_rad])
    else:
        ax.set_xlim([-world_rad, world_rad])
        ax.set_ylim([-world_rad, world_rad])
        ax.set_zlim([-world_rad, world_rad])

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    fig.canvas.draw()
    plt.draw()

    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)

    artists = {
        "trajectory": ax.plot([], [], [])[0], 
        "drone": ax.plot([], [], [], 'o-')[0],
        "drone_x": ax.plot([], [],[], 'o-', color='r')[0],
        "missing_targets": ax.plot([], [], [], color='r', marker='o', linestyle='None', markersize=12)[0],
        "reached_targets": ax.plot([], [], [], color='g', marker='o', linestyle='None', markersize=12)[0],
        "sim_trajectory": [ax.plot([], [], [], '-', color='tab:blue', alpha=0.9 - i * 0.2 / n_props)[0]
                           for i in range(n_props)],
        "int_trajectory": [ax.plot([], [], [], '-', color='tab:orange', alpha=0.9 - i * 0.5 / n_props)[0]
                           for i in range(n_props + 1)],
        "prop_trajectory": [ax.plot([], [], [], '-', color='tab:red', alpha=0.9 - i * 0.2 / n_props)[0]
                            for i in range(n_props)],
        "prop_covariance": [ax.plot([], [], [], color='r', alpha=0.5 - i * 0.45 / n_props)[0]
                            for i in range(n_props)],
        "projection_lines": [ax.plot([], [], [], '-', color='tab:blue', alpha=0.2)[0],
                             ax.plot([], [], [], '-', color='tab:blue', alpha=0.2)[0]],
        "projection_target": [ax.plot([], [], [], marker='o', color='r', linestyle='None', alpha=0.2)[0],
                              ax.plot([], [], [], marker='o', color='r', linestyle='None', alpha=0.2)[0]]}

    art_pack = fig, ax, artists, background, world_rad
    return art_pack


def draw_drone(pos, q_rot, x_f, y_f):

    # Define quadrotor extremities in body reference frame
    x1 = np.array([x_f[0], y_f[0], 0])
    x2 = np.array([x_f[1], y_f[1], 0])
    x3 = np.array([x_f[2], y_f[2], 0])
    x4 = np.array([x_f[3], y_f[3], 0])

    # Convert to world reference frame and add quadrotor center point:
    x1 = v_dot_q(x1, q_rot) + pos
    x2 = v_dot_q(x2, q_rot) + pos
    x3 = v_dot_q(x3, q_rot) + pos
    x4 = v_dot_q(x4, q_rot) + pos

    # Build set of coordinates for plotting
    return ([x1[0], x3[0], pos[0], x2[0], x4[0]],
            [x1[1], x3[1], pos[1], x2[1], x4[1]],
            [x1[2], x3[2], pos[2], x2[2], x4[2]])



if __name__ == "__main__":
    main()