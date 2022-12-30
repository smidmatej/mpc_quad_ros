import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
class Explorer:
    def __init__(self, gpe) -> None:
        velocities_to_explore = [0,5,10,15,20]*3 
        self.desired_explored_vmax = 20
        self.exploration_step = 5

        explored_velocities = self.get_explored_velicities_from_gpe(gpe)
        explored_vmax = self.calculate_explored_vmax(explored_velocities)
        self.velocity_to_explore = self.calculate_velocity_to_explore(explored_vmax)
        print(f'Explored vmax: {explored_vmax}')
        print(f'Velocity to explore: {self.velocity_to_explore}')


    def calculate_velocity_to_explore(self, explored_vmax):
        
        # TODO: Setting vmax for trajectory generation does not guarantee that the max velocity is reached is in the trajectory
        #       Also, the max velocity of the trajectory is not necessarily in the GPE
        if explored_vmax + self.exploration_step < self.desired_explored_vmax:
            velocity_to_explore = explored_vmax + self.exploration_step
        else:
            velocity_to_explore = self.desired_explored_vmax
        return velocity_to_explore


    def calculate_explored_vmax(self, explored_velocities):
        # Calculate the max velocity that was explored
        vmax = 0
        for v in explored_velocities:
            if v['max'] > vmax:
                vmax = v['max']

            if abs(v['min']) > vmax:
                vmax = abs(v['min'])

        return vmax

    def get_explored_velicities_from_gpe(self, gpe):
        # Initialization
        explored_velocities = [None]*len(gpe.gp)
        min_over_all = 1000
        max_over_all = -1000

        # Loop over GPE to find the min and max of each dimension
        for gp, n in zip(gpe.gp, range(len(gpe.gp))):
            explored_velocities[n] = dict()
            explored_velocities[n]['min'] = gp.z_train.min()
            explored_velocities[n]['max'] = gp.z_train.max()

            # Remember the min and max over all dimensions for plotting purposes
            if explored_velocities[n]['min'] < min_over_all:
                min_over_all = explored_velocities[n]['min']

            if explored_velocities[n]['max'] > max_over_all:
                max_over_all = explored_velocities[n]['max']


        # -------- Plot the inducing points 1D --------
        xyz = ['x','y','z']

        sns.set_theme()
        plt.figure(figsize=(10, 6), dpi=100)
        for col in range(len(explored_velocities)):

            print(f'Velocity {xyz[col]}: {explored_velocities[col]}')
            plt.subplot(1,3,col+1)
            plt.plot(np.zeros((2,)), np.array([explored_velocities[col]['min'], explored_velocities[col]['max']]))
            plt.ylabel(f'Velocity {xyz[col]} [ms-1]')
            plt.ylim([min_over_all, max_over_all])
            plt.xticks([])
        plt.tight_layout()
        plt.show()
        
        return explored_velocities

        #D_to_explore = (max(De[vx]) + 5, max(De[vy]) + 5, max(De[vz]) + 5) # Here I will probably need to use inducing points
        

