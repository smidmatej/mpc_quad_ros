import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
class Explorer:
    def __init__(self) -> None:
        velocities_to_explore = [0,5,10,15,20]*3 
        

    def get_explored_velicities_from_gpe(self, gpe):
        explored_velocities = []*len(gpe.gp)
        max_vel_dim = []*len(gpe.gp)
        for gp, n in zip(gpe.gp, len(gpe.gp)):
            explored_velocities[n] = gp.z_train # Body rate velocities of the inducing points
            max_vel_dim[n] = max(abs(explored_velocities[n]))

        # The smallest of the three velocities dictates the max velocities explored
        velocity_explored_in_all_dimensions = min(max_vel_dim)
        
        

        # -------- Plot the inducing points 1D --------
        xyz = ['x','y','z']
        sns.set_theme()
        plt.figure(figsize=(10, 6), dpi=100)
        for col in range(len(explored_velocities)):
            plt.subplot(1,3,col+1)
            plt.plot(np.zeros_like(explored_velocities[:,col]), explored_velocities[:,col])
            plt.ylabel(f'Velocity {xyz[col]} [ms-1]')
        plt.tight_layout()

        

        D_to_explore = (max(De[vx]) + 5, max(De[vy]) + 5, max(De[vz]) + 5) # Here I will probably need to use inducing points
        

