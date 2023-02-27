"""Takeoff-hover-land for one CF. Useful to validate hardware config."""


import sys
sys.path.append('/home/ja/Projects/MPC_drone/mpc_quad_ros/src/crazyswarm/ros_ws/src/crazyswarm/scripts')

from pycrazyswarm import Crazyswarm

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0


def main():
    swarm_path_yaml = '/home/ja/Projects/MPC_drone/mpc_quad_ros/src/crazyswarm/ros_ws/src/crazyswarm/launch/crazyflies.yaml'
    swarm = Crazyswarm(swarm_path_yaml)
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == "__main__":
    main()
