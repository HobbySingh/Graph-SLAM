import numpy as np
import scipy

import icp
from pose_se2 import PoseSE2

"""
Detects loop closures and adds the corresponding edge between the current
pose and detected loop closure pose
Input:
    @param poses all poses excluding current pose
    @param curr_pose pose of the vertex being added to the graph
    @param curr_idx index of current pose
    @param laser laser readings at every pose
    @param g pose graph optimizer class object
"""


def find_loop_closure(curr_pose, curr_idx, laser, g):
    # print("Attempt: Loop Closure")
    kdTreeR = 4.25
    poses = np.array(
        [g.get_pose(idx).arr[0:2] for idx in range(curr_idx - 1)]
    ).squeeze()
    # breakpoint()
    kdTree = scipy.spatial.cKDTree(poses)
    # breakpoint()
    idxs = kdTree.query_ball_point(curr_pose.arr[0:2].T, kdTreeR)[0]

    loopThresh = 0.15
    for i in idxs:
        # breakpoint()
        with np.errstate(all="raise"):
            try:
                tf, dist, _, cov = icp.icp(
                    laser[i],
                    laser[curr_idx],
                    np.eye(3),
                    max_iterations=80,
                    tolerance=0.0001,
                )
            except Exception as e:
                print("ICP Exception", e)
                continue

            if np.mean(dist) < loopThresh:
                # print("Success: Loop Closure")
                g.add_edge(
                    [curr_idx, i], PoseSE2.from_rt_matrix(tf), np.linalg.inv(cov)
                )
