import numpy as np

from pose_se2 import PoseSE2
import matplotlib.pyplot as plt

class EdgeOdometry:

    def __init__(self, vertex_ids, information, estimate, vertices=None):
        """
        A class for representing odometry edges in Graph SLAM

        Args:
            vertex_ids (list[int]): The ids of all vertices constrained by this edge
            information (np.ndarray): The information matrix associated with the edge
            estimate (PoseSE2): The expected measurement 
            vertices (list[graphslam.vertex.Vertex], optional): A list of vertices constrained by the edge
        """

        self.vertex_ids = vertex_ids
        self.information = information
        self.estimate = estimate
        self.vertices = vertices

    def plot(self, color='b'):

        # import ipdb; ipdb.set_trace()
        xy = np.array([v.pose.position for v in self.vertices])
        plt.plot(xy[:, 0], xy[:, 1], color=color)
