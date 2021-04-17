import numpy as np

from pose_se2 import PoseSE2

class EdgeOdometry:

    def __init__(self, vertex_ids, information, estimate, vertices=None):
        """
        A class for representing odometry edges in Graph SLAM

        Args:
            vertex_ids (list[int]): The ids of all vertices constrained by this edge
            information (np.ndarray): The information matrix associated with the edge
            estimate (PoseSE2): The expected measurement 
            vertices (list[graphslam.vertex.Vertex], optional): [description]. Defaults to None.
        """
        self.vertex_ids = vertex_ids
        self.information = information
        self.estimate = estimate
        self.vertices = vertices
