
from pose_se2 import PoseSE2
import matplotlib.pyplot as plt
class Vertex:

    def __init__(self, vertex_id, pose, vertex_index=None):
        """
        A class for representing a vertex in Graph SLAM

        Args:
            vertex_id (int): The vertex's unique id
            pose (PoseSE2): The pose associated with the vertex 
            vertex_index (int, optional): The vertex's index in the graph's vertices list. Defaults to None.
        """
        self.id = vertex_id
        self.pose = pose
        self.index = vertex_index

    def plot(self, color='r', marker='o', marker_size=3):

        x, y = self.pose.position
        plt.plot(x, y, color=color, marker=marker, markersize=marker_size)