
from pose_se2 import PoseSE2

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
