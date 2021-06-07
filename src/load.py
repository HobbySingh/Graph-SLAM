import numpy as np

from edge_odometry import EdgeOdometry
from graph import Graph
from pose_se2 import PoseSE2
from util import upper_triangular_matrix_to_full_matrix
from vertex import Vertex


def data_loader(data_file):
    """
    Generate SE(2) graph from .g2o file
    
    VERTEX_SE2: ID x y th
    EDGE_SE2: IDout IDin dx dy dth I11 I12 I13 I22 I23 I33

    Args:
        data_file (str): Path to .g2o file
    
    Returns:
        graph (Graph): The g2o edges and vertices loaded to our PoseGraph
    """

    edges = []
    vertices = []

    with open(data_file) as file:

        for line in file.readlines():
            line = line.split()

            if line[0] == "VERTEX_SE2":
                vertex_id = int(line[1])
                arr = np.array([float(number) for number in line[2:]], dtype=np.float64)
                p = PoseSE2(arr[:2], arr[2])
                v = Vertex(vertex_id, p)
                vertices.append(v)
                continue

            if line[0] == "EDGE_SE2":

                vertex_ids = [int(line[1]), int(line[2])]
                arr = np.array([float(number) for number in line[3:]], dtype=np.float64)

                estimate = PoseSE2(arr[:2], arr[2])
                information = upper_triangular_matrix_to_full_matrix(arr[3:], 3)
                e = EdgeOdometry(vertex_ids, information, estimate)

                edges.append(e)
                continue

    return Graph(edges, vertices)

