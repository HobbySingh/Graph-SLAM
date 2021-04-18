import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
class Graph:

    def __init__(self, edges, vertices):

        self._edges = edges
        self._vertices = vertices
        
    def plot(self, vertex_color='r', vertex_maker='o', vertex_markersize=3, edge_color='b', title=None):

        fig = plt.figure()
        
        for e in self._edges:
            xy = np.array([self._vertices[v_indx].pose.position for v_indx in e.vertex_ids])
            plt.plot(xy[:, 0], xy[:, 1], color=edge_color)

        for v in self._vertices:
            v.plot(vertex_color, vertex_maker, vertex_markersize)
        
        plt.show()