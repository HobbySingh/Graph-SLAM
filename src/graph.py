import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
class Graph:

    def __init__(self, edges, vertices):

        self._edges = edges
        self._vertices = vertices

        self._chi2 = None
        self._gradient = None
        self._hessian = None

        self._link_edges()
    
    def _link_edges(self):

        index_id_dict = {i: v.id for i, v in enumerate(self._vertices)}
        id_index_dict = {v_id: v_index for v_index, v_id in index_id_dict.items()}

        for v in self._vertices:
            v.index = id_index_dict[v.id]

        for e in self._edges:
            e.vertices = [self._vertices[id_index_dict[v_id]] for v_id in e.vertex_ids]

    def calc_chi2(self):

        self._chi2 = sum((e.calc_chi2() for e in self._edges))
        return self._chi2

    def optimize(self):

        #TODO:

    def plot(self, vertex_color='r', vertex_maker='o', vertex_markersize=3, edge_color='b', title=None):

        fig = plt.figure()
        
        for e in self._edges:
            e.plot(edge_color)
            # xy = np.array([self._vertices[v_indx].pose.position for v_indx in e.vertex_ids])
            # plt.plot(xy[:, 0], xy[:, 1], color=edge_color)

        for v in self._vertices:
            v.plot(vertex_color, vertex_maker, vertex_markersize)
        
        plt.show()