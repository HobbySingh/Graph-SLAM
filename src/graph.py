import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
from functools import reduce
from scipy.sparse import SparseEfficiencyWarning, lil_matrix
from scipy.sparse.linalg import spsolve

from pose_se2 import PoseSE2
from chi2_grad_hess import _Chi2GradientHessian

EPS = np.finfo(float).eps


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
        self._chi2 = sum((e.calc_chi2() for e in self._edges))[0, 0]
        return self._chi2

    def _calc_chi2_grad_hess(self):
        n = len(self._vertices)
        dim = len(self._vertices[0].pose.arr.squeeze())
        chi2_gradient_hessian = reduce(
            _Chi2GradientHessian.update,
            (e.calc_chi2_gradient_hessian() for e in self._edges),
            _Chi2GradientHessian(dim),
        )
        # breakpoint()
        self._chi2 = chi2_gradient_hessian.chi2[0, 0]

        # Populate the gradient vector
        self._gradient = np.zeros(n * dim, dtype=np.float64)
        for idx, contrib in chi2_gradient_hessian.gradient.items():
            self._gradient[idx * dim : (idx + 1) * dim] += contrib

        # Populate the Hessian matrix
        self._hessian = lil_matrix((n * dim, n * dim), dtype=np.float64)
        for (row_idx, col_idx), contrib in chi2_gradient_hessian.hessian.items():
            self._hessian[
                row_idx * dim : (row_idx + 1) * dim, col_idx * dim : (col_idx + 1) * dim
            ] = contrib

            if row_idx != col_idx:
                # mirror the hessian along diagonal
                self._hessian[
                    col_idx * dim : (col_idx + 1) * dim,
                    row_idx * dim : (row_idx + 1) * dim,
                ] = np.transpose(contrib)

    def optimize(self, tol=1e-4, max_iter=40, fix_first_pose=True):
        n = len(self._vertices)
        dim = len(self._vertices[0].pose.arr.squeeze())

        prev_chi2_err = -1

        # For displaying the optimization progress
        print("\nIteration                chi^2        rel. change")
        print("---------                -----        -----------")

        for i in range(max_iter):
            self._calc_chi2_grad_hess()

            # only do this for the 2nd iteration onwards
            if i > 0:
                rel_diff = (prev_chi2_err - self._chi2) / (prev_chi2_err + EPS)
                print(f"{i:9} {self._chi2:20.4f} {-rel_diff:18.6f}")
                if (self._chi2 < prev_chi2_err) and rel_diff < tol:
                    return
            else:
                print(f"{i:9} {self._chi2:20.4f}")

            # Store the prev chi2 error
            prev_chi2_err = self._chi2
            if fix_first_pose:
                self._hessian[:dim, :] = 0.0
                self._hessian[:, :dim] = 0.0
                self._hessian[:dim, :dim] += np.eye(dim)
                self._gradient[:dim] = 0.0

            # run solver
            dx = spsolve(self._hessian, -self._gradient)

            # apply
            for v, dxi in zip(self._vertices, np.split(dx, n)):
                v.pose += PoseSE2.from_array(dxi)
        self.calc_chi2()
        rel_diff = (prev_chi2_err - self._chi2) / (prev_chi2_err + EPS)
        # breakpoint()
        print(f"{i:9} {self._chi2:20.4f} {-rel_diff:18.6f}")

    def plot(
        self,
        vertex_color="r",
        vertex_maker="o",
        vertex_markersize=3,
        edge_color="b",
        title=None,
    ):

        fig = plt.figure()

        for e in self._edges:
            e.plot(edge_color)
            # xy = np.array([self._vertices[v_indx].pose.position for v_indx in e.vertex_ids])
            # plt.plot(xy[:, 0], xy[:, 1], color=edge_color)

        for v in self._vertices:
            v.plot(vertex_color, vertex_maker, vertex_markersize)
        plt.savefig(f"{title}.png")
        plt.show()
