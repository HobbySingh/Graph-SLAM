import numpy as np

from pose_se2 import PoseSE2
import matplotlib.pyplot as plt

#: The difference that will be used for numerical differentiation
EPSILON = 1e-6


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

    def calc_error(self):

        return self.estimate - (self.vertices[1].pose - self.vertices[0].pose)

    def calc_chi2(self):

        err = self.calc_error()
        return np.transpose(err.arr) @ self.information @ err.arr

    def plot(self, color="b"):

        # import ipdb; ipdb.set_trace()
        xy = np.array([v.pose.position for v in self.vertices])
        plt.plot(xy[:, 0], xy[:, 1], color=color)

    def calc_chi2_gradient_hessian(self):
        r"""Calculate the edge's contributions to the graph's :math:`\chi^2` error, gradient (:math:`\mathbf{b}`), and Hessian (:math:`H`).

        Returns
        -------
        float
            The :math:`\chi^2` error for the edge
        dict
            The edge's contribution(s) to the gradient
        dict
            The edge's contribution(s) to the Hessian

        """
        chi2 = self.calc_chi2()

        err = self.calc_error()

        jacobians = self.calc_jacobians()

        return (
            chi2,
            {
                v.index: np.dot(
                    np.dot(np.transpose(err.arr.squeeze()), self.information), jacobian
                )
                for v, jacobian in zip(self.vertices, jacobians)
            },
            {
                (self.vertices[i].index, self.vertices[j].index): np.dot(
                    np.dot(np.transpose(jacobians[i]), self.information), jacobians[j]
                )
                for i in range(len(jacobians))
                for j in range(i, len(jacobians))
            },
        )

    def calc_jacobians(self):
        r"""Calculate the Jacobian of the edge's error with respect to each constrained pose.

        .. math::

           \frac{\partial}{\partial \Delta \mathbf{x}^k} \left[ \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \right]


        Returns
        -------
        list[np.ndarray]
            The Jacobian matrices for the edge with respect to each constrained pose

        """
        err = self.calc_error()

        # The dimensionality of the compact pose representation
        dim = len(self.vertices[0].pose.arr)

        return [self._calc_jacobian(err, dim, i) for i in range(len(self.vertices))]

    def _calc_jacobian(self, err, dim, vertex_index):
        r"""Calculate the Jacobian of the edge with respect to the specified vertex's pose.

        Parameters
        ----------
        err : np.ndarray
            The current error for the edge (see :meth:`BaseEdge.calc_error`)
        dim : int
            The dimensionality of the compact pose representation
        vertex_index : int
            The index of the vertex (pose) for which we are computing the Jacobian

        Returns
        -------
        np.ndarray
            The Jacobian of the edge with respect to the specified vertex's pose

        """
        jacobian = np.zeros(err.arr.squeeze().shape + (dim,))
        p0 = self.vertices[vertex_index].pose.copy()
        # breakpoint()
        for d in range(dim):
            # update the pose
            delta_pose = np.zeros(dim)
            delta_pose[d] = EPSILON
            self.vertices[vertex_index].pose += PoseSE2.from_array(delta_pose)

            # compute the numerical derivative
            jacobian[:, d] = (self.calc_error().arr - err.arr).squeeze() / EPSILON

            # restore the pose
            self.vertices[vertex_index].pose = p0.copy()

        return jacobian

