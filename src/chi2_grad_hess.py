from collections import defaultdict
import numpy as np


class _Chi2GradientHessian:
    r"""A class that is used to aggregate the :math:`\chi^2` error, gradient, and Hessian.

    Parameters
    ----------
    dim : int
        The compact dimensionality of the poses

    Attributes
    ----------
    chi2 : float
        The :math:`\chi^2` error
    dim : int
        The compact dimensionality of the poses
    gradient : defaultdict
        The contributions to the gradient vector
    hessian : defaultdict
        The contributions to the Hessian matrix

    """

    def __init__(self, dim):
        self.chi2 = 0.0
        self.dim = dim
        self.gradient = defaultdict(lambda: np.zeros(dim))
        self.hessian = defaultdict(lambda: np.zeros((dim, dim)))

    @staticmethod
    def update(chi2_grad_hess, incoming):
        r"""Update the :math:`\chi^2` error and the gradient and Hessian dictionaries.

        Parameters
        ----------
        chi2_grad_hess : _Chi2GradientHessian
            The ``_Chi2GradientHessian`` that will be updated
        incoming : tuple
        """
        chi2_grad_hess.chi2 += incoming[0]

        for idx, contrib in incoming[1].items():
            chi2_grad_hess.gradient[idx] += contrib

        for (idx1, idx2), contrib in incoming[2].items():
            if idx1 <= idx2:
                chi2_grad_hess.hessian[idx1, idx2] += contrib
            else:
                chi2_grad_hess.hessian[idx2, idx1] += np.transpose(contrib)

        return chi2_grad_hess
