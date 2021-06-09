# Original code: https://github.com/ClayFlannigan/icp
# Modified to reject pairs that have greater distance than the specified threshold
# Add covariance check

import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_C_k(point1, point2):
    d = point1 - point2
    alpha = np.pi/2 + np.arctan2(d[1], d[0])
    c = np.cos(alpha)
    s = np.sin(alpha)
    m = np.array([[c*c, c*s],
                  [c*s, s*s]])
    return m

def dC_drho(point1, point2):
    eps = 0.001
    C_k = compute_C_k(point1, point2)
    point1b = point1 + point1 * (eps/np.linalg.norm(point1))
    C_k_eps = compute_C_k(point1b, point2)
    return (1/eps) * (C_k_eps - C_k)

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def vers(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def compute_covariance(laser_ref, laser_sens, t, theta, angles):
    # Reference: https://censi.science/research/robot-perception/icpcov/
    d2J_dxdy1 = np.zeros((3, laser_ref.shape[0]))
    d2J_dxdy2 = np.zeros((3, laser_sens.shape[0]))

    d2J_dt2 = np.zeros((2, 2))
    d2J_dt_dtheta = np.zeros((2,1));
    d2J_dtheta2 = np.zeros((1, 1))

    for i in range(laser_ref.shape[0]-1):
        p_i = laser_sens[i]
        p_j1 = laser_ref[i]
        p_j2 = laser_ref[i+1]
        v1 = rotation_matrix(theta + np.pi/2) @ p_i
        v2 = rotation_matrix(theta) @ p_i + t - p_j1
        v3 = vers(theta + angles[i])
        v4 = vers(theta + angles[i] + np.pi/2)

        C_k = compute_C_k(p_j1, p_j2)
        d2J_dt2_k = 2 * C_k
        d2J_dt_dtheta_k = 2 * (C_k @ v1)

        v_new = rotation_matrix(theta+np.pi) @ p_i

        d2J_dtheta2_k = 2 * ((C_k @ v_new @ v2.T) + (C_k @ v1 @ v1.T))

        d2J_dt2 += d2J_dt2_k
        d2J_dt_dtheta += d2J_dt_dtheta_k[:, np.newaxis]
        d2J_dtheta2 += d2J_dtheta2_k
        d2Jk_dtdrho_i = 2 * C_k @ v3
        d2Jk_dtheta_drho_i = 2 * ((C_k @ v4 @ v2.T) + (C_k @ v1 @ v3.T))
        d2J_dxdy2[:, i] += np.hstack([d2Jk_dtdrho_i, d2Jk_dtheta_drho_i])
        dC_drho_j1 = dC_drho(p_j1, p_j2)
        dC_drho_j2 = dC_drho(p_j2, p_j1)
        v_j1 = vers(angles[i]) # Probably wrong
        d2Jk_dt_drho_j1 = (-2 * (C_k @ v_j1)) + (2 * (dC_drho_j1 @ v2))
        d2Jk_dtheta_drho_j1 = (-2 * (C_k @ v1 @ v_j1.T)) + (dC_drho_j1 @ v1 @ v2.T)
        d2J_dxdy1[:, i] += np.hstack([d2Jk_dt_drho_j1, d2Jk_dtheta_drho_j1])
        d2Jk_dt_drho_j2 = 2 * (dC_drho_j2 @ v2)
        d2Jk_dtheta_drho_j2 =  2 * (dC_drho_j2 @ v1 @ v2.T)
        d2J_dxdy1[:, i+1] += np.hstack([d2Jk_dt_drho_j2, d2Jk_dtheta_drho_j2])

    d2J_dx2 = np.vstack([np.hstack([d2J_dt2, d2J_dt_dtheta]),
                         np.hstack([d2J_dt_dtheta.T, d2J_dtheta2])])

    edx_dy1 = -np.linalg.inv(d2J_dx2) @ d2J_dxdy1
    edx_dy2 = -np.linalg.inv(d2J_dx2) @ d2J_dxdy2

    ecov0_x = edx_dy1 @ edx_dy1.T + edx_dy2 @ edx_dy2.T

    return ecov0_x, edx_dy1, edx_dy2

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # Reject pairs that have 1 meter distance between them
        indices = indices[np.linalg.norm(src[:m, :], axis=0) < 80]
        distances = distances[np.linalg.norm(src[:m, :], axis=0) < 80]
        filtered_src = src[:, np.linalg.norm(src[:m, :], axis=0) < 80]
        indices = indices[distances < 1.0]

        # compute the transformation between
        # the current source and nearest destination points
        T,_,_ = best_fit_transform(filtered_src[:m, distances < 1.0].T, dst[:m,indices].T)

        # distances = distances[distances <= 1.0]

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)


    # FIXME: BROKEN
    # theta = np.arctan2(T[1,0], T[0,0])
    # t = T[0:2, 2]
    # angle_res = 1
    # angles = np.arange(-90, 91, angle_res)
    # cov, a, b = compute_covariance(A, src[:m, :].T, t, theta, np.radians(angles))
    cov = np.eye(3)

    return T, distances, i, cov
