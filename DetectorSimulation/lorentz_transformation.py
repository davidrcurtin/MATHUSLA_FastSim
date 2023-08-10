import numpy as np

c = 1 # speed of light in natural units


def lorentz_contraction(beta: np.ndarray):
    return (1 - np.dot(beta, beta)) ** (-0.5)


def boost_matrix(beta: np.ndarray):
    gamma = lorentz_contraction(beta)
    beta_norm = np.dot(beta, beta)
    boost = np.zeros((4, 4))  # construct boost matrix
    boost[0, 0] = gamma  # L_00
    for i in range(1, 4):
        boost[i, 0] = -gamma * beta[i - 1]  # L_0i = L_i0
        boost[0, i] = boost[i, 0]
        boost[i, i] = (gamma - 1) * beta[i - 1] ** 2 / beta_norm + 1  # L_ii
    for i in range(1, 4):
        for j in range(1, i):
            boost[i, j] = (gamma - 1) * beta[i - 1] * beta[j - 1] / beta_norm  # L_ij = L_ji
            boost[j, i] = boost[i, j]
    return np.linalg.inv(boost)


def boost_1d(beta: float):
    gamma = lorentz_contraction(np.ndarray([beta]))
    boost_x = np.zeros((2, 2))
    boost_x[0, 0], boost_x[1, 1] = gamma, gamma
    boost_x[1, 0], boost_x[0, 1] = -gamma * beta, -gamma * beta
    return boost_x


def rotation(vector: np.ndarray, axis: np.ndarray, angle: float):
    return np.cos(angle) * vector + np.cross(axis, vector) * np.sin(angle) + axis * np.dot(axis, vector) * (1 - np.cos(angle))
