import numpy as np


def build_qubo_matrix(returns, covariance, k, penalty=2.0, risk_weight=1.0, return_weight=1.0):
    n = len(returns)

    qubo = np.zeros((n, n))

    # risk and return part
    for i in range(n):
        for j in range(n):
            qubo[i][j] += risk_weight * covariance[i][j]

    for i in range(n):
        qubo[i][i] -= return_weight * returns[i]

    # penalty for choosing more or fewer than k asset
    for i in range(n):
        qubo[i][i] += penalty * (1 - 2 * k)

    for i in range(n):
        for j in range(i + 1, n):
            qubo[i][j] += 2 * penalty

    return qubo
