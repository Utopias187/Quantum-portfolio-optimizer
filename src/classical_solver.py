from itertools import combinations
import numpy as np


def get_portfolio_score(x, returns, covariance, risk_weight=1.0, return_weight=1.0):
    x = np.array(x)

    risk = x.T @ covariance @ x
    expected_return = returns @ x

    score = (risk_weight * risk) - (return_weight * expected_return)

    return score, risk, expected_return


def brute_force_solver(returns, covariance, k, risk_weight=1.0, return_weight=1.0):
    n = len(returns)

    best_x = None
    best_score = None
    best_risk = None
    best_return = None

    results = []

    for combo in combinations(range(n), k):
        x = np.zeros(n, dtype=int)

        for index in combo:
            x[index] = 1

        score, risk, expected_return = get_portfolio_score(
            x,
            returns,
            covariance,
            risk_weight,
            return_weight
        )

        results.append({
            "bitstring": "".join(str(i) for i in x),
            "score": score,
            "risk": risk,
            "return": expected_return
        })

        if best_score is None or score < best_score:
            best_score = score
            best_x = x
            best_risk = risk
            best_return = expected_return

    return best_x, best_score, best_risk, best_return, results