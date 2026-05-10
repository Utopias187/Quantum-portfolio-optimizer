import numpy as np


def run_qaoa_solver(qubo):
    n = qubo.shape[0]

    best_x = None
    best_score = None

    # simple simulated quantum-style search for now
    for _ in range(500):
        x = np.random.randint(0, 2, size=n)

        score = x.T @ qubo @ x

        if best_score is None or score < best_score:
            best_score = score
            best_x = x

    return best_x, best_score