import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler

def make_quadratic_program(qubo):
    n = qubo.shape[0]

    problem = QuadraticProgram()

    for i in range(n):
        problem.binary_var(name=f"x{i}")

    linear = {}
    quadratic = {}

    for i in range(n):
        linear[f"x{i}"] = qubo[i][i]

    for i in range(n):
        for j in range(i + 1, n):
            if qubo[i][j] != 0:
                quadratic[(f"x{i}", f"x{j}")] = qubo[i][j]

    problem.minimize(linear=linear, quadratic=quadratic)

    return problem


def run_qaoa_solver(qubo):
    problem = make_quadratic_program(qubo)

    sampler = Sampler()
    optimizer = COBYLA(maxiter=100)

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=1
    )

    qaoa_solver = MinimumEigenOptimizer(qaoa)
    result = qaoa_solver.solve(problem)

    best_x = np.array([int(v) for v in result.x])
    best_score = result.fval

    return best_x, best_score