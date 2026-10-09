import numpy as np

from src.modules.jaya import Jaya
from src.modules.tlbo import TLBO


def rosenbrock(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0 + (1 - x[:, :-1]) ** 2.0, axis=1)

def main():
    # Initialize and run
    optimizer = Jaya(
    # optimizer = TLBO(
        objective_fn=rosenbrock,
        pop_size=50,
        dimensions=3,
        bounds=(-100, 100),
    )

    best_params, best_score = optimizer.optimize(epochs=200)

    print("\nOptimization Complete!")
    print(f"Optimal parameters: {best_params}")
    print(f"Optimal Score: {best_score}")

if __name__ == "__main__":
    main()