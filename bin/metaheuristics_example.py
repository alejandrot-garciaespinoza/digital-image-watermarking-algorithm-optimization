# Copyright 2026 Alejandro Tonatiuh García Espinoza
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from src.modules.jaya import Jaya
# from src.modules.tlbo import TLBO


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