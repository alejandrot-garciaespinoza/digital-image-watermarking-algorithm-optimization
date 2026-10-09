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

from src.optimizers.base import BaseMetaheuristic


class Jaya(BaseMetaheuristic):
    def step(self):
        worst_idx = np.argmax(self.fitness)
        worst_candidate = self.population[worst_idx]

        r1 = np.random.rand(self.pop_size, self.dimensions)
        r2 = np.random.rand(self.pop_size, self.dimensions)

        new_population = self.population + \
                         r1 * (self.best_candidate - np.abs(self.population)) - \
                         r2 * (worst_candidate - np.abs(self.population))

        new_population = np.clip(new_population, self.lower_bound, self.upper_bound)

        new_fitness = self.objective_fn(new_population)
        improved_mask = new_fitness < self.fitness

        self.population[improved_mask] = new_population[improved_mask]
        self.fitness[improved_mask] = new_fitness[improved_mask]