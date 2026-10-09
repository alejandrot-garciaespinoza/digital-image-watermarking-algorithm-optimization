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

from abc import ABC, abstractmethod


class BaseMetaheuristic(ABC):
    def __init__(self, objective_fn, pop_size, dimensions, bounds):
        self.objective_fn = objective_fn
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.bounds = bounds

        if isinstance(bounds, tuple):
            self.lower_bound, self.upper_bound = bounds[0], bounds[1]
        else:
            raise TypeError('bounds must be tuple')

        self.population = self._initialize_population()
        self.fitness = self.objective_fn(self.population)

        self.best_candidate = None
        self.best_fitness = float('inf')
        self._update_global_best()

    def _initialize_population(self):
        # Uniform distribution between lower and upper bounds
        rand_array = np.random.rand(self.pop_size, self.dimensions)
        return self.lower_bound + rand_array * (self.upper_bound - self.lower_bound)

    def _clip_to_bounds(self):
        return np.clip(self.population, self.lower_bound, self.upper_bound)

    def _update_global_best(self):
        current_best_idx = np.argmin(self.fitness)
        current_best_fit = self.fitness[current_best_idx]

        if current_best_fit < self.best_fitness:
            self.best_fitness = current_best_fit
            self.best_candidate = np.copy(self.population[current_best_idx])

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    def optimize(self, epochs):
        for epoch in range(epochs):
            self.step()

            self._clip_to_bounds()

            self.fitness = self.objective_fn(self.population)

            self._update_global_best()

        return self.best_candidate, self.best_fitness