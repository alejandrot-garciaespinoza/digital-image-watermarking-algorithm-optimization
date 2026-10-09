import numpy as np

from src.modules.base import BaseMetaheuristic


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