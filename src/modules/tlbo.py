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

from src.modules.base import BaseMetaheuristic


class TLBO(BaseMetaheuristic):
    def step(self):
        # ==========================================
        # PHASE 1: TEACHER PHASE
        # ==========================================
        mean = self.population.mean(axis=0)

        tf = np.random.randint(1, 2, (self.pop_size, 1))
        r1 = np.random.rand(self.pop_size, self.dimensions)

        teacher_phase_pop = self.population + r1 * (self.best_candidate - tf * mean)
        teacher_phase_pop = np.clip(teacher_phase_pop, self.lower_bound, self.upper_bound)

        teacher_phase_fitness = self.objective_fn(teacher_phase_pop)
        teacher_phase_improved = teacher_phase_fitness < self.fitness
        self.population[teacher_phase_improved] = teacher_phase_pop[teacher_phase_improved]

        # ==========================================
        # PHASE 2: LEARNER PHASE
        # ==========================================

        # We assign a random peer to every student (ensuring peer != self)
        # We do this by creating a random offset and adding it to the current index
        offsets = np.random.randint(1, self.pop_size, (self.pop_size,))
        indices = np.arange(self.pop_size)
        peer_indices = (indices + offsets) % self.pop_size

        peer_candidates = self.population[peer_indices]
        peer_fitness = self.fitness[peer_indices]

        r2 = np.random.rand(self.pop_size, self.dimensions)

        # Determine direction: towards peer if peer is better, away if peer is worse
        # Unsqueeze is used to broadcast the 1D boolean mask across all dimensions
        am_i_better = np.expand_dims(self.fitness < peer_fitness, axis=1)

        direction = np.where(
            am_i_better,
            self.population - peer_candidates, # I am better, peer moves away from me
            peer_candidates - self.population # Peer is better, I move towards peer
        )

        learner_phase_pop = self.population + r2 * direction
        learner_phase_pop = np.clip(learner_phase_pop, self.lower_bound, self.upper_bound)

        learner_phase_fitness = self.objective_fn(learner_phase_pop)
        learner_phase_improved = learner_phase_fitness < self.fitness
        self.population[learner_phase_improved] = learner_phase_pop[learner_phase_improved]
        self.fitness[learner_phase_improved] = learner_phase_fitness[learner_phase_improved]