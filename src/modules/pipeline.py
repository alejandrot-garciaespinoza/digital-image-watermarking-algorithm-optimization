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

import time

import numpy as np
import skimage as ski

from src.optimizers.jaya import Jaya
from src.optimizers.tlbo import TLBO


class WatermarkPipeline:
    def __init__(self, engine, attacker, size = 5, bounds = (-5, 5)):
        self.engine = engine
        self.attacker = attacker
        self.size = size
        self.bounds = bounds
        self.dimensions = 1

        self.algorithms = {
            'jaya': Jaya,
            'tlbo': TLBO,
        }

    def _evaluate_single(self, alpha):
        watermarked = self.engine.encode(alpha)
        reconstructed = self.engine.decode(watermarked, alpha)

        mse = ski.metrics.mean_squared_error(self.engine.watermark_ref, reconstructed)
        nccs = self.attacker.evaluate_robustness(self.engine, watermarked, alpha)

        mean_ncc = np.mean(nccs)
        return mse + (1 - mean_ncc)

    def _objective(self, population):
        return np.array([self._evaluate_single(candidate[0]) for candidate in population])

    def execute(self, host, watermark, epochs = 30, heuristic='jaya'):
        if heuristic.lower() not in self.algorithms:
            raise ValueError(f'Invalid heuristic: {heuristic}. Choose from: {list(self.algorithms.keys())}')

        start_time = time.time()

        self.engine.precompute(host, watermark)

        AlgorithmClass = self.algorithms[heuristic.lower()]
        optimizer = AlgorithmClass(
            objective_fn=self._objective,
            pop_size=self.size,
            dimensions=self.dimensions,
            bounds=self.bounds,
        )

        best_candidate, best_score = optimizer.optimize(epochs=epochs)
        best_alpha = best_candidate[0]

        elapsed_time = time.time() - start_time

        watermarked = self.engine.encode(best_alpha)
        reconstructed = self.engine.decode(watermarked, best_alpha)

        return {
            'Best_Alpha': best_alpha,
            'Score': best_score,
            'Time': elapsed_time,
            'PSNR': ski.metrics.peak_signal_noise_ratio(self.engine.watermark_ref, reconstructed),
            'SSIM': ski.metrics.structural_similarity(self.engine.watermark_ref, reconstructed, data_range=255),
        }
