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

import cv2 as cv
import numpy as np

from skimage.util import img_as_float, random_noise


class Attacker:
    @staticmethod
    def apply_attacks(image):
        img_float = img_as_float(image)
        return [
            (random_noise(img_float, mode='gaussian', var=0.1) * 255).astype(np.uint8),
            (random_noise(img_float, mode='gaussian', var=0.5) * 255).astype(np.uint8),
            (random_noise(img_float, mode='s&p', amount=0.02) * 255).astype(np.uint8),
            (random_noise(img_float, mode='s&p', amount=0.06) * 255).astype(np.uint8),
            (random_noise(img_float, mode='poisson') * 255).astype(np.uint8),
            cv.medianBlur(image, 3),
            cv.medianBlur(image, 5),
            cv.equalizeHist(image),
            cv.convertScaleAbs(image, alpha=1.5, beta=0),
        ]

    @classmethod
    def evaluate_robustness(cls, engine, watermarked,  alpha):
        attacked_images = cls.apply_attacks(watermarked)
        nccs = []

        for attacked in attacked_images:
            reconstructed = engine.decode(attacked, alpha)
            ncc = np.corrcoef(reconstructed.flatten(), engine.watermark_ref.flatten())[0, 1]
            nccs.append(ncc)

        return np.array(nccs)