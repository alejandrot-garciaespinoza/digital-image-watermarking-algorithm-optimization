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

import pywt


class WatermarkEngine:
    def __init__(self, wavelet='haar', level=1):
        self.level = level
        self.wavelet = wavelet

        self.coeffs = None
        self.U_host, self.S_host, self.Vh_host = None, None, None
        self.U_wm, self.S_wm, self.Vh_wm = None, None, None
        self.watermark_ref = None

    def precompute(self, image, watermark):
        ll_coeffs, self.coeffs = pywt.wavedec2(image, wavelet=self.wavelet, level=self.level)
        width, height = ll_coeffs.shape
        watermark = cv.resize(watermark, (width, height))

        self.U_host, self.S_host, self.Vh_host = np.linalg.svd(ll_coeffs, full_matrices=False)
        self.U_wm, self.S_wm, self.Vh_wm = np.linalg.svd(watermark, full_matrices=False)
        self.watermark_ref = watermark

    def encode(self, alpha):
        S_new = self.S_host + (alpha * self.S_wm)
        ll_watermarked = (self.U_host * S_new) @ self.Vh_host

        watermarked = pywt.waverec2((ll_watermarked, self.coeffs), self.wavelet)
        return np.clip(np.round(watermarked), 0, 255).astype(np.uint8)

    def decode(self, watermarked, alpha):
        watermarked_ll, _ = pywt.wavedec2(watermarked, self.wavelet, level=self.level)
        _, S_watermarked, _ = np.linalg.svd(watermarked_ll, full_matrices=False)

        S_reconstructed = (S_watermarked - self.S_host) / alpha
        reconstructed = (self.U_wm * S_reconstructed) @ self.Vh_wm
        return np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)

