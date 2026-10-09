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

from src.modules.attacker import Attacker
from src.modules.engine import WatermarkEngine
from src.modules.pipeline import WatermarkPipeline


def main():
    host = cv.imread('../data/hosts/barbara.512.tiff', cv.IMREAD_GRAYSCALE)
    watermark = cv.imread('../data/watermark/yacht.tiff', cv.IMREAD_GRAYSCALE)

    engine = WatermarkEngine()
    attacker = Attacker()

    pipeline = WatermarkPipeline(engine, attacker, size=5, bounds=(-5, 5))
    # results = pipeline.execute(host, watermark, epochs=30, heuristic='jaya')
    results = pipeline.execute(host, watermark, epochs=30, heuristic='tlbo')
    print(results)

if __name__ == "__main__":
    main()