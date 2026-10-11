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

import concurrent.futures
import os.path

import cv2 as cv
import pandas as pd

from src.modules.attacker import Attacker
from src.modules.engine import WatermarkEngine
from src.modules.pipeline import WatermarkPipeline


def evaluate_imperceptibility(args):
    name, host, watermark, run = args

    engine = WatermarkEngine()
    attacker = Attacker()

    pipeline = WatermarkPipeline(engine, attacker, size=5, bounds=(-5, 5))
    results = pipeline.execute(host, watermark, epochs=30, heuristic='jaya')
    # results = pipeline.execute(host, watermark, epochs=30, heuristic='tlbo')

    return {
        'Name': name,
        'PSNR': results['PSNR'],
        'SSIM': results['SSIM'],
        'Run': run,
    }

def main():
    runs = 5

    hosts = [
        '../data/hosts/barbara.512.tiff',
        '../data/hosts/boat.512.tiff',
        '../data/hosts/elaine.512.tiff',
        '../data/hosts/house1.512.tiff',
        '../data/hosts/house2.512.tif',
        '../data/hosts/pentagon.512.tiff',
        '../data/hosts/pirate.512.tif',
        '../data/hosts/zelda.512.tiff',
    ]

    watermark = cv.imread('../data/watermark/yacht.tiff', cv.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f'Could not load watermark')

    tasks = []
    for host_path in hosts:
        host = cv.imread(host_path, cv.IMREAD_GRAYSCALE)
        if host is None:
            print(f'Could not load host {host_path}. Skipping...')
            continue
        tasks.extend((os.path.basename(host_path), host, watermark, run) for run in range(runs))

    max_workers = 10
    print(f'Starting {len(tasks)} independent runs across {max_workers} CPU cores...')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(evaluate_imperceptibility, tasks))

    print('All runs completed.')


    df = pd.DataFrame(results)

    df.to_csv('jaya_imperceptibility_results.csv', index=False)
    print("Data saved to 'jaya_imperceptibility_results.csv'")

    # ==========================================
    # STATISTICAL AGGREGATION
    # ==========================================

    agg_df = df.groupby('Name').agg(
        Mean_PSNR=('PSNR', 'mean'),
        Std_PSNR=('PSNR', 'std'),
        Mean_SSIM=('SSIM', 'mean'),
        Std_SSIM=('SSIM', 'std'),
    ).reset_index()

    agg_df.to_csv('jaya_imperceptibility_aggregated_results.csv', index=False)
    print("Aggregated data saved to 'jaya_imperceptibility_aggregated_results.csv'")

if __name__ == '__main__':
    main()