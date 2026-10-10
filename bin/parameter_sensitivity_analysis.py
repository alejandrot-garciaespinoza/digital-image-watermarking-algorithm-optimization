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

import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from src.modules.attacker import Attacker
from src.modules.engine import WatermarkEngine
from src.modules.pipeline import WatermarkPipeline


host = None
watermark = None

def initialize_worker(host_path, watermark_path):
    global host, watermark

    host = cv.imread(host_path, cv.IMREAD_GRAYSCALE)
    watermark = cv.imread(watermark_path, cv.IMREAD_GRAYSCALE)

def evaluate_configuration(args):
    p, t, r = args

    engine = WatermarkEngine()
    attacker = Attacker()

    pipeline = WatermarkPipeline(engine, attacker, size=p, bounds=(-5, 5))
    results = pipeline.execute(host, watermark, epochs=t, heuristic='jaya')
    # results = pipeline.execute(host, watermark, epochs=t, heuristic='tlbo')

    return {
        'Population_Size': p,
        'Max_Iterations': t,
        'Run': r,
        'Objective_Value': results['Score'],
        'Execution_Time': results['Time'],
    }

def main():
    pop_sizes = [5, 10, 15, 20, 25]
    max_iters = [10, 20, 30, 40, 50]
    runs = 5

    tasks = [(p, t, r) for p in pop_sizes for t in max_iters for r in range(runs)]

    max_workers = 10
    print(f'Starting {len(tasks)} independent runs across {max_workers} CPU cores...')

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initialize_worker,
        initargs=('../data/hosts/barbara.512.tiff', '../data/watermark/yacht.tiff')
    ) as executor:
        results = list(executor.map(evaluate_configuration, tasks))

    print('All runs completed. Generating plots...')

    df = pd.DataFrame(results)

    # ==========================================
    # STATISTICAL AGGREGATION
    # ==========================================
    agg_df = df.groupby(['Population_Size', 'Max_Iterations']).agg(
        Mean_Obj=('Objective_Value', 'mean'),
        Std_Obj=('Objective_Value', 'std'),
        Mean_Time=('Execution_Time', 'mean'),
        Count=('Objective_Value', 'count'),
    ).reset_index()

    agg_df['CI_Obj'] = 1.96 * (agg_df['Std_Obj'] / np.sqrt(agg_df['Count']))

    df.to_csv('jaya_watermarking_results.csv', index=False)
    agg_df.to_csv('jaya_watermarking_aggregated_results.csv', index=False)
    print("Aggregated data saved to 'jaya_watermarking_aggregated_results.csv'")

    # ==========================================
    # GENERATE VISUALIZATIONS
    # ==========================================
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(22, 6))

    # --- Plot A: Pareto Front ---
    ax = axes[0]
    sns.scatterplot(data=agg_df, x='Mean_Time', y='Mean_Obj',
                    hue='Population_Size', style='Max_Iterations',
                    s=150, ax=ax, palette='viridis', alpha=0.8)

    sorted_df = agg_df.sort_values('Mean_Time')
    pareto_front = []
    min_obj = float('inf')
    for index, row in sorted_df.iterrows():
        if row['Mean_Obj'] < min_obj:
            pareto_front.append(row)
            min_obj = row['Mean_Obj']

    pareto_df = pd.DataFrame(pareto_front)
    ax.plot(pareto_df['Mean_Time'], pareto_df['Mean_Obj'], color='red',
            linestyle='--', linewidth=2, label='Pareto Front')

    ax.set_title('A: Time vs Objective (Pareto Front)', fontsize=14)
    ax.set_xlabel('Mean Execution Time (s)', fontsize=12)
    ax.set_ylabel('Mean Objective Value', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # --- Plot B: Line Plot with Error Bars ---
    ax = axes[1]
    for t in max_iters:
        subset = agg_df[agg_df['Max_Iterations'] == t]
        ax.errorbar(subset['Population_Size'], subset['Mean_Obj'], yerr=subset['CI_Obj'],
                    marker='o', linewidth=2, capsize=5, capthick=2, label=f'Iters: {t}')

    # Parameter Interaction & Stability (95% CI)
    ax.set_title('B: Parameter Interaction & Stability', fontsize=14)
    ax.set_xlabel('Population Size', fontsize=12)
    ax.set_ylabel('Mean Objective Value', fontsize=12)
    ax.legend(title='Max Iterations', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Plot C: Heatmap ---
    ax = axes[2]
    pivot_table = agg_df.pivot(index='Max_Iterations', columns='Population_Size', values='Mean_Obj')
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlGnBu_r',
                ax=ax, cbar_kws={'label': 'Mean Objective'})
    ax.set_title('C: Objective Value Landscape', fontsize=14)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('jaya_parameter_sensitivity.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()