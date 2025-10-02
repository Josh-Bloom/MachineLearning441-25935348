"""
Statistical evaluation and visualisation functions for incremental learning experiments.

This module provides functions for statistical analysis and visualisation of
experimental results, including significance testing and result plotting.
"""

from typing import List
from scipy import stats
import numpy as np

def wilcoxon_test(f1_scores_a: List[float], f1_scores_b: List[float], alpha: float = 0.05) -> tuple:
    """
    Perform Wilcoxon signed-rank test between two groups of results.
    """
    # Perform Wilcoxon signed-rank test
    stat, p = stats.wilcoxon(f1_scores_a, f1_scores_b, alternative='two-sided')
    
    # Calculate effect size (rank-biserial correlation)
    diff = np.array(f1_scores_a) - np.array(f1_scores_b)
    n = len(diff[diff != 0])
    if n > 0:
        r_rb = (np.sum(diff > 0) - np.sum(diff < 0)) / n  
        print(f"Effect size (Rank-biserial correlation): {r_rb:.5f}")
    
    # Calculate means for comparison
    f1_scores_a_mean = np.mean(f1_scores_a)
    f1_scores_b_mean = np.mean(f1_scores_b)
    
    # Determine winner based on significance and mean difference
    if p < alpha/2.0 and f1_scores_a_mean > f1_scores_b_mean:
        return 'A', p
    elif p < alpha/2.0 and f1_scores_a_mean < f1_scores_b_mean:
        return 'B', p
    else:
        return 'Tie', p
    
def visualise(overall_accuracy_base: List[float], overall_accuracy_icl: List[float], 
              overall_neuron_base: List[int], overall_neuron_icl: List[int], 
              DATASETS: List[str], SEEDS: List[int], n_folds: int) -> None:
    """
    Create visualisation plots comparing baseline vs incremental learning results.
    """
    from data import expand_name
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Expand dataset names to full names for better readability
    expanded_datasets = [expand_name(dataset) for dataset in DATASETS]

    def average_groups(lst: List[float], group_size: int = 4) -> List[float]:
        """
        Average values in groups of specified size.
        """
        return [
            sum(lst[i:i+group_size]) / group_size
            for i in range(0, len(lst), group_size)
            if len(lst[i:i+group_size]) == group_size
        ]
    
    def interleave(a: List[float], b: List[float]) -> List[float]:
        """
        Interleave two lists element by element.
        """
        return [x for pair in zip(a, b) for x in pair]
    
    # Create DataFrame for visualisation
    df = pd.DataFrame({
        "dataset": expanded_datasets * len(SEEDS) * len(["Base", "ICL"]),
        "model": ["Base", "ICL"] * len(SEEDS) * len(expanded_datasets),
        "seed": SEEDS * len(expanded_datasets) * len(["Base", "ICL"]),
        "accuracy": interleave(average_groups(overall_accuracy_base, group_size=n_folds),
                               average_groups(overall_accuracy_icl, group_size=n_folds)),
        "neurons": interleave(average_groups(overall_neuron_base, group_size=n_folds),
                              average_groups(overall_neuron_icl, group_size=n_folds)),
    })

    # Create accuracy comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot individual seed results as points
    sns.stripplot(
        x='dataset', y='accuracy', hue='model', data=df,
        jitter=True, dodge=True, ax=ax, alpha=0.5, size=6
    )
    # Plot overall means as diamond markers
    sns.pointplot(
        x='dataset', y='accuracy', hue='model', data=df,
        dodge=0.3, markers='D', errorbar=None,
        ax=ax, palette='dark',
        linestyle='none', markersize=8
    )
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Dataset')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('1accuracy.pdf', bbox_inches='tight', dpi=200)

    # Create hidden neurons comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot individual seed results as points
    sns.stripplot(
        x='dataset', y='neurons', hue='model', data=df,
        jitter=True, dodge=True, ax=ax, alpha=0.5, size=6
    )
    # Plot overall means as diamond markers
    sns.pointplot(
        x='dataset', y='neurons', hue='model', data=df,
        dodge=0.3, markers='D', errorbar=None,
        ax=ax, palette='dark',
        linestyle='none', markersize=8
    )
    ax.set_ylabel('Number of Neurons')
    ax.set_xlabel('Dataset')
    ax.get_legend().remove()  # Remove legend for second plot

    plt.tight_layout()
    plt.savefig('1neurons.pdf', bbox_inches='tight', dpi=200)