"""
Visualize all experimental results from CSV summaries.

Generates publication-ready figures for the report:
- Accuracy comparison (bar charts + boxplots)
- Per-class F1 scores
- Confusion matrix heatmaps
- Transition matrix heatmaps
- Statistical comparisons

All figures saved to figures/ directory.
"""

import os
import csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_csv_as_dicts(path: str) -> List[Dict[str, str]]:
    """Load CSV file as list of dictionaries."""
    with open(path, 'r', newline='') as f:
        return list(csv.DictReader(f))


def safe_float(x: str) -> float:
    """Convert string to float, handling 'nan'."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return float('nan')


def plot_accuracy_comparison():
    """Bar chart comparing mean accuracy across all experiments."""
    print("[VIZ] Generating accuracy comparison bar chart...")
    data = load_csv_as_dicts(os.path.join(RESULTS_DIR, "metrics_summary.csv"))
    
    # Group by dataset
    datasets = {}
    for row in data:
        ds = row['dataset']
        if ds not in datasets:
            datasets[ds] = {'models': [], 'means': [], 'stds': []}
        datasets[ds]['models'].append(row['model'])
        datasets[ds]['means'].append(safe_float(row['mean_acc']) * 100)
        datasets[ds]['stds'].append(safe_float(row['std_acc']) * 100)
    
    # Create subplots
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 5))
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, (ds_name, ds_data) in enumerate(sorted(datasets.items())):
        ax = axes[idx]
        x_pos = np.arange(len(ds_data['models']))
        colors = ['#3498db', '#e74c3c']  # Blue for CNN, Red for CNNWithNAL
        
        bars = ax.bar(x_pos, ds_data['means'], yerr=ds_data['stds'], 
                      capsize=5, color=colors[:len(ds_data['models'])],
                      edgecolor='black', linewidth=1.2, alpha=0.85)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{ds_name}', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ds_data['models'], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, ds_data['means'], ds_data['stds']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                   f'{mean:.1f}±{std:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/accuracy_comparison.png")


def plot_accuracy_boxplots():
    """Boxplot showing accuracy distribution across runs."""
    print("[VIZ] Generating accuracy distribution boxplots...")
    data = load_csv_as_dicts(os.path.join(RESULTS_DIR, "metrics_per_run.csv"))
    
    # Group by dataset and model
    datasets = {}
    for row in data:
        ds = row['dataset']
        model = row['model']
        acc = safe_float(row['accuracy']) * 100
        
        if ds not in datasets:
            datasets[ds] = {}
        if model not in datasets[ds]:
            datasets[ds][model] = []
        datasets[ds][model].append(acc)
    
    # Create subplots
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 5))
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, (ds_name, models_data) in enumerate(sorted(datasets.items())):
        ax = axes[idx]
        
        # Prepare data for boxplot
        box_data = []
        labels = []
        for model in sorted(models_data.keys()):
            box_data.append(models_data[model])
            labels.append(model)
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='green', linewidth=2, linestyle='--'),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{ds_name} - Distribution Across 10 Runs', fontsize=13, fontweight='bold')
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Mean')
    ]
    axes[-1].legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/accuracy_boxplots.png")


def plot_per_class_metrics():
    """Bar charts showing per-class F1 scores."""
    print("[VIZ] Generating per-class metrics...")
    data = load_csv_as_dicts(os.path.join(RESULTS_DIR, "per_class_metrics_summary.csv"))
    
    # Group by dataset
    datasets = {}
    for row in data:
        ds = row['dataset']
        if ds not in datasets:
            datasets[ds] = {}
        
        model = row['model']
        if model not in datasets[ds]:
            datasets[ds][model] = {'classes': [], 'f1_means': [], 'f1_stds': []}
        
        datasets[ds][model]['classes'].append(int(row['class']))
        datasets[ds][model]['f1_means'].append(safe_float(row['f1_mean']) * 100)
        datasets[ds][model]['f1_stds'].append(safe_float(row['f1_std']) * 100)
    
    # Create subplots
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 5))
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, (ds_name, models_data) in enumerate(sorted(datasets.items())):
        ax = axes[idx]
        
        num_classes = 3
        num_models = len(models_data)
        x = np.arange(num_classes)
        width = 0.35
        
        colors = ['#3498db', '#e74c3c']
        for i, (model, metrics) in enumerate(sorted(models_data.items())):
            offset = width * (i - num_models/2 + 0.5)
            ax.bar(x + offset, metrics['f1_means'], width, 
                   yerr=metrics['f1_stds'], capsize=3,
                   label=model, color=colors[i], alpha=0.85,
                   edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{ds_name} - Per-Class F1', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
        ax.legend(loc='best', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'per_class_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/per_class_f1.png")


def plot_confusion_matrices():
    """Heatmaps of mean confusion matrices."""
    print("[VIZ] Generating confusion matrix heatmaps...")
    data = load_csv_as_dicts(os.path.join(RESULTS_DIR, "confusion_matrices_mean.csv"))
    
    # Group by experiment
    experiments = {}
    for row in data:
        exp = row['experiment']
        if exp not in experiments:
            experiments[exp] = {'i': [], 'j': [], 'value': [], 'dataset': row['dataset'], 'model': row['model']}
        experiments[exp]['i'].append(int(row['i']))
        experiments[exp]['j'].append(int(row['j']))
        experiments[exp]['value'].append(safe_float(row['value']))
    
    # Determine grid size
    num_exp = len(experiments)
    ncols = 3
    nrows = (num_exp + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (exp_name, exp_data) in enumerate(sorted(experiments.items())):
        ax = axes[idx]
        
        # Reconstruct confusion matrix
        num_classes = max(exp_data['i']) + 1
        cm = np.zeros((num_classes, num_classes))
        for i, j, val in zip(exp_data['i'], exp_data['j'], exp_data['value']):
            cm[i, j] = val
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', 
                   cbar_kws={'label': 'Proportion'},
                   ax=ax, vmin=0, vmax=1, square=True,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title(f'{exp_data["model"]} - {exp_data["dataset"]}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_exp, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/confusion_matrices.png")


def plot_transition_matrices():
    """Heatmaps of estimated transition matrices (T_hat)."""
    print("[VIZ] Generating transition matrix heatmaps...")
    csv_path = os.path.join(RESULTS_DIR, "transition_matrices_summary.csv")
    
    if not os.path.exists(csv_path):
        print("[INFO] No transition matrices found, skipping.")
        return
    
    data = load_csv_as_dicts(csv_path)
    if not data:
        print("[INFO] Empty transition matrices file, skipping.")
        return
    
    # Group by experiment
    experiments = {}
    for row in data:
        exp = row['experiment']
        if exp not in experiments:
            experiments[exp] = {'i': [], 'j': [], 'mean': [], 'std': [], 
                              'dataset': row['dataset'], 'model': row['model']}
        experiments[exp]['i'].append(int(row['i']))
        experiments[exp]['j'].append(int(row['j']))
        experiments[exp]['mean'].append(safe_float(row['mean']))
        experiments[exp]['std'].append(safe_float(row['std']))
    
    fig, axes = plt.subplots(1, len(experiments), figsize=(7 * len(experiments), 5))
    if len(experiments) == 1:
        axes = [axes]
    
    for idx, (exp_name, exp_data) in enumerate(sorted(experiments.items())):
        ax = axes[idx]
        
        # Reconstruct matrix
        num_classes = max(exp_data['i']) + 1
        T = np.zeros((num_classes, num_classes))
        T_std = np.zeros((num_classes, num_classes))
        for i, j, m, s in zip(exp_data['i'], exp_data['j'], exp_data['mean'], exp_data['std']):
            T[i, j] = m
            T_std[i, j] = s
        
        # Plot heatmap with std annotations
        annot_text = np.empty_like(T, dtype=object)
        for i in range(num_classes):
            for j in range(num_classes):
                annot_text[i, j] = f'{T[i,j]:.3f}\n±{T_std[i,j]:.3f}'
        
        sns.heatmap(T, annot=annot_text, fmt='', cmap='RdYlGn', 
                   cbar_kws={'label': 'Transition Probability'},
                   ax=ax, vmin=0, vmax=1, square=True,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title(f'T_hat: {exp_data["model"]} - {exp_data["dataset"]}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Observed Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'transition_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/transition_matrices.png")


def plot_statistical_comparison():
    """Statistical comparison with significance tests."""
    print("[VIZ] Generating statistical comparison...")
    data = load_csv_as_dicts(os.path.join(RESULTS_DIR, "metrics_per_run.csv"))
    
    # Group by dataset, then by model
    datasets = {}
    for row in data:
        ds = row['dataset']
        model = row['model']
        acc = safe_float(row['accuracy']) * 100
        
        if np.isnan(acc):
            continue
        
        if ds not in datasets:
            datasets[ds] = {}
        if model not in datasets[ds]:
            datasets[ds][model] = []
        datasets[ds][model].append(acc)
    
    # Perform paired t-tests
    results = []
    for ds_name in sorted(datasets.keys()):
        models = sorted(datasets[ds_name].keys())
        if len(models) == 2:
            model1, model2 = models
            data1 = datasets[ds_name][model1]
            data2 = datasets[ds_name][model2]
            
            # Ensure same length for paired test
            min_len = min(len(data1), len(data2))
            data1 = data1[:min_len]
            data2 = data2[:min_len]
            
            if len(data1) > 1:
                t_stat, p_value = stats.ttest_rel(data1, data2)
                mean_diff = np.mean(data1) - np.mean(data2)
                
                # Cohen's d effect size
                std_pooled = np.sqrt((np.var(data1) + np.var(data2)) / 2)
                cohens_d = mean_diff / std_pooled if std_pooled > 0 else 0
                
                results.append({
                    'dataset': ds_name,
                    'model1': model1,
                    'model2': model2,
                    'mean_diff': mean_diff,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                })
    
    if not results:
        print("[INFO] Not enough data for statistical tests.")
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean difference with significance
    datasets_names = [r['dataset'] for r in results]
    mean_diffs = [r['mean_diff'] for r in results]
    colors = ['green' if r['significant'] else 'gray' for r in results]
    
    bars = ax1.barh(range(len(results)), mean_diffs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(results)))
    ax1.set_yticklabels(datasets_names)
    ax1.set_xlabel('Mean Accuracy Difference (%)\n(CNN - CNNWithNAL)', fontsize=11, fontweight='bold')
    ax1.set_title('Paired T-Test Results', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add p-value annotations
    for i, (bar, r) in enumerate(zip(bars, results)):
        width = bar.get_width()
        label = f"p={r['p_value']:.4f}"
        if r['significant']:
            label += " *"
        ax1.text(width + 0.5 if width > 0 else width - 0.5, bar.get_y() + bar.get_height()/2,
                label, ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    # Plot 2: Effect sizes (Cohen's d)
    cohens_ds = [r['cohens_d'] for r in results]
    bars2 = ax2.barh(range(len(results)), cohens_ds, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(results)))
    ax2.set_yticklabels(datasets_names)
    ax2.set_xlabel("Cohen's d (Effect Size)", fontsize=11, fontweight='bold')
    ax2.set_title('Effect Size Analysis', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Add effect size interpretation lines
    ax2.axvline(x=0.2, color='orange', linestyle=':', alpha=0.5, label='Small (0.2)')
    ax2.axvline(x=0.5, color='red', linestyle=':', alpha=0.5, label='Medium (0.5)')
    ax2.axvline(x=0.8, color='darkred', linestyle=':', alpha=0.5, label='Large (0.8)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'statistical_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/statistical_comparison.png")
    
    # Print summary
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON SUMMARY")
    print("="*70)
    for r in results:
        sig_marker = "***" if r['significant'] else "n.s."
        print(f"\n{r['dataset']}:")
        print(f"  {r['model1']} vs {r['model2']}")
        print(f"  Mean Difference: {r['mean_diff']:+.2f}%")
        print(f"  p-value: {r['p_value']:.4f} {sig_marker}")
        print(f"  Cohen's d: {r['cohens_d']:.3f}")
        if abs(r['cohens_d']) < 0.2:
            effect = "negligible"
        elif abs(r['cohens_d']) < 0.5:
            effect = "small"
        elif abs(r['cohens_d']) < 0.8:
            effect = "medium"
        else:
            effect = "large"
        print(f"  Effect size: {effect}")
    print("="*70 + "\n")


def plot_f1_vs_accuracy():
    """Scatter plot showing F1 vs Accuracy correlation."""
    print("[VIZ] Generating F1 vs Accuracy scatter plot...")
    data = load_csv_as_dicts(os.path.join(RESULTS_DIR, "metrics_summary.csv"))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'cnn': '#3498db', 'cnnwithnal': '#e74c3c'}
    markers = {'FashionMNIST0.3': 'o', 'FashionMNIST0.6': 's', 'CIFAR': '^'}
    
    for row in data:
        acc = safe_float(row['mean_acc']) * 100
        f1 = safe_float(row.get('mean_macro_f1', 'nan')) * 100
        if np.isnan(f1):
            continue
        
        model = row['model']
        dataset = row['dataset']
        
        ax.scatter(acc, f1, s=200, alpha=0.7,
                  color=colors.get(model, 'gray'),
                  marker=markers.get(dataset, 'o'),
                  edgecolors='black', linewidths=1.5,
                  label=f"{model} - {dataset}")
    
    # Add diagonal reference line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, label='Perfect correlation')
    
    ax.set_xlabel('Mean Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Macro F1 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs F1 Score Correlation', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'f1_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/f1_vs_accuracy.png")


def main():
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("VISUALIZING EXPERIMENTAL RESULTS")
    print("="*70 + "\n")
    
    # Check if CSV files exist
    required_files = [
        "metrics_summary.csv",
        "metrics_per_run.csv",
        "per_class_metrics_summary.csv",
        "confusion_matrices_mean.csv"
    ]
    
    missing = []
    for fname in required_files:
        fpath = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(fpath):
            missing.append(fname)
    
    if missing:
        print(f"[ERROR] Missing required CSV files: {', '.join(missing)}")
        print(f"[ERROR] Please run 'python generate_results_summary.py' first!")
        return
    
    # Generate all plots
    try:
        plot_accuracy_comparison()
        plot_accuracy_boxplots()
        plot_per_class_metrics()
        plot_confusion_matrices()
        plot_transition_matrices()
        plot_statistical_comparison()
        plot_f1_vs_accuracy()
        
        print("\n" + "="*70)
        print(f"SUCCESS! All visualizations saved to: {FIGURES_DIR}")
        print("="*70)
        print("\nGenerated figures:")
        print("  1. accuracy_comparison.png    - Mean accuracy bar charts")
        print("  2. accuracy_boxplots.png       - Accuracy distribution across runs")
        print("  3. per_class_f1.png            - Per-class F1 scores")
        print("  4. confusion_matrices.png      - Confusion matrix heatmaps")
        print("  5. transition_matrices.png     - Estimated T_hat heatmaps")
        print("  6. statistical_comparison.png  - T-test and effect sizes")
        print("  7. f1_vs_accuracy.png          - Correlation scatter plot")
        print("\nThese are publication-ready figures for your report!\n")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
