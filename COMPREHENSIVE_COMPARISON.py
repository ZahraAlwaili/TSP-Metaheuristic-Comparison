"""
Comprehensive Comparison Script: AOA vs AGWO
Generates all visualizations and comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Import the algorithm classes (adjust import paths if needed)
from AOA_TSP_WITH_VISUALIZATIONS import AOA_TSP, compare_multiple_runs as aoa_compare_runs, plot_comparison_bars
from AGWO_TSP_WITH_VISUALIZATIONS import AGWO_TSP, compare_multiple_runs as agwo_compare_runs, plot_multiple_runs_comparison

plt.style.use('seaborn-v0_8-darkgrid')
sns = __import__('seaborn')
sns.set_palette("husl")


def create_test_instances():
    """Create multiple TSP test instances of different sizes"""
    instances = {}
    
    # 10-city instance
    np.random.seed(42)
    coords_10 = np.random.rand(10, 2) * 100
    instances['10-city'] = np.linalg.norm(coords_10[:, np.newaxis] - coords_10[np.newaxis, :], axis=2)
    
    # 20-city instance
    np.random.seed(43)
    coords_20 = np.random.rand(20, 2) * 100
    instances['20-city'] = np.linalg.norm(coords_20[:, np.newaxis] - coords_20[np.newaxis, :], axis=2)
    
    # 30-city instance
    np.random.seed(44)
    coords_30 = np.random.rand(30, 2) * 100
    instances['30-city'] = np.linalg.norm(coords_30[:, np.newaxis] - coords_30[np.newaxis, :], axis=2)
    
    return instances


def benchmark_algorithms(instances, n_runs=10):
    """Benchmark both algorithms on multiple instances"""
    results = {}
    
    for instance_name, distance_matrix in instances.items():
        print(f"\n{'='*70}")
        print(f"Benchmarking on {instance_name} instance ({distance_matrix.shape[0]} cities)")
        print(f"{'='*70}")
        
        # AOA
        print(f"\nRunning AOA ({n_runs} runs)...")
        aoa_results = aoa_compare_runs(
            distance_matrix, AOA_TSP,
            n_runs=n_runs,
            population_size=30,
            max_iterations=300
        )
        
        # AGWO
        print(f"Running AGWO ({n_runs} runs)...")
        agwo_results = agwo_compare_runs(
            distance_matrix, AGWO_TSP,
            n_runs=n_runs,
            population_size=30,
            max_iterations=300
        )
        
        results[instance_name] = {
            'AOA': aoa_results,
            'AGWO': agwo_results,
            'distance_matrix': distance_matrix
        }
        
        # Print statistics
        print(f"\n{instance_name} Results:")
        print(f"{'Algorithm':<15} {'Best':<12} {'Mean':<12} {'Std':<12} {'Worst':<12}")
        print("-" * 60)
        
        for algo_name, algo_results in [('AOA', aoa_results), ('AGWO', agwo_results)]:
            distances = algo_results['best_distances']
            print(f"{algo_name:<15} {np.min(distances):<12.2f} {np.mean(distances):<12.2f} {np.std(distances):<12.2f} {np.max(distances):<12.2f}")
    
    return results


def generate_all_visualizations(results):
    """Generate all comparison visualizations"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    for instance_name, instance_results in results.items():
        print(f"\nGenerating visualizations for {instance_name}...")
        
        aoa_results = instance_results['AOA']
        agwo_results = instance_results['AGWO']
        
        # 1. Convergence comparison
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = {'AOA': '#1f77b4', 'AGWO': '#ff7f0e'}
        
        for algo_name, algo_results in [('AOA', aoa_results), ('AGWO', agwo_results)]:
            curves = algo_results['convergence_curves']
            avg_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            iterations = range(len(avg_curve))
            
            ax.plot(iterations, avg_curve, color=colors[algo_name], linewidth=3, 
                   label=f'{algo_name} (Mean)', marker='o', markersize=4, markevery=15)
            ax.fill_between(iterations, 
                           np.array(avg_curve) - np.array(std_curve),
                           np.array(avg_curve) + np.array(std_curve),
                           color=colors[algo_name], alpha=0.2)
        
        ax.set_xlabel('Iteration', fontsize=13, weight='bold')
        ax.set_ylabel('Best Distance Found', fontsize=13, weight='bold')
        ax.set_title(f'Convergence Comparison: AOA vs AGWO ({instance_name})', fontsize=15, weight='bold')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.4)
        
        filename = f'comparison_convergence_{instance_name.replace("-", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {filename}")
        plt.close()
        
        # 2. Bar chart comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        aoa_means = np.mean(aoa_results['best_distances'])
        aoa_stds = np.std(aoa_results['best_distances'])
        agwo_means = np.mean(agwo_results['best_distances'])
        agwo_stds = np.std(agwo_results['best_distances'])
        
        algos = ['AOA', 'AGWO']
        means = [aoa_means, agwo_means]
        stds = [aoa_stds, agwo_stds]
        colors_list = ['#1f77b4', '#ff7f0e']
        
        bars = axes[0].bar(algos, means, yerr=stds, capsize=12, color=colors_list, 
                          alpha=0.8, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Best Distance', fontsize=12, weight='bold')
        axes[0].set_title(f'Best Solution Quality ({instance_name})', fontsize=13, weight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + std + 20,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=11, weight='bold')
        
        # Distribution box plot
        aoa_dists = aoa_results['best_distances']
        agwo_dists = agwo_results['best_distances']
        
        bp = axes[1].boxplot([aoa_dists, agwo_dists], labels=algos, patch_artist=True,
                            whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
                            medianprops=dict(color='red', linewidth=2),
                            flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5))
        
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].set_ylabel('Best Distance', fontsize=12, weight='bold')
        axes[1].set_title(f'Solution Distribution ({instance_name})', fontsize=13, weight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f'comparison_quality_{instance_name.replace("-", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {filename}")
        plt.close()
        
        # 3. Fitness over time (best vs average)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, (algo_name, algo_results) in enumerate([('AOA', aoa_results), ('AGWO', agwo_results)]):
            curves = algo_results['convergence_curves']
            avg_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            iterations = range(len(avg_curve))
            
            color = colors[algo_name]
            
            axes[idx].plot(iterations, avg_curve, color=color, linewidth=2.5, label='Best Fitness')
            axes[idx].fill_between(iterations, avg_curve, alpha=0.3, color=color)
            
            axes[idx].set_xlabel('Iteration', fontsize=12, weight='bold')
            axes[idx].set_ylabel('Distance', fontsize=12, weight='bold')
            axes[idx].set_title(f'{algo_name} Convergence ({instance_name})', fontsize=13, weight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(fontsize=11)
        
        plt.tight_layout()
        filename = f'comparison_fitness_{instance_name.replace("-", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {filename}")
        plt.close()
        
        # 4. Multiple runs comparison scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        aoa_runs = list(range(1, len(aoa_results['best_distances']) + 1))
        agwo_runs = list(range(1, len(agwo_results['best_distances']) + 1))
        
        ax.scatter(aoa_runs, aoa_results['best_distances'], s=150, c='#1f77b4', 
                  alpha=0.7, edgecolors='black', linewidth=2, label='AOA', marker='o')
        ax.scatter(agwo_runs, agwo_results['best_distances'], s=150, c='#ff7f0e', 
                  alpha=0.7, edgecolors='black', linewidth=2, label='AGWO', marker='s')
        
        ax.axhline(y=np.mean(aoa_results['best_distances']), color='#1f77b4', 
                  linestyle='--', linewidth=2, label=f"AOA Mean: {np.mean(aoa_results['best_distances']):.1f}")
        ax.axhline(y=np.mean(agwo_results['best_distances']), color='#ff7f0e', 
                  linestyle='--', linewidth=2, label=f"AGWO Mean: {np.mean(agwo_results['best_distances']):.1f}")
        
        ax.set_xlabel('Run Number', fontsize=12, weight='bold')
        ax.set_ylabel('Best Distance', fontsize=12, weight='bold')
        ax.set_title(f'Multiple Runs Results ({instance_name})', fontsize=14, weight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'comparison_runs_{instance_name.replace("-", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {filename}")
        plt.close()


def create_summary_table(results):
    """Create a comprehensive summary table"""
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*100)
    
    summary_data = []
    
    for instance_name, instance_results in results.items():
        for algo_name in ['AOA', 'AGWO']:
            algo_results = instance_results[algo_name]
            distances = algo_results['best_distances']
            
            summary_data.append({
                'Instance': instance_name,
                'Algorithm': algo_name,
                'Best': f"{np.min(distances):.2f}",
                'Mean': f"{np.mean(distances):.2f}",
                'Std': f"{np.std(distances):.2f}",
                'Worst': f"{np.max(distances):.2f}",
                'Improvement': f"{((np.mean(distances) / np.min(distances)) - 1) * 100:.1f}%"
            })
    
    # Print table
    print(f"\n{'Instance':<15} {'Algorithm':<12} {'Best':<12} {'Mean':<12} {'Std':<12} {'Worst':<12} {'Improvement':<15}")
    print("-" * 100)
    
    for row in summary_data:
        print(f"{row['Instance']:<15} {row['Algorithm']:<12} {row['Best']:<12} {row['Mean']:<12} {row['Std']:<12} {row['Worst']:<12} {row['Improvement']:<15}")
    
    return summary_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPREHENSIVE AOA vs AGWO COMPARISON")
    print("="*70)
    
    # Create test instances
    print("\nCreating test instances...")
    instances = create_test_instances()
    print(f"✓ Created {len(instances)} test instances")
    
    # Run benchmarks
    results = benchmark_algorithms(instances, n_runs=10)
    
    # Generate visualizations
    generate_all_visualizations(results)
    
    # Create summary
    summary = create_summary_table(results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Visualizations:")
    print("  ✓ Convergence curves (AOA vs AGWO)")
    print("  ✓ Solution quality comparisons (bars + box plots)")
    print("  ✓ Fitness over time analysis")
    print("  ✓ Multiple runs distribution")
    print("\nAll plots have been saved as PNG files!")
    print("="*70 + "\n")
