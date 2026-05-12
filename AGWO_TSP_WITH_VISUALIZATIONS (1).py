"""
Adaptive Grey Wolf Optimization (AGWO) for Traveling Salesman Problem
With comprehensive visualization capabilities
Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class AGWO_TSP:
    """
    Adaptive Grey Wolf Optimization for TSP with visualization
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Square symmetric distance matrix
    population_size : int
        Number of wolves (default: 30)
    max_iterations : int
        Maximum iterations (default: 500)
    gamma : float
        Adaptivity exponent (default: 2.5)
    diversity_threshold : float
        Trigger recovery when diversity < this (default: 0.1)
    epsilon : float
        Small constant for numerical stability (default: 1e-10)
    random_seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, distance_matrix, population_size=30, max_iterations=500,
                 gamma=2.5, diversity_threshold=0.1, epsilon=1e-10, random_seed=None):
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.shape[0]
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.diversity_threshold = diversity_threshold
        self.epsilon = epsilon
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _lov_permutation(self, x):
        """
        Largest Order Value permutation mapping
        Converts continuous values to valid city permutation
        """
        return np.argsort(x % 1.0)
    
    def _calculate_tour_distance(self, tour):
        """Calculate total distance of a tour"""
        distance = 0
        for i in range(len(tour)):
            distance += self.distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return distance
    
    def _calculate_diversity(self, population, fitness):
        """Calculate population diversity"""
        if len(fitness) == 0:
            return 0
        max_fitness = np.max(fitness)
        min_fitness = np.min(fitness)
        if max_fitness == min_fitness:
            return 0
        diversity = np.std(fitness) / (max_fitness - min_fitness + self.epsilon)
        return diversity
    
    def _calculate_adaptive_a(self, t):
        """Calculate adaptive parameter a"""
        return 2 - 2 * ((t / self.max_iterations) ** self.gamma)
    
    def _hamming_distance(self, tour1, tour2):
        """Calculate Hamming distance between two tours"""
        return np.sum(tour1 != tour2)
    
    def _apply_perturbation(self, tour):
        """Apply perturbation to escape local optima"""
        perturbed = tour.copy()
        # Random swap
        idx1, idx2 = np.random.choice(len(tour), 2, replace=False)
        perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]
        return perturbed
    
    def solve(self, verbose=True):
        """
        Main AGWO solving method
        
        Returns:
        --------
        best_tour : np.ndarray
            Best tour found
        best_distance : float
            Distance of best tour
        history : dict
            Convergence and statistics history
        """
        # Initialize population with random continuous values
        population = np.random.rand(self.population_size, self.n_cities)
        
        # Evaluate initial population
        fitness = np.array([self._calculate_tour_distance(self._lov_permutation(ind)) 
                          for ind in population])
        
        # Find best three (Alpha, Beta, Delta)
        sorted_indices = np.argsort(fitness)
        alpha_idx, beta_idx, delta_idx = sorted_indices[:3]
        
        alpha_solution = population[alpha_idx].copy()
        beta_solution = population[beta_idx].copy()
        delta_solution = population[delta_idx].copy()
        
        best_distance = fitness[alpha_idx]
        
        # History tracking
        convergence_curve = [best_distance]
        best_distances_per_iteration = [best_distance]
        average_fitness_per_iteration = [np.mean(fitness)]
        diversity_per_iteration = []
        adaptive_a_history = []
        
        # Main loop
        for t in range(self.max_iterations):
            a = self._calculate_adaptive_a(t)
            adaptive_a_history.append(a)
            
            # Calculate diversity
            diversity = self._calculate_diversity(population, fitness)
            diversity_per_iteration.append(diversity)
            
            # Apply perturbation if diversity is low
            if diversity < self.diversity_threshold:
                worst_indices = np.argsort(fitness)[-3:]  # Worst 3
                for idx in worst_indices:
                    population[idx] = self._lov_permutation(self._apply_perturbation(
                        self._lov_permutation(population[idx])))
                    fitness[idx] = self._calculate_tour_distance(self._lov_permutation(population[idx]))
            
            # Update each wolf
            for i in range(self.population_size):
                # Update using Alpha
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha_solution - population[i])
                X_alpha = alpha_solution - A1 * D_alpha
                
                # Update using Beta
                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta_solution - population[i])
                X_beta = beta_solution - A2 * D_beta
                
                # Update using Delta
                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta_solution - population[i])
                X_delta = delta_solution - A3 * D_delta
                
                # Average the three updates
                population[i] = (X_alpha + X_beta + X_delta) / 3.0
                
                # Evaluate new solution
                tour = self._lov_permutation(population[i])
                distance = self._calculate_tour_distance(tour)
                
                if distance < fitness[i]:
                    fitness[i] = distance
                    
                    # Update hierarchy if better than current best
                    if distance < best_distance:
                        delta_solution = beta_solution.copy()
                        beta_solution = alpha_solution.copy()
                        alpha_solution = population[i].copy()
                        best_distance = distance
            
            # Track history
            best_distances_per_iteration.append(best_distance)
            average_fitness_per_iteration.append(np.mean(fitness))
            convergence_curve.append(best_distance)
            
            if verbose and (t + 1) % 100 == 0:
                print(f"Iteration {t + 1}/{self.max_iterations}: Best Distance = {best_distance:.2f}, Diversity = {diversity:.4f}")
        
        best_tour = self._lov_permutation(alpha_solution)
        
        history = {
            'convergence_curve': convergence_curve,
            'best_distances': best_distances_per_iteration,
            'average_fitness': average_fitness_per_iteration,
            'diversity': diversity_per_iteration,
            'adaptive_a': adaptive_a_history,
            'best_tour': best_tour,
            'best_distance': best_distance,
            'population_size': self.population_size,
            'iterations': self.max_iterations
        }
        
        return best_tour, best_distance, history
    
    def visualize_solution(self, tour, title="AGWO TSP Solution"):
        """
        Visualize the TSP tour
        
        Parameters:
        -----------
        tour : array-like
            City sequence to visualize
        title : str
            Plot title
        """
        n = len(tour)
        
        # Generate random coordinates for visualization
        np.random.seed(42)
        coords = np.random.rand(n, 2) * 100
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot cities
        ax.scatter(coords[:, 0], coords[:, 1], s=200, c='darkgreen', zorder=3, edgecolors='black', linewidth=2)
        
        # Add city labels
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), fontsize=10, ha='center', va='center', color='white', weight='bold')
        
        # Plot tour
        for i in range(len(tour)):
            start = coords[tour[i]]
            end = coords[tour[(i + 1) % len(tour)]]
            ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                    head_width=1.5, head_length=1, fc='darkgreen', ec='darkgreen', alpha=0.6, length_includes_head=True)
        
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect('equal')
        ax.set_title(f"{title}", fontsize=14, weight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_convergence(self, history, title="AGWO Convergence Curve"):
        """Plot convergence curve"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = range(len(history['convergence_curve']))
        ax.plot(iterations, history['convergence_curve'], 'g-', linewidth=2.5, label='Best Distance')
        ax.fill_between(iterations, history['convergence_curve'], alpha=0.3, color='green')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Tour Distance', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_fitness_analysis(self, history, title="AGWO Fitness Analysis"):
        """Plot best and average fitness over iterations"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = range(len(history['best_distances']))
        ax.plot(iterations, history['best_distances'], 'g-', linewidth=2.5, label='Best Fitness')
        ax.plot(iterations, history['average_fitness'], 'orange', linestyle='--', linewidth=2, label='Average Fitness')
        ax.fill_between(iterations, history['best_distances'], alpha=0.2, color='green')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Fitness (Distance)', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_diversity(self, history, title="AGWO Population Diversity"):
        """Plot population diversity over iterations"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = range(len(history['diversity']))
        ax.plot(iterations, history['diversity'], 'purple', linewidth=2.5)
        ax.fill_between(iterations, history['diversity'], alpha=0.3, color='purple')
        
        ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Diversity Threshold')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Population Diversity', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_adaptive_parameter(self, history, title="AGWO Adaptive Parameter 'a'"):
        """Plot adaptive parameter evolution"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = range(len(history['adaptive_a']))
        ax.plot(iterations, history['adaptive_a'], 'brown', linewidth=2.5)
        ax.fill_between(iterations, history['adaptive_a'], alpha=0.3, color='brown')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Parameter a', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax


def compare_multiple_runs(distance_matrix, algorithm_class, n_runs=10, **kwargs):
    """Run algorithm multiple times and compare results"""
    results = {
        'best_distances': [],
        'avg_distances': [],
        'convergence_curves': [],
        'final_solutions': [],
        'diversity_curves': []
    }
    
    for run in range(n_runs):
        algo = algorithm_class(distance_matrix, random_seed=run, **kwargs)
        tour, distance, history = algo.solve(verbose=False)
        
        results['best_distances'].append(distance)
        results['avg_distances'].append(np.mean(history['best_distances']))
        results['convergence_curves'].append(history['convergence_curve'])
        results['final_solutions'].append(tour)
        if 'diversity' in history:
            results['diversity_curves'].append(history['diversity'])
    
    return results


def plot_multiple_runs_comparison(results_dict, algorithm_names, title="Multiple Runs Comparison"):
    """Plot comparison of multiple algorithms across multiple runs"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithm_names)]
    
    # Plot 1: Best distances box plot
    best_distances_list = [results_dict[name]['best_distances'] for name in algorithm_names]
    axes[0, 0].boxplot(best_distances_list, labels=algorithm_names, patch_artist=True)
    for patch, color in zip(axes[0, 0].artists, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 0].set_ylabel('Best Distance', fontsize=11)
    axes[0, 0].set_title('Best Solution Quality Distribution', fontsize=12, weight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average convergence
    for algo_name, color in zip(algorithm_names, colors):
        curves = results_dict[algo_name]['convergence_curves']
        avg_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        iterations = range(len(avg_curve))
        axes[0, 1].plot(iterations, avg_curve, color=color, linewidth=2, label=algo_name, marker='o', markersize=2, markevery=20)
        axes[0, 1].fill_between(iterations, 
                               np.array(avg_curve) - np.array(std_curve),
                               np.array(avg_curve) + np.array(std_curve),
                               color=color, alpha=0.15)
    axes[0, 1].set_xlabel('Iteration', fontsize=11)
    axes[0, 1].set_ylabel('Best Distance', fontsize=11)
    axes[0, 1].set_title('Convergence Curves (Mean ± Std)', fontsize=12, weight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Bar chart of means
    best_means = [np.mean(results_dict[name]['best_distances']) for name in algorithm_names]
    best_stds = [np.std(results_dict[name]['best_distances']) for name in algorithm_names]
    
    bars = axes[1, 0].bar(algorithm_names, best_means, yerr=best_stds, capsize=10, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 0].set_ylabel('Best Distance', fontsize=11)
    axes[1, 0].set_title('Mean Best Distance ± Std', fontsize=12, weight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, best_means, best_stds):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + std + 30,
                       f'{mean:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Plot 4: Success rate statistics
    stats_text = "Algorithm Statistics (10 runs)\n\n"
    for algo_name in algorithm_names:
        distances = results_dict[algo_name]['best_distances']
        stats_text += f"{algo_name}:\n"
        stats_text += f"  Mean: {np.mean(distances):.2f}\n"
        stats_text += f"  Std:  {np.std(distances):.2f}\n"
        stats_text += f"  Best: {np.min(distances):.2f}\n"
        stats_text += f"  Worst: {np.max(distances):.2f}\n\n"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.suptitle(title, fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    return fig, axes


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create a simple distance matrix (10-city TSP)
    np.random.seed(42)
    n_cities = 10
    coords = np.random.rand(n_cities, 2) * 100
    distance_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
    
    print("=" * 70)
    print("ADAPTIVE GREY WOLF OPTIMIZATION FOR TSP - WITH VISUALIZATIONS")
    print("=" * 70)
    
    # Single run
    print("\n1. Running AGWO (single run)...")
    agwo = AGWO_TSP(distance_matrix, population_size=30, max_iterations=200)
    best_tour, best_distance, history = agwo.solve(verbose=True)
    
    print(f"\nBest tour found: {best_tour}")
    print(f"Best distance: {best_distance:.2f}")
    
    # Visualizations
    print("\n2. Creating visualizations...")
    
    # Plot 1: Tour visualization
    fig1, ax1 = agwo.visualize_solution(best_tour, title="AGWO TSP Solution Route")
    plt.savefig('agwo_route_visualization.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: agwo_route_visualization.png")
    
    # Plot 2: Convergence curve
    fig2, ax2 = agwo.plot_convergence(history, title="AGWO Convergence Curve")
    plt.savefig('agwo_convergence_curve.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: agwo_convergence_curve.png")
    
    # Plot 3: Fitness analysis
    fig3, ax3 = agwo.plot_fitness_analysis(history, title="AGWO Fitness Analysis")
    plt.savefig('agwo_fitness_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: agwo_fitness_analysis.png")
    
    # Plot 4: Diversity
    fig4, ax4 = agwo.plot_diversity(history, title="AGWO Population Diversity")
    plt.savefig('agwo_diversity.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: agwo_diversity.png")
    
    # Plot 5: Adaptive parameter
    fig5, ax5 = agwo.plot_adaptive_parameter(history, title="AGWO Adaptive Parameter 'a'")
    plt.savefig('agwo_adaptive_parameter.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: agwo_adaptive_parameter.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("AGWO implementation complete with full visualization support!")
    print("=" * 70)
