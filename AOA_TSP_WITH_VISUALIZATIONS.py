"""
Arithmetic Optimization Algorithm (AOA) for Traveling Salesman Problem
With comprehensive visualization capabilities
Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class AOA_TSP:
    """
    Arithmetic Optimization Algorithm for TSP with visualization
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Square symmetric distance matrix
    population_size : int
        Number of solutions (default: 30)
    max_iterations : int
        Maximum iterations (default: 500)
    alpha : float
        Adaptivity parameter (default: 5.0)
    epsilon : float
        Small constant for numerical stability (default: 1e-10)
    random_seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, distance_matrix, population_size=30, max_iterations=500, 
                 alpha=5.0, epsilon=1e-10, random_seed=None):
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.shape[0]
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.epsilon = epsilon
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _lov_permutation(self, x):
        """
        Largest Order Value permutation mapping
        Converts continuous values [0,1) to valid city permutation
        """
        return np.argsort(x % 1.0)
    
    def _calculate_tour_distance(self, tour):
        """Calculate total distance of a tour"""
        distance = 0
        for i in range(len(tour)):
            distance += self.distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return distance
    
    def _calculate_mop(self, t):
        """Math Optimizer Probability"""
        return min(t / self.max_iterations, 1)
    
    def _calculate_mor(self, t):
        """Math Optimizer Ratio"""
        return 1 - ((t ** (1 / self.alpha)) / (self.max_iterations ** (1 / self.alpha)))
    
    def _exploration_division(self, x_best, x_current):
        """Division operator for exploration"""
        return x_best / (self.epsilon + np.abs(np.random.rand(self.n_cities) * x_best - x_current))
    
    def _exploration_multiplication(self, x_best):
        """Multiplication operator for exploration"""
        return x_best * np.random.rand(self.n_cities)
    
    def _exploitation_subtraction(self, x_best, x_current):
        """Subtraction operator for exploitation"""
        return x_best - np.random.rand(self.n_cities) * x_current
    
    def _exploitation_addition(self, x_best, x_current):
        """Addition operator for exploitation"""
        return x_best + np.random.rand(self.n_cities) * x_current
    
    def solve(self, verbose=True):
        """
        Main AOA solving method
        
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
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_distance = fitness[best_idx]
        
        # History tracking
        convergence_curve = [best_distance]
        best_distances_per_iteration = [best_distance]
        average_fitness_per_iteration = [np.mean(fitness)]
        diversity_per_iteration = []
        
        # Main loop
        for t in range(self.max_iterations):
            mop = self._calculate_mop(t)
            mor = self._calculate_mor(t)
            
            for i in range(self.population_size):
                if np.random.rand() > mop:  # Exploration phase
                    if np.random.rand() > mor:
                        # Division
                        population[i] = self._exploration_division(best_solution, population[i])
                    else:
                        # Multiplication
                        population[i] = self._exploration_multiplication(best_solution)
                else:  # Exploitation phase
                    if np.random.rand() > mor:
                        # Subtraction
                        population[i] = self._exploitation_subtraction(best_solution, population[i])
                    else:
                        # Addition
                        population[i] = self._exploitation_addition(best_solution, population[i])
                
                # Evaluate new solution
                tour = self._lov_permutation(population[i])
                distance = self._calculate_tour_distance(tour)
                
                if distance < fitness[i]:
                    fitness[i] = distance
                    
                    # Update best solution
                    if distance < best_distance:
                        best_distance = distance
                        best_solution = population[i].copy()
            
            # Track history
            best_distances_per_iteration.append(best_distance)
            average_fitness_per_iteration.append(np.mean(fitness))
            convergence_curve.append(best_distance)
            
            # Calculate diversity
            best_tour = self._lov_permutation(best_solution)
            diversity = np.std([self._calculate_tour_distance(self._lov_permutation(ind)) 
                              for ind in population])
            diversity_per_iteration.append(diversity)
            
            if verbose and (t + 1) % 100 == 0:
                print(f"Iteration {t + 1}/{self.max_iterations}: Best Distance = {best_distance:.2f}")
        
        best_tour = self._lov_permutation(best_solution)
        
        history = {
            'convergence_curve': convergence_curve,
            'best_distances': best_distances_per_iteration,
            'average_fitness': average_fitness_per_iteration,
            'diversity': diversity_per_iteration,
            'best_tour': best_tour,
            'best_distance': best_distance,
            'population_size': self.population_size,
            'iterations': self.max_iterations
        }
        
        return best_tour, best_distance, history
    
    def visualize_solution(self, tour, distance_matrix=None, title="AOA TSP Solution"):
        """
        Visualize the TSP tour
        
        Parameters:
        -----------
        tour : array-like
            City sequence to visualize
        distance_matrix : np.ndarray, optional
            Distance matrix for computing tour cost
        title : str
            Plot title
        """
        if distance_matrix is None:
            distance_matrix = self.distance_matrix
        
        n = len(tour)
        
        # Generate random coordinates for visualization
        np.random.seed(42)
        coords = np.random.rand(n, 2) * 100
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot cities
        ax.scatter(coords[:, 0], coords[:, 1], s=200, c='red', zorder=3, edgecolors='black', linewidth=2)
        
        # Add city labels
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), fontsize=10, ha='center', va='center', color='white', weight='bold')
        
        # Plot tour
        for i in range(len(tour)):
            start = coords[tour[i]]
            end = coords[tour[(i + 1) % len(tour)]]
            ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                    head_width=1.5, head_length=1, fc='blue', ec='blue', alpha=0.6, length_includes_head=True)
        
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\nDistance: {distance_matrix.sum():.2f}", fontsize=14, weight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_convergence(self, history, title="AOA Convergence Curve"):
        """
        Plot convergence curve
        
        Parameters:
        -----------
        history : dict
            History dictionary from solve()
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = range(len(history['convergence_curve']))
        ax.plot(iterations, history['convergence_curve'], 'b-', linewidth=2.5, label='Best Distance')
        ax.fill_between(iterations, history['convergence_curve'], alpha=0.3)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Tour Distance', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_fitness_analysis(self, history, title="AOA Fitness Analysis"):
        """
        Plot best and average fitness over iterations
        
        Parameters:
        -----------
        history : dict
            History dictionary from solve()
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = range(len(history['best_distances']))
        ax.plot(iterations, history['best_distances'], 'b-', linewidth=2.5, label='Best Fitness')
        ax.plot(iterations, history['average_fitness'], 'r--', linewidth=2, label='Average Fitness')
        ax.fill_between(iterations, history['best_distances'], alpha=0.2, color='blue')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Fitness (Distance)', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_diversity(self, history, title="AOA Population Diversity"):
        """
        Plot population diversity over iterations
        
        Parameters:
        -----------
        history : dict
            History dictionary from solve()
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = range(len(history['diversity']))
        ax.plot(iterations, history['diversity'], 'g-', linewidth=2.5)
        ax.fill_between(iterations, history['diversity'], alpha=0.3, color='green')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Population Diversity (Std Dev)', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax


def compare_multiple_runs(distance_matrix, algorithm_class, n_runs=10, **kwargs):
    """
    Run algorithm multiple times and compare results
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix
    algorithm_class : class
        Algorithm class (AOA_TSP or AGWO_TSP)
    n_runs : int
        Number of runs
    **kwargs : dict
        Algorithm parameters
    
    Returns:
    --------
    results : dict
        Statistics from multiple runs
    """
    results = {
        'best_distances': [],
        'avg_distances': [],
        'convergence_curves': [],
        'final_solutions': []
    }
    
    for run in range(n_runs):
        algo = algorithm_class(distance_matrix, random_seed=run, **kwargs)
        tour, distance, history = algo.solve(verbose=False)
        
        results['best_distances'].append(distance)
        results['avg_distances'].append(np.mean(history['best_distances']))
        results['convergence_curves'].append(history['convergence_curve'])
        results['final_solutions'].append(tour)
    
    return results


def plot_comparison_bars(results_dict, algorithm_names, title="Algorithm Comparison"):
    """
    Plot bar chart comparing multiple algorithms
    
    Parameters:
    -----------
    results_dict : dict
        Results from compare_multiple_runs for each algorithm
    algorithm_names : list
        Names of algorithms
    title : str
        Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Best distances
    best_means = [np.mean(results_dict[name]['best_distances']) for name in algorithm_names]
    best_stds = [np.std(results_dict[name]['best_distances']) for name in algorithm_names]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithm_names)]
    
    axes[0].bar(algorithm_names, best_means, yerr=best_stds, capsize=10, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Best Distance', fontsize=12)
    axes[0].set_title('Best Solution Quality', fontsize=13, weight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(best_means, best_stds)):
        axes[0].text(i, mean + std + 50, f'{mean:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Average distances
    avg_means = [np.mean([np.mean(curve) for curve in results_dict[name]['convergence_curves']]) 
                 for name in algorithm_names]
    avg_stds = [np.std([np.mean(curve) for curve in results_dict[name]['convergence_curves']]) 
               for name in algorithm_names]
    
    axes[1].bar(algorithm_names, avg_means, yerr=avg_stds, capsize=10, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Average Fitness', fontsize=12)
    axes[1].set_title('Average Fitness During Search', fontsize=13, weight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(avg_means, avg_stds)):
        axes[1].text(i, mean + std + 50, f'{mean:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    return fig, axes


def plot_convergence_comparison(results_dict, algorithm_names, title="Convergence Comparison"):
    """
    Plot convergence curves of multiple algorithms
    
    Parameters:
    -----------
    results_dict : dict
        Results from multiple runs
    algorithm_names : list
        Names of algorithms
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithm_names)]
    
    for algo_name, color in zip(algorithm_names, colors):
        # Average convergence across runs
        curves = results_dict[algo_name]['convergence_curves']
        avg_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        
        iterations = range(len(avg_curve))
        ax.plot(iterations, avg_curve, color=color, linewidth=2.5, label=algo_name, marker='o', markersize=3, markevery=10)
        ax.fill_between(iterations, 
                       np.array(avg_curve) - np.array(std_curve),
                       np.array(avg_curve) + np.array(std_curve),
                       color=color, alpha=0.2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Distance', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


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
    print("ARITHMETIC OPTIMIZATION ALGORITHM FOR TSP - WITH VISUALIZATIONS")
    print("=" * 70)
    
    # Single run
    print("\n1. Running AOA (single run)...")
    aoa = AOA_TSP(distance_matrix, population_size=30, max_iterations=200)
    best_tour, best_distance, history = aoa.solve(verbose=True)
    
    print(f"\nBest tour found: {best_tour}")
    print(f"Best distance: {best_distance:.2f}")
    
    # Visualizations
    print("\n2. Creating visualizations...")
    
    # Plot 1: Tour visualization
    fig1, ax1 = aoa.visualize_solution(best_tour, title="AOA TSP Solution Route")
    plt.savefig('aoa_route_visualization.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: aoa_route_visualization.png")
    
    # Plot 2: Convergence curve
    fig2, ax2 = aoa.plot_convergence(history, title="AOA Convergence Curve")
    plt.savefig('aoa_convergence_curve.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: aoa_convergence_curve.png")
    
    # Plot 3: Fitness analysis
    fig3, ax3 = aoa.plot_fitness_analysis(history, title="AOA Fitness Analysis")
    plt.savefig('aoa_fitness_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: aoa_fitness_analysis.png")
    
    # Plot 4: Diversity
    fig4, ax4 = aoa.plot_diversity(history, title="AOA Population Diversity")
    plt.savefig('aoa_diversity.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: aoa_diversity.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("AOA implementation complete with full visualization support!")
    print("=" * 70)
