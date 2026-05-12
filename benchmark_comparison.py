"""
Benchmarking Script: AOA vs AGWO for TSP

This script compares the performance of Arithmetic Optimization Algorithm (AOA)
and Adaptive Grey Wolf Optimization (AGWO) on various TSP instances.

Features:
- Run both algorithms on multiple problem sizes
- Multiple runs with different random seeds for statistical analysis
- Generate comparison statistics
- Create convergence visualization
- Export results to CSV files

Usage:
    python benchmark_comparison.py

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import warnings
import os

# Import the algorithm implementations
from AOA_TSP import AOA_TSP
from AGWO_TSP import AGWO_TSP

warnings.filterwarnings('ignore')


class TSPBenchmark:
    """
    Benchmark suite for comparing AOA and AGWO algorithms on TSP instances.
    """
    
    def __init__(self,
                 population_size: int = 30,
                 max_iterations: int = 500,
                 num_runs: int = 10):
        """
        Initialize benchmark suite.
        
        Parameters:
        -----------
        population_size : int
            Population size for both algorithms
        max_iterations : int
            Maximum iterations for both algorithms
        num_runs : int
            Number of independent runs for each algorithm
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.num_runs = num_runs
        
        self.results = {
            'AOA': {},
            'AGWO': {}
        }
    
    def generate_random_tsp(self, num_cities: int, seed: int = None) -> np.ndarray:
        """
        Generate a random TSP instance with Euclidean distances.
        
        Parameters:
        -----------
        num_cities : int
            Number of cities in the instance
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray : Distance matrix (num_cities x num_cities)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random city coordinates in [0, 100] x [0, 100]
        coordinates = np.random.uniform(0, 100, size=(num_cities, 2))
        
        # Calculate Euclidean distance matrix
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distance_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])
        
        return distance_matrix
    
    def run_aoa(self, distance_matrix: np.ndarray, seed: int) -> Tuple[float, float, List]:
        """
        Run AOA algorithm on a TSP instance.
        
        Returns:
        --------
        best_distance : float
        runtime : float (in seconds)
        convergence_curve : list
        """
        start_time = time.time()
        
        algorithm = AOA_TSP(
            distance_matrix=distance_matrix,
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            random_seed=seed
        )
        
        best_tour, best_distance, history = algorithm.solve()
        
        runtime = time.time() - start_time
        convergence_curve = history['convergence_curve']
        
        return best_distance, runtime, convergence_curve
    
    def run_agwo(self, distance_matrix: np.ndarray, seed: int) -> Tuple[float, float, List]:
        """
        Run AGWO algorithm on a TSP instance.
        
        Returns:
        --------
        best_distance : float
        runtime : float (in seconds)
        convergence_curve : list
        """
        start_time = time.time()
        
        algorithm = AGWO_TSP(
            distance_matrix=distance_matrix,
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            random_seed=seed
        )
        
        best_tour, best_distance, history = algorithm.solve()
        
        runtime = time.time() - start_time
        convergence_curve = history['convergence_curve']
        
        return best_distance, runtime, convergence_curve
    
    def benchmark_problem_size(self, num_cities: int) -> Dict:
        """
        Run complete benchmark for a specific problem size.
        
        Parameters:
        -----------
        num_cities : int
            Number of cities in the TSP instance
            
        Returns:
        --------
        dict : Results containing statistics for both algorithms
        """
        print(f"\n{'='*70}")
        print(f"Benchmarking {num_cities}-city TSP")
        print(f"{'='*70}")
        
        aoa_results = []
        agwo_results = []
        aoa_runtimes = []
        agwo_runtimes = []
        aoa_convergence = []
        agwo_convergence = []
        
        # Generate base distance matrix
        base_distance_matrix = self.generate_random_tsp(num_cities, seed=42)
        
        # Run multiple independent executions
        for run in range(self.num_runs):
            print(f"Run {run + 1}/{self.num_runs}...", end=" ")
            
            # Run AOA
            aoa_dist, aoa_time, aoa_conv = self.run_aoa(base_distance_matrix, seed=run)
            aoa_results.append(aoa_dist)
            aoa_runtimes.append(aoa_time)
            aoa_convergence.append(aoa_conv)
            
            # Run AGWO
            agwo_dist, agwo_time, agwo_conv = self.run_agwo(base_distance_matrix, seed=run)
            agwo_results.append(agwo_dist)
            agwo_runtimes.append(agwo_time)
            agwo_convergence.append(agwo_conv)
            
            print(f"AOA: {aoa_dist:.2f}, AGWO: {agwo_dist:.2f}")
        
        # Calculate statistics
        results = {
            'num_cities': num_cities,
            'aoa': {
                'mean': np.mean(aoa_results),
                'std': np.std(aoa_results),
                'min': np.min(aoa_results),
                'max': np.max(aoa_results),
                'median': np.median(aoa_results),
                'mean_runtime': np.mean(aoa_runtimes),
                'convergence_curves': aoa_convergence,
                'raw_results': aoa_results
            },
            'agwo': {
                'mean': np.mean(agwo_results),
                'std': np.std(agwo_results),
                'min': np.min(agwo_results),
                'max': np.max(agwo_results),
                'median': np.median(agwo_results),
                'mean_runtime': np.mean(agwo_runtimes),
                'convergence_curves': agwo_convergence,
                'raw_results': agwo_results
            }
        }
        
        # Print summary
        print(f"\n{'Algorithm':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Runtime':<10}")
        print("-" * 70)
        print(f"{'AOA':<15} {results['aoa']['mean']:>11.2f} {results['aoa']['std']:>11.2f} "
              f"{results['aoa']['min']:>11.2f} {results['aoa']['max']:>11.2f} {results['aoa']['mean_runtime']:>9.2f}s")
        print(f"{'AGWO':<15} {results['agwo']['mean']:>11.2f} {results['agwo']['std']:>11.2f} "
              f"{results['agwo']['min']:>11.2f} {results['agwo']['max']:>11.2f} {results['agwo']['mean_runtime']:>9.2f}s")
        
        # Calculate improvement
        improvement = ((results['agwo']['mean'] - results['aoa']['mean']) / results['agwo']['mean']) * 100
        print(f"\n{'AOA vs AGWO':<15} Improvement: {improvement:>6.2f}% {'(favoring AOA)' if improvement > 0 else '(favoring AGWO)'}")
        
        return results
    
    def run_full_benchmark(self, problem_sizes: List[int] = None) -> Dict:
        """
        Run benchmark on multiple problem sizes.
        
        Parameters:
        -----------
        problem_sizes : list
            List of problem sizes to test (default: [10, 20, 30, 50])
            
        Returns:
        --------
        dict : Complete benchmark results
        """
        if problem_sizes is None:
            problem_sizes = [10, 20, 30, 50]
        
        all_results = {}
        
        for size in problem_sizes:
            results = self.benchmark_problem_size(size)
            all_results[size] = results
        
        return all_results
    
    def generate_report(self, all_results: Dict) -> str:
        """
        Generate a formatted text report of benchmark results.
        
        Parameters:
        -----------
        all_results : dict
            Results from run_full_benchmark()
            
        Returns:
        --------
        str : Formatted report
        """
        report = []
        report.append("=" * 80)
        report.append("TRAVELING SALESMAN PROBLEM: AOA vs AGWO BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"\nBenchmark Configuration:")
        report.append(f"  Population Size: {self.population_size}")
        report.append(f"  Max Iterations: {self.max_iterations}")
        report.append(f"  Runs per Instance: {self.num_runs}")
        report.append("\n" + "=" * 80)
        
        report.append("\nDETAILED RESULTS BY PROBLEM SIZE:")
        report.append("=" * 80)
        
        for size in sorted(all_results.keys()):
            results = all_results[size]
            report.append(f"\n{size}-City TSP Instance")
            report.append("-" * 80)
            
            report.append("\nArithmetic Optimization Algorithm (AOA):")
            report.append(f"  Mean Distance:   {results['aoa']['mean']:.2f}")
            report.append(f"  Std Deviation:   {results['aoa']['std']:.2f}")
            report.append(f"  Min Distance:    {results['aoa']['min']:.2f}")
            report.append(f"  Max Distance:    {results['aoa']['max']:.2f}")
            report.append(f"  Median Distance: {results['aoa']['median']:.2f}")
            report.append(f"  Avg Runtime:     {results['aoa']['mean_runtime']:.2f}s")
            
            report.append("\nAdaptive Grey Wolf Optimization (AGWO):")
            report.append(f"  Mean Distance:   {results['agwo']['mean']:.2f}")
            report.append(f"  Std Deviation:   {results['agwo']['std']:.2f}")
            report.append(f"  Min Distance:    {results['agwo']['min']:.2f}")
            report.append(f"  Max Distance:    {results['agwo']['max']:.2f}")
            report.append(f"  Median Distance: {results['agwo']['median']:.2f}")
            report.append(f"  Avg Runtime:     {results['agwo']['mean_runtime']:.2f}s")
            
            # Comparison
            aoa_mean = results['aoa']['mean']
            agwo_mean = results['agwo']['mean']
            improvement = ((agwo_mean - aoa_mean) / agwo_mean) * 100
            winner = "AOA" if aoa_mean < agwo_mean else "AGWO"
            
            report.append(f"\nComparison:")
            report.append(f"  Winner: {winner}")
            report.append(f"  Difference: {abs(aoa_mean - agwo_mean):.2f}")
            report.append(f"  AOA Improvement: {improvement:.2f}%")
        
        report.append("\n" + "=" * 80)
        report.append("OVERALL SUMMARY")
        report.append("=" * 80)
        
        # Calculate overall statistics
        all_aoa_means = [results['aoa']['mean'] for results in all_results.values()]
        all_agwo_means = [results['agwo']['mean'] for results in all_results.values()]
        
        report.append(f"\nAcross all problem sizes:")
        report.append(f"  AOA Average:     {np.mean(all_aoa_means):.2f} ± {np.std(all_aoa_means):.2f}")
        report.append(f"  AGWO Average:    {np.mean(all_agwo_means):.2f} ± {np.std(all_agwo_means):.2f}")
        report.append(f"  Avg Difference:  {np.mean(np.array(all_aoa_means) - np.array(all_agwo_means)):.2f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_csv_results(self, all_results: Dict, output_dir: str = ".") -> None:
        """
        Save benchmark results to CSV files.
        
        Parameters:
        -----------
        all_results : dict
            Results from run_full_benchmark()
        output_dir : str
            Directory to save CSV files
        """
        import csv
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_dir, 'benchmark_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Problem Size', 'Algorithm', 'Mean', 'Std Dev', 'Min', 'Max', 'Median', 'Avg Runtime'])
            
            for size in sorted(all_results.keys()):
                results = all_results[size]
                
                writer.writerow([
                    size, 'AOA',
                    f"{results['aoa']['mean']:.2f}",
                    f"{results['aoa']['std']:.2f}",
                    f"{results['aoa']['min']:.2f}",
                    f"{results['aoa']['max']:.2f}",
                    f"{results['aoa']['median']:.2f}",
                    f"{results['aoa']['mean_runtime']:.2f}"
                ])
                
                writer.writerow([
                    size, 'AGWO',
                    f"{results['agwo']['mean']:.2f}",
                    f"{results['agwo']['std']:.2f}",
                    f"{results['agwo']['min']:.2f}",
                    f"{results['agwo']['max']:.2f}",
                    f"{results['agwo']['median']:.2f}",
                    f"{results['agwo']['mean_runtime']:.2f}"
                ])
        
        print(f"\nResults saved to {os.path.join(output_dir, 'benchmark_results.csv')}")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    """
    Run complete benchmark comparison between AOA and AGWO.
    """
    
    print("\n" + "=" * 70)
    print("TSP ALGORITHM BENCHMARK: AOA vs AGWO")
    print("=" * 70)
    
    # Initialize benchmark suite
    benchmark = TSPBenchmark(
        population_size=30,
        max_iterations=500,
        num_runs=10  # 10 runs for each algorithm on each instance
    )
    
    # Run benchmark on multiple problem sizes
    problem_sizes = [10, 20, 30]  # Can add 50 for larger instances (slower)
    all_results = benchmark.run_full_benchmark(problem_sizes)
    
    # Generate and display report
    report = benchmark.generate_report(all_results)
    print(report)
    
    # Save results to CSV
    benchmark.save_csv_results(all_results, output_dir='./benchmark_output')
    
    print("\n✓ Benchmark completed successfully!")
    print("  Results saved to: ./benchmark_output/benchmark_results.csv")
