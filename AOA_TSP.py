"""
Arithmetic Optimization Algorithm (AOA) for Traveling Salesman Problem
Implementation based on: Abualigah, L., et al. (2021)

This module implements the Arithmetic Optimization Algorithm adapted for solving
the Traveling Salesman Problem (TSP) using position-based encoding and Largest
Order Value (LOV) permutation mapping for discrete tour representation.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')


class AOA_TSP:
    """
    Arithmetic Optimization Algorithm implementation for Traveling Salesman Problem.
    
    The algorithm uses arithmetic operators (addition, subtraction, multiplication, division)
    to explore and exploit the solution space, dynamically switching between exploration
    and exploitation phases through MOP (Math Optimizer Probability) and MOR 
    (Math Optimizer Ratio) parameters.
    """
    
    def __init__(self, 
                 distance_matrix: np.ndarray,
                 population_size: int = 30,
                 max_iterations: int = 500,
                 alpha: float = 5.0,
                 epsilon: float = 1e-10,
                 random_seed: int = None):
        """
        Initialize AOA for TSP.
        
        Parameters:
        -----------
        distance_matrix : np.ndarray
            Square symmetric matrix of distances between cities (n x n)
        population_size : int
            Number of solutions in the population (default: 30)
        max_iterations : int
            Maximum number of iterations (default: 500)
        alpha : float
            Tuning parameter for MOR calculation (default: 5.0)
        epsilon : float
            Small constant to prevent division by zero (default: 1e-10)
        random_seed : int
            Random seed for reproducibility (default: None)
        """
        
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.epsilon = epsilon
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize tracking variables
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.convergence_curve = []
    
    def _calculate_tour_distance(self, tour: np.ndarray) -> float:
        """
        Calculate total distance of a tour (fitness function to minimize).
        
        Parameters:
        -----------
        tour : np.ndarray
            Array of city indices representing the tour
            
        Returns:
        --------
        float : Total distance traveled in the tour
        """
        distance = 0.0
        for i in range(len(tour) - 1):
            distance += self.distance_matrix[int(tour[i]), int(tour[i + 1])]
        
        # Add distance from last city back to first city (return to origin)
        distance += self.distance_matrix[int(tour[-1]), int(tour[0])]
        
        return distance
    
    def _lov_permutation(self, continuous_solution: np.ndarray) -> np.ndarray:
        """
        Convert continuous values to valid tour using Largest Order Value (LOV) method.
        
        This is crucial for adapting the continuous AOA algorithm to the discrete TSP.
        The method ranks continuous values and uses these ranks as city indices.
        
        Parameters:
        -----------
        continuous_solution : np.ndarray
            Continuous values in range [0, 1), one value per city
            
        Returns:
        --------
        np.ndarray : Valid tour as permutation of city indices [0, 1, ..., n-1]
        
        Algorithm:
        ----------
        1. Get the rank of each position when values are sorted
        2. The ranks represent the order cities should be visited
        3. This ensures each city appears exactly once in the tour
        """
        # Ensure values are in valid range
        continuous_solution = np.clip(continuous_solution, 0, 1 - self.epsilon)
        
        # Get argsort which gives indices that would sort the array
        # These indices represent the city order
        tour = np.argsort(continuous_solution)
        
        return tour.astype(int)
    
    def _initialize_population(self) -> np.ndarray:
        """
        Initialize population with random continuous solutions.
        
        Returns:
        --------
        np.ndarray : Population matrix of shape (population_size, num_cities)
                    Each row is a continuous solution in [0, 1)
        """
        population = np.random.uniform(0, 1, (self.population_size, self.num_cities))
        return population
    
    def _calculate_mop(self, iteration: int) -> float:
        """
        Calculate Math Optimizer Probability (MOP).
        
        MOP controls the balance between exploration and exploitation.
        As iterations progress, MOP increases from 0 to 1, shifting from
        exploration-heavy to exploitation-heavy phases.
        
        Formula: MOP(t) = min(t / T, 1)
        
        Parameters:
        -----------
        iteration : int
            Current iteration number (0-indexed)
            
        Returns:
        --------
        float : MOP value in range [0, 1]
        """
        return min(iteration / self.max_iterations, 1.0)
    
    def _calculate_mor(self, iteration: int) -> float:
        """
        Calculate Math Optimizer Ratio (MOR).
        
        MOR determines which arithmetic operator to use (division/multiplication
        for exploration, subtraction/addition for exploitation).
        
        Formula: MOR(t) = 1 - (t^(1/α)) / (T^(1/α))
        
        Parameters:
        -----------
        iteration : int
            Current iteration number (0-indexed)
            
        Returns:
        --------
        float : MOR value in range [0, 1]
        """
        return 1 - ((iteration ** (1 / self.alpha)) / (self.max_iterations ** (1 / self.alpha)))
    
    def _aoa_division(self, 
                     solution: np.ndarray, 
                     best_solution: np.ndarray) -> np.ndarray:
        """
        AOA Division operator (exploration phase).
        
        Formula: X_i = S_best / (ε + |rand(0,1) × S_best - X_i|)
        
        This operator helps explore new regions by dividing the best solution
        by a normalized difference term.
        
        Parameters:
        -----------
        solution : np.ndarray
            Current solution
        best_solution : np.ndarray
            Current best solution found
            
        Returns:
        --------
        np.ndarray : Updated solution
        """
        denominator = self.epsilon + np.abs(
            np.random.uniform(0, 1, self.num_cities) * best_solution - solution
        )
        new_solution = best_solution / denominator
        
        # Clip to valid range [0, 1)
        return np.clip(new_solution, 0, 1 - self.epsilon)
    
    def _aoa_multiplication(self, 
                           solution: np.ndarray, 
                           best_solution: np.ndarray) -> np.ndarray:
        """
        AOA Multiplication operator (exploration phase).
        
        Formula: X_i = S_best × rand(0,1)
        
        Simple exploration operator that scales the best solution by random factors.
        
        Parameters:
        -----------
        solution : np.ndarray
            Current solution (used for context)
        best_solution : np.ndarray
            Current best solution found
            
        Returns:
        --------
        np.ndarray : Updated solution
        """
        new_solution = best_solution * np.random.uniform(0, 1, self.num_cities)
        return np.clip(new_solution, 0, 1 - self.epsilon)
    
    def _aoa_subtraction(self, 
                        solution: np.ndarray, 
                        best_solution: np.ndarray) -> np.ndarray:
        """
        AOA Subtraction operator (exploitation phase).
        
        Formula: X_i = S_best - rand(0,1) × X_i
        
        Exploitation operator that moves current solution towards the best solution.
        
        Parameters:
        -----------
        solution : np.ndarray
            Current solution
        best_solution : np.ndarray
            Current best solution found
            
        Returns:
        --------
        np.ndarray : Updated solution
        """
        new_solution = best_solution - np.random.uniform(0, 1, self.num_cities) * solution
        return np.clip(new_solution, 0, 1 - self.epsilon)
    
    def _aoa_addition(self, 
                     solution: np.ndarray, 
                     best_solution: np.ndarray) -> np.ndarray:
        """
        AOA Addition operator (exploitation phase).
        
        Formula: X_i = S_best + rand(0,1) × X_i
        
        Exploitation operator that moves towards the best solution with exploration component.
        
        Parameters:
        -----------
        solution : np.ndarray
            Current solution
        best_solution : np.ndarray
            Current best solution found
            
        Returns:
        --------
        np.ndarray : Updated solution
        """
        new_solution = best_solution + np.random.uniform(0, 1, self.num_cities) * solution
        return np.clip(new_solution, 0, 1 - self.epsilon)
    
    def solve(self) -> Tuple[np.ndarray, float, Dict]:
        """
        Execute the AOA algorithm for TSP.
        
        Returns:
        --------
        best_tour : np.ndarray
            The best tour found (sequence of city indices)
        best_distance : float
            Total distance of the best tour
        history : dict
            Dictionary containing convergence history and statistics
        
        Algorithm Flow:
        ---------------
        1. Initialize population with random continuous solutions
        2. For each iteration:
            a. Calculate MOP and MOR parameters
            b. For each solution:
                - Choose exploration or exploitation based on MOP
                - Choose operator (division/mult or sub/add) based on MOR
                - Apply selected operator
                - Convert to valid tour using LOV
                - Evaluate fitness
                - Update best solution if improved
            c. Record best fitness for convergence analysis
        3. Return best solution found
        """
        
        # Step 1: Initialize population
        population = self._initialize_population()
        
        # Evaluate initial population
        fitness_values = np.array([
            self._calculate_tour_distance(self._lov_permutation(solution))
            for solution in population
        ])
        
        # Find initial best solution
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        
        # Step 2: Main optimization loop
        for iteration in range(self.max_iterations):
            mop = self._calculate_mop(iteration)
            mor = self._calculate_mor(iteration)
            
            # Update each solution in the population
            for i in range(self.population_size):
                rand_val = np.random.uniform()
                
                # Exploration phase
                if rand_val > mop:
                    if np.random.uniform() > mor:
                        # Division operator
                        new_solution = self._aoa_division(population[i], self.best_solution)
                    else:
                        # Multiplication operator
                        new_solution = self._aoa_multiplication(population[i], self.best_solution)
                
                # Exploitation phase
                else:
                    if np.random.uniform() > mor:
                        # Subtraction operator
                        new_solution = self._aoa_subtraction(population[i], self.best_solution)
                    else:
                        # Addition operator
                        new_solution = self._aoa_addition(population[i], self.best_solution)
                
                # Convert continuous solution to valid tour and evaluate
                tour = self._lov_permutation(new_solution)
                fitness = self._calculate_tour_distance(tour)
                
                # Update solution if improved
                if fitness < fitness_values[i]:
                    population[i] = new_solution
                    fitness_values[i] = fitness
                
                # Update global best if improved
                if fitness < self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = fitness
            
            # Record convergence history
            self.convergence_curve.append(self.best_fitness)
            
            # Print progress (every 50 iterations)
            if (iteration + 1) % 50 == 0:
                print(f"AOA Iteration {iteration + 1}/{self.max_iterations} - Best Distance: {self.best_fitness:.2f}")
        
        # Convert best solution to tour
        best_tour = self._lov_permutation(self.best_solution)
        
        # Compile statistics
        history = {
            'best_distance': self.best_fitness,
            'best_tour': best_tour,
            'convergence_curve': self.convergence_curve,
            'total_iterations': self.max_iterations,
            'population_size': self.population_size,
            'avg_distance': np.mean(self.convergence_curve),
            'std_distance': np.std(self.convergence_curve)
        }
        
        return best_tour, self.best_fitness, history


# =====================================================================
# Example Usage and Testing
# =====================================================================

if __name__ == "__main__":
    """
    Example: Solve a small TSP instance with 5 cities
    """
    
    # Define distance matrix (symmetric TSP)
    # This is a simple 5-city example
    example_distance_matrix = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 20],
        [20, 25, 30, 0, 15],
        [25, 30, 20, 15, 0]
    ])
    
    # Initialize and run AOA
    aoa = AOA_TSP(
        distance_matrix=example_distance_matrix,
        population_size=20,
        max_iterations=200,
        random_seed=42
    )
    
    print("Running Arithmetic Optimization Algorithm (AOA) for TSP...")
    print("=" * 60)
    
    best_tour, best_distance, history = aoa.solve()
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Best Tour Found: {best_tour}")
    print(f"Best Distance: {best_distance:.2f}")
    print(f"Average Distance (across iterations): {history['avg_distance']:.2f}")
    print(f"Standard Deviation: {history['std_distance']:.2f}")
    print(f"Final Convergence Value: {history['convergence_curve'][-1]:.2f}")
