"""
Adaptive Grey Wolf Optimization (AGWO) for Traveling Salesman Problem
Implementation based on: Mirjalili et al. (2014) with adaptive enhancements

This module implements the Adaptive Grey Wolf Optimization algorithm adapted for
solving the Traveling Salesman Problem (TSP) using position-based encoding and
Largest Order Value (LOV) permutation mapping for discrete tour representation.

The algorithm incorporates adaptive mechanisms including dynamic control parameter
adjustment and diversity-based perturbations to prevent premature convergence.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')


class AGWO_TSP:
    """
    Adaptive Grey Wolf Optimization implementation for Traveling Salesman Problem.
    
    The algorithm simulates the hunting behavior of grey wolves with hierarchical
    leadership (Alpha, Beta, Delta, Omega). Adaptive mechanisms include:
    - Dynamic control parameter 'a' that smoothly transitions from exploration to exploitation
    - Diversity monitoring to prevent premature convergence
    - Adaptive perturbation strategy for stagnant solutions
    """
    
    def __init__(self,
                 distance_matrix: np.ndarray,
                 population_size: int = 30,
                 max_iterations: int = 500,
                 gamma: float = 2.5,
                 diversity_threshold: float = 0.1,
                 epsilon: float = 1e-10,
                 random_seed: int = None):
        """
        Initialize AGWO for TSP.
        
        Parameters:
        -----------
        distance_matrix : np.ndarray
            Square symmetric matrix of distances between cities (n x n)
        population_size : int
            Number of wolves (solutions) in the pack (default: 30)
        max_iterations : int
            Maximum number of iterations (default: 500)
        gamma : float
            Adaptivity parameter controlling exploration-exploitation transition (default: 2.5)
        diversity_threshold : float
            Threshold below which diversity recovery is triggered (default: 0.1)
        epsilon : float
            Small constant for numerical stability (default: 1e-10)
        random_seed : int
            Random seed for reproducibility (default: None)
        """
        
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.diversity_threshold = diversity_threshold
        self.epsilon = epsilon
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize tracking variables
        self.alpha = None  # Best solution (first rank)
        self.beta = None   # Second-best solution
        self.delta = None  # Third-best solution
        
        self.alpha_fitness = float('inf')
        self.beta_fitness = float('inf')
        self.delta_fitness = float('inf')
        
        self.fitness_history = []
        self.convergence_curve = []
        self.diversity_history = []
    
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
        
        # Add distance from last city back to first city
        distance += self.distance_matrix[int(tour[-1]), int(tour[0])]
        
        return distance
    
    def _lov_permutation(self, continuous_solution: np.ndarray) -> np.ndarray:
        """
        Convert continuous values to valid tour using Largest Order Value (LOV) method.
        
        This is crucial for adapting the continuous AGWO algorithm to the discrete TSP.
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
        
        # Argsort gives indices that would sort the array - these are city orders
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
    
    def _calculate_adaptive_a(self, iteration: int) -> float:
        """
        Calculate adaptive control parameter 'a' for this iteration.
        
        The adaptive formula ensures smooth transition from exploration to exploitation:
        
        Formula: a(t) = 2 - 2 * (t / T)^γ
        
        At t=0: a ≈ 2 (high exploration)
        At t=T: a ≈ 0 (high exploitation)
        
        The exponent γ controls the transition smoothness:
        - γ < 1: Sharp transition
        - γ ≈ 2.5: Smooth transition (recommended)
        - γ > 3: Very gradual transition
        
        Parameters:
        -----------
        iteration : int
            Current iteration number (0-indexed)
            
        Returns:
        --------
        float : Adaptive parameter 'a' value in range [0, 2]
        """
        progress = iteration / self.max_iterations
        a = 2 - 2 * (progress ** self.gamma)
        return max(0, a)  # Ensure non-negative
    
    def _hamming_distance(self, tour1: np.ndarray, tour2: np.ndarray) -> int:
        """
        Calculate Hamming distance between two tours (permutations).
        
        Hamming distance counts the number of positions where the two tours differ.
        It's used to measure diversity in the population.
        
        Parameters:
        -----------
        tour1, tour2 : np.ndarray
            Two tours to compare
            
        Returns:
        --------
        int : Number of differing positions
        """
        return np.sum(tour1 != tour2)
    
    def _calculate_diversity(self, population: np.ndarray) -> float:
        """
        Calculate population diversity metric.
        
        Diversity is calculated as the average Hamming distance of each solution
        from the best (alpha) solution, normalized by the maximum possible distance.
        
        Parameters:
        -----------
        population : np.ndarray
            Current population of continuous solutions
            
        Returns:
        --------
        float : Diversity measure in range [0, 1]
                0 = no diversity (all solutions identical)
                1 = maximum diversity
        """
        # Convert to tours for distance calculation
        alpha_tour = self._lov_permutation(self.alpha)
        
        total_distance = 0.0
        for solution in population:
            tour = self._lov_permutation(solution)
            total_distance += self._hamming_distance(alpha_tour, tour)
        
        # Normalize by maximum possible distance
        max_distance = self.population_size * self.num_cities
        diversity = total_distance / max_distance if max_distance > 0 else 0
        
        return diversity
    
    def _apply_perturbation(self, solution: np.ndarray) -> np.ndarray:
        """
        Apply perturbation to solution to restore diversity.
        
        When diversity falls below threshold, this function randomly modifies
        a solution to encourage exploration of new regions.
        
        Strategy: Randomly shuffle a subset of cities in the tour
        
        Parameters:
        -----------
        solution : np.ndarray
            Current solution
            
        Returns:
        --------
        np.ndarray : Perturbed solution
        """
        perturbed = solution.copy()
        
        # Randomly select 2-4 positions and scramble their order
        num_perturbations = np.random.randint(2, min(5, self.num_cities))
        perturbation_indices = np.random.choice(self.num_cities, num_perturbations, replace=False)
        
        # Add small random noise to these positions
        noise = np.random.uniform(-0.3, 0.3, num_perturbations)
        perturbed[perturbation_indices] += noise
        
        # Clip to valid range
        perturbed = np.clip(perturbed, 0, 1 - self.epsilon)
        
        return perturbed
    
    def _update_hierarchy(self, 
                         new_solution: np.ndarray,
                         new_fitness: float) -> None:
        """
        Update the hierarchical ranking (Alpha, Beta, Delta).
        
        Maintains the three best solutions found so far, which guide the optimization.
        
        Parameters:
        -----------
        new_solution : np.ndarray
            New solution to potentially add to hierarchy
        new_fitness : float
            Fitness value of the new solution
        """
        # Update Alpha if new solution is better
        if new_fitness < self.alpha_fitness:
            self.delta = self.beta.copy() if self.beta is not None else None
            self.delta_fitness = self.beta_fitness
            
            self.beta = self.alpha.copy() if self.alpha is not None else None
            self.beta_fitness = self.alpha_fitness
            
            self.alpha = new_solution.copy()
            self.alpha_fitness = new_fitness
        
        # Update Beta if new solution is better than Beta but worse than Alpha
        elif new_fitness < self.beta_fitness:
            self.delta = self.beta.copy() if self.beta is not None else None
            self.delta_fitness = self.beta_fitness
            
            self.beta = new_solution.copy()
            self.beta_fitness = new_fitness
        
        # Update Delta if new solution is better than Delta but worse than Beta
        elif new_fitness < self.delta_fitness:
            self.delta = new_solution.copy()
            self.delta_fitness = new_fitness
    
    def _gwo_update_position(self,
                            current_solution: np.ndarray,
                            prey_solution: np.ndarray,
                            a: float) -> np.ndarray:
        """
        Update position of a wolf towards prey using GWO equations.
        
        Equations:
        ----------
        A = 2*a*r1 - a      (coefficient to balance exploration-exploitation)
        C = 2*r2             (coefficient for random weighting)
        D = |C * X_prey - X| (distance to prey)
        X_new = X_prey - A*D  (new position)
        
        Parameters:
        -----------
        current_solution : np.ndarray
            Current position of the wolf
        prey_solution : np.ndarray
            Position of the prey (alpha, beta, or delta)
        a : float
            Adaptive parameter (controls exploration vs exploitation)
            
        Returns:
        --------
        np.ndarray : Updated position
        """
        # Generate random coefficients
        r1 = np.random.uniform(0, 1, self.num_cities)
        r2 = np.random.uniform(0, 1, self.num_cities)
        
        # Calculate A and C coefficients
        A = 2 * a * r1 - a
        C = 2 * r2
        
        # Calculate distance to prey
        D = np.abs(C * prey_solution - current_solution)
        
        # Update position towards prey
        new_solution = prey_solution - A * D
        
        # Clip to valid range [0, 1)
        new_solution = np.clip(new_solution, 0, 1 - self.epsilon)
        
        return new_solution
    
    def solve(self) -> Tuple[np.ndarray, float, Dict]:
        """
        Execute the AGWO algorithm for TSP.
        
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
        2. Identify and set initial Alpha, Beta, Delta
        3. For each iteration:
            a. Calculate adaptive parameter 'a'
            b. Calculate population diversity
            c. If diversity < threshold, apply perturbations
            d. For each wolf:
                - Update position based on Alpha, Beta, and Delta
                - Convert to valid tour using LOV
                - Evaluate fitness
                - Update hierarchy if improved
            e. Record convergence metrics
        4. Return best (Alpha) solution found
        """
        
        # Step 1: Initialize population
        population = self._initialize_population()
        
        # Evaluate initial population
        fitness_values = np.array([
            self._calculate_tour_distance(self._lov_permutation(solution))
            for solution in population
        ])
        
        # Step 2: Identify initial Alpha, Beta, Delta
        sorted_indices = np.argsort(fitness_values)
        
        self.alpha = population[sorted_indices[0]].copy()
        self.alpha_fitness = fitness_values[sorted_indices[0]]
        
        self.beta = population[sorted_indices[1]].copy()
        self.beta_fitness = fitness_values[sorted_indices[1]]
        
        self.delta = population[sorted_indices[2]].copy()
        self.delta_fitness = fitness_values[sorted_indices[2]]
        
        # Step 3: Main optimization loop
        for iteration in range(self.max_iterations):
            
            # Calculate adaptive parameter
            a = self._calculate_adaptive_a(iteration)
            
            # Calculate diversity
            diversity = self._calculate_diversity(population)
            self.diversity_history.append(diversity)
            
            # Apply diversity recovery if needed
            if diversity < self.diversity_threshold and iteration > self.max_iterations * 0.1:
                # Perturb worst solutions
                worst_indices = np.argsort(fitness_values)[-5:]  # 5 worst solutions
                for idx in worst_indices:
                    population[idx] = self._apply_perturbation(population[idx])
            
            # Update each wolf in the pack
            for i in range(self.population_size):
                # Update position based on Alpha (prey location)
                new_sol_alpha = self._gwo_update_position(population[i], self.alpha, a)
                
                # Update position based on Beta
                new_sol_beta = self._gwo_update_position(population[i], self.beta, a)
                
                # Update position based on Delta
                new_sol_delta = self._gwo_update_position(population[i], self.delta, a)
                
                # Average the three updates (standard GWO approach for omega wolves)
                new_solution = (new_sol_alpha + new_sol_beta + new_sol_delta) / 3
                new_solution = np.clip(new_solution, 0, 1 - self.epsilon)
                
                # Convert to tour and evaluate
                tour = self._lov_permutation(new_solution)
                fitness = self._calculate_tour_distance(tour)
                
                # Update population and fitness
                if fitness < fitness_values[i]:
                    population[i] = new_solution
                    fitness_values[i] = fitness
                
                # Update hierarchy if improved
                if fitness < self.alpha_fitness:
                    self._update_hierarchy(new_solution, fitness)
            
            # Record convergence
            self.convergence_curve.append(self.alpha_fitness)
            
            # Print progress
            if (iteration + 1) % 50 == 0:
                print(f"AGWO Iteration {iteration + 1}/{self.max_iterations} - Best Distance: {self.alpha_fitness:.2f} - Diversity: {diversity:.4f}")
        
        # Convert best solution to tour
        best_tour = self._lov_permutation(self.alpha)
        
        # Compile statistics
        history = {
            'best_distance': self.alpha_fitness,
            'best_tour': best_tour,
            'convergence_curve': self.convergence_curve,
            'diversity_history': self.diversity_history,
            'total_iterations': self.max_iterations,
            'population_size': self.population_size,
            'avg_distance': np.mean(self.convergence_curve),
            'std_distance': np.std(self.convergence_curve),
            'avg_diversity': np.mean(self.diversity_history),
            'final_diversity': self.diversity_history[-1]
        }
        
        return best_tour, self.alpha_fitness, history


# =====================================================================
# Example Usage and Testing
# =====================================================================

if __name__ == "__main__":
    """
    Example: Solve a small TSP instance with 5 cities
    """
    
    # Define distance matrix (symmetric TSP)
    example_distance_matrix = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 20],
        [20, 25, 30, 0, 15],
        [25, 30, 20, 15, 0]
    ])
    
    # Initialize and run AGWO
    agwo = AGWO_TSP(
        distance_matrix=example_distance_matrix,
        population_size=20,
        max_iterations=200,
        random_seed=42
    )
    
    print("Running Adaptive Grey Wolf Optimization (AGWO) for TSP...")
    print("=" * 60)
    
    best_tour, best_distance, history = agwo.solve()
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Best Tour Found: {best_tour}")
    print(f"Best Distance: {best_distance:.2f}")
    print(f"Average Distance (across iterations): {history['avg_distance']:.2f}")
    print(f"Standard Deviation: {history['std_distance']:.2f}")
    print(f"Average Population Diversity: {history['avg_diversity']:.4f}")
    print(f"Final Diversity: {history['final_diversity']:.4f}")
