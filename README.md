# Traveling Salesman Problem: AOA vs AGWO Implementation

## Overview

This repository contains implementations of two metaheuristic algorithms for solving the Traveling Salesman Problem (TSP):
- **Arithmetic Optimization Algorithm (AOA)**
- **Adaptive Grey Wolf Optimization (AGWO)**

Both algorithms are implemented in Python with extensive documentation and can be easily benchmarked against each other.

---

## Table of Contents

1. [Installation & Requirements](#installation--requirements)
2. [File Structure](#file-structure)
3. [How to Run](#how-to-run)
4. [Code Structure & Architecture](#code-structure--architecture)
5. [Input Format](#input-format)
6. [Output Format](#output-format)
7. [Example Usage](#example-usage)
8. [Benchmarking Both Algorithms](#benchmarking-both-algorithms)
9. [Customization & Parameters](#customization--parameters)
10. [Troubleshooting](#troubleshooting)

---

## Installation & Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```bash
numpy>=1.19.0
matplotlib>=3.3.0  (optional, for visualization)
pandas>=1.1.0      (optional, for result analysis)
```

### Installation Steps

1. **Clone or download the project files:**
   ```bash
   # Ensure you have the following files:
   # - AOA_TSP.py
   # - AGWO_TSP.py
   # - benchmark_comparison.py
   # - test_data/
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib pandas
   ```

   Or for minimal installation (only numpy required):
   ```bash
   pip install numpy
   ```

3. **Verify installation:**
   ```bash
   python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
   ```

---

## File Structure

```
TSP_Algorithms/
├── AOA_TSP.py                    # Arithmetic Optimization Algorithm implementation
├── AGWO_TSP.py                   # Adaptive Grey Wolf Optimization implementation
├── benchmark_comparison.py        # Script to run both algorithms and compare results
├── README.md                      # This file
├── TSP_Mathematical_Formulation.md  # Mathematical details of both algorithms
└── test_data/
    ├── tsp_5cities.npy           # 5-city test instance
    ├── tsp_10cities.npy          # 10-city test instance
    ├── tsp_20cities.npy          # 20-city test instance
    └── tsp_50cities.npy          # 50-city test instance
```

---

## How to Run

### Option 1: Run Individual Algorithms

#### Run AOA for TSP
```bash
python AOA_TSP.py
```

This will:
- Use a built-in 5-city example
- Run the algorithm for 200 iterations with population size 20
- Display progress every 50 iterations
- Print final results with best tour and distance

**Output Example:**
```
Running Arithmetic Optimization Algorithm (AOA) for TSP...
============================================================
AOA Iteration 50/200 - Best Distance: 85.34
AOA Iteration 100/200 - Best Distance: 78.92
AOA Iteration 150/200 - Best Distance: 78.92
AOA Iteration 200/200 - Best Distance: 78.92

============================================================
RESULTS:
============================================================
Best Tour Found: [0 3 2 4 1]
Best Distance: 78.92
Average Distance (across iterations): 82.15
Standard Deviation: 3.47
Final Convergence Value: 78.92
```

#### Run AGWO for TSP
```bash
python AGWO_TSP.py
```

Similar output to AOA with additional diversity metrics:
```
AGWO Iteration 50/200 - Best Distance: 82.15 - Diversity: 0.3421
AGWO Iteration 100/200 - Best Distance: 78.92 - Diversity: 0.2156
...
Final Diversity: 0.1250
```

### Option 2: Run Comparison Benchmark
```bash
python benchmark_comparison.py
```

This will:
- Test both algorithms on multiple TSP instances
- Run each algorithm 10 times with different random seeds
- Calculate statistics (mean, std, min, max)
- Generate comparison plots
- Save results to CSV files

---

## Code Structure & Architecture

### AOA_TSP.py

**Main Class: `AOA_TSP`**

Key Methods:
- `__init__()`: Initialize algorithm parameters
- `solve()`: Main optimization loop - returns best tour and distance
- `_calculate_tour_distance()`: Evaluate tour length (fitness function)
- `_lov_permutation()`: Convert continuous values to valid tours
- `_calculate_mop()`: Math Optimizer Probability for phase control
- `_calculate_mor()`: Math Optimizer Ratio for operator selection
- `_aoa_division()`, `_aoa_multiplication()`, `_aoa_subtraction()`, `_aoa_addition()`: Arithmetic operators

**Data Flow:**
```
Initialize Random Solutions (continuous)
    ↓
For each iteration:
    - Calculate MOP and MOR parameters
    - For each solution:
      * Select arithmetic operator (exploration or exploitation)
      * Apply operator to generate new solution
      * Convert to valid tour (LOV permutation)
      * Evaluate fitness
      * Update best solution if improved
    ↓
Return best tour and distance
```

### AGWO_TSP.py

**Main Class: `AGWO_TSP`**

Key Methods:
- `__init__()`: Initialize algorithm parameters
- `solve()`: Main optimization loop - returns best tour and distance
- `_calculate_adaptive_a()`: Compute adaptive control parameter
- `_calculate_diversity()`: Monitor population diversity
- `_apply_perturbation()`: Restore diversity when needed
- `_gwo_update_position()`: Update wolf position towards prey
- `_update_hierarchy()`: Maintain Alpha, Beta, Delta ranking

**Data Flow:**
```
Initialize Population + Identify Alpha, Beta, Delta
    ↓
For each iteration:
    - Calculate adaptive parameter 'a'
    - Check diversity, apply perturbations if needed
    - For each wolf:
      * Update position towards Alpha, Beta, and Delta
      * Average the three updates
      * Convert to valid tour (LOV permutation)
      * Evaluate fitness
      * Update hierarchy if improved
    ↓
Return best tour (Alpha) and distance
```

---

## Input Format

### Distance Matrix

Both algorithms expect a **square, symmetric distance matrix** representing the TSP instance.

**Format: NumPy Array (N × N)**

```python
import numpy as np

# Example: 5-city TSP
distance_matrix = np.array([
    [0,    10,   15,   20,   25],    # City 0 distances to all cities
    [10,   0,    35,   25,   30],    # City 1 distances
    [15,   35,   0,    30,   20],    # City 2 distances
    [20,   25,   30,   0,    15],    # City 3 distances
    [25,   30,   20,   15,   0]      # City 4 distances
])
```

**Requirements:**
- Matrix must be square (n × n)
- Diagonal must be zeros: `distance_matrix[i][i] = 0`
- Matrix should be symmetric: `distance_matrix[i][j] = distance_matrix[j][i]`
- All values must be non-negative

### Loading from File

```python
import numpy as np

# Load from .npy file
distance_matrix = np.load('test_data/tsp_10cities.npy')

# Load from CSV file
distance_matrix = np.loadtxt('my_tsp_data.csv', delimiter=',')
```

---

## Output Format

Both algorithms return:

1. **best_tour** (numpy array): City indices in optimal order
   ```python
   array([0, 3, 2, 4, 1])  # Visit cities in this sequence
   ```

2. **best_distance** (float): Total distance of the tour
   ```python
   78.92
   ```

3. **history** (dictionary): Convergence and performance data
   ```python
   {
       'best_distance': 78.92,
       'best_tour': array([0, 3, 2, 4, 1]),
       'convergence_curve': [...],  # Distance at each iteration
       'total_iterations': 200,
       'population_size': 20,
       'avg_distance': 82.15,
       'std_distance': 3.47
   }
   ```

### Interpreting Results

- **best_tour**: The sequence of cities to visit (0-indexed)
- **best_distance**: Total distance traveled in this tour
- **convergence_curve**: Shows how the algorithm improved over iterations
- **avg_distance**: Average best distance across all iterations
- **std_distance**: Standard deviation (stability measure)

---

## Example Usage

### Basic Usage

```python
import numpy as np
from AOA_TSP import AOA_TSP
from AGWO_TSP import AGWO_TSP

# Define distance matrix
distance_matrix = np.array([
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 15],
    [25, 30, 20, 15, 0]
])

# Run AOA
aoa = AOA_TSP(distance_matrix, population_size=30, max_iterations=500)
aoa_tour, aoa_distance, aoa_history = aoa.solve()

print(f"AOA Best Distance: {aoa_distance:.2f}")
print(f"AOA Best Tour: {aoa_tour}")

# Run AGWO
agwo = AGWO_TSP(distance_matrix, population_size=30, max_iterations=500)
agwo_tour, agwo_distance, agwo_history = agwo.solve()

print(f"AGWO Best Distance: {agwo_distance:.2f}")
print(f"AGWO Best Tour: {agwo_tour}")
```

### Loading from File and Running

```python
import numpy as np
from AOA_TSP import AOA_TSP

# Load TSP instance
distance_matrix = np.load('test_data/tsp_20cities.npy')

# Create algorithm instance
algorithm = AOA_TSP(
    distance_matrix=distance_matrix,
    population_size=40,
    max_iterations=1000,
    random_seed=42  # For reproducibility
)

# Solve
best_tour, best_distance, history = algorithm.solve()

# Extract results
print(f"Best tour found: {best_tour}")
print(f"Total distance: {best_distance:.2f}")
print(f"Convergence (mean): {history['avg_distance']:.2f} ± {history['std_distance']:.2f}")
```

### Multiple Runs for Statistical Analysis

```python
import numpy as np
from AOA_TSP import AOA_TSP

distance_matrix = np.load('test_data/tsp_10cities.npy')

results = []
for run in range(10):
    algorithm = AOA_TSP(distance_matrix, population_size=30, max_iterations=500)
    tour, distance, history = algorithm.solve()
    results.append(distance)

print(f"Mean Distance: {np.mean(results):.2f}")
print(f"Std Dev: {np.std(results):.2f}")
print(f"Best: {np.min(results):.2f}")
print(f"Worst: {np.max(results):.2f}")
```

---

## Benchmarking Both Algorithms

### Using benchmark_comparison.py

```bash
python benchmark_comparison.py
```

This script:
1. Loads multiple TSP instances from `test_data/`
2. Runs both AOA and AGWO 10 times each
3. Calculates statistics for each algorithm
4. Generates comparison tables
5. Creates visualization plots (if matplotlib available)
6. Exports results to CSV

**Output Files:**
- `results_aoa.csv`: AOA performance on each instance
- `results_agwo.csv`: AGWO performance on each instance
- `comparison_summary.csv`: Head-to-head comparison statistics
- `convergence_plot.png`: Convergence curves comparison

### Custom Benchmarking

```python
import numpy as np
import pandas as pd
from AOA_TSP import AOA_TSP
from AGWO_TSP import AGWO_TSP

# Load test instance
dist_matrix = np.load('test_data/tsp_20cities.npy')

# Run experiments
runs = 20
aoa_results = []
agwo_results = []

for i in range(runs):
    # AOA
    aoa = AOA_TSP(dist_matrix, population_size=30, max_iterations=500)
    _, aoa_dist, _ = aoa.solve()
    aoa_results.append(aoa_dist)
    
    # AGWO
    agwo = AGWO_TSP(dist_matrix, population_size=30, max_iterations=500)
    _, agwo_dist, _ = agwo.solve()
    agwo_results.append(agwo_dist)

# Analyze results
print("AOA Results:")
print(f"  Mean: {np.mean(aoa_results):.2f}")
print(f"  Std:  {np.std(aoa_results):.2f}")
print(f"  Min:  {np.min(aoa_results):.2f}")
print(f"  Max:  {np.max(aoa_results):.2f}")

print("\nAGWO Results:")
print(f"  Mean: {np.mean(agwo_results):.2f}")
print(f"  Std:  {np.std(agwo_results):.2f}")
print(f"  Min:  {np.min(agwo_results):.2f}")
print(f"  Max:  {np.max(agwo_results):.2f}")

# Statistical comparison
print(f"\nDifference in means: {abs(np.mean(aoa_results) - np.mean(agwo_results)):.2f}")
```

---

## Customization & Parameters

### AOA Parameters

When initializing AOA_TSP:

```python
algorithm = AOA_TSP(
    distance_matrix=dist_matrix,
    population_size=30,        # Number of solutions (higher = more thorough)
    max_iterations=500,        # More iterations = longer runtime but better convergence
    alpha=5.0,                 # Adaptivity parameter (5-10 recommended)
    epsilon=1e-10,             # Small constant for numerical stability
    random_seed=42             # For reproducibility
)
```

**Parameter Guidance:**
- `population_size`: 20-50 for small problems (n<50), 50-100 for medium (50-200)
- `max_iterations`: 200-500 for testing, 1000-5000 for serious optimization
- `alpha`: Lower values (3-5) = sharper transition, Higher (8-10) = smoother

### AGWO Parameters

```python
algorithm = AGWO_TSP(
    distance_matrix=dist_matrix,
    population_size=30,        # Number of wolves
    max_iterations=500,        # Iterations
    gamma=2.5,                 # Adaptivity exponent (2-3 recommended)
    diversity_threshold=0.1,   # Trigger diversity recovery below this
    epsilon=1e-10,             # Numerical stability
    random_seed=42             # Reproducibility
)
```

**Parameter Guidance:**
- `gamma`: 2-3 for smooth transition, <1 for sharp
- `diversity_threshold`: 0.05-0.2 (lower = more adaptive)
- `population_size`: Same as AOA recommendations

---

## Troubleshooting

### Common Issues

**Issue 1: "ModuleNotFoundError: No module named 'numpy'"**
```bash
# Solution: Install numpy
pip install numpy
```

**Issue 2: Convergence seems slow**
```python
# Solution: Increase iterations or population
algorithm = AOA_TSP(dist_matrix, population_size=50, max_iterations=1000)
```

**Issue 3: Algorithm gets stuck at local optimum**
```python
# For AGWO: Lower diversity threshold to trigger more perturbations
algorithm = AGWO_TSP(dist_matrix, diversity_threshold=0.05)

# For AOA: Use larger population
algorithm = AOA_TSP(dist_matrix, population_size=50)
```

**Issue 4: "ValueError: operands could not be broadcast together"**
```python
# Solution: Ensure distance matrix is 2D and square
assert dist_matrix.ndim == 2
assert dist_matrix.shape[0] == dist_matrix.shape[1]
```

**Issue 5: Results vary greatly between runs**
```python
# Solution: Run multiple times and take statistics
results = []
for i in range(10):
    _, distance, _ = algorithm.solve()
    results.append(distance)
print(f"Average: {np.mean(results):.2f} ± {np.std(results):.2f}")
```

### Performance Optimization

For large TSP instances (n > 100):
1. **Increase population size** (helps exploration)
2. **Increase iterations** (helps convergence)
3. **Use multiple runs** (find best solution)
4. **Consider hybrid approaches** (combine with local search)

```python
# Example for large instance
from AOA_TSP import AOA_TSP

dist_matrix = np.load('large_tsp_100cities.npy')

algorithm = AOA_TSP(
    distance_matrix=dist_matrix,
    population_size=100,        # Large population
    max_iterations=2000,        # Many iterations
    random_seed=42
)

# Run multiple times and keep best
best_solutions = []
for _ in range(5):
    tour, distance, _ = algorithm.solve()
    best_solutions.append(distance)

print(f"Best solution found: {min(best_solutions):.2f}")
```

---

## Key Concepts in Implementation

### Largest Order Value (LOV) Permutation
- Converts continuous values [0, 1) to discrete city permutations
- Ensures each city appears exactly once
- Example: [0.7, 0.3, 0.9] → argsort → [1, 0, 2] → Tour: [City1, City0, City2]

### Adaptive Parameters
- **AOA**: MOP and MOR control exploration vs exploitation dynamically
- **AGWO**: Parameter 'a' decreases smoothly, diversity monitoring prevents stagnation

### Fitness Function
- Total tour distance: sum of distances between consecutive cities
- Goal: Minimize this value

---

## References

1. Abualigah, L., et al. (2021). "Arithmetic Optimization Algorithm: Architecture, estimation perspectives and practical applications." *Journal of Computational Design and Engineering*, 8(5), 1228-1255.

2. Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). "Grey wolf optimizer." *Advances in Engineering Software*, 69, 46-61.

3. Lawler, E. L., et al. (1985). *The Traveling Salesman Problem: A Guided Tour of Combinatorial Optimization*. John Wiley & Sons.

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{tsp_aoa_agwo_2024,
  author = {[Oswalds]},
  title = {TSP Solver: Arithmetic Optimization Algorithm vs Adaptive Grey Wolf Optimization},
  year = {2026},}
```

---

## Support & Questions

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review **TSP_Mathematical_Formulation.md** for algorithm details
3. Check code comments for implementation details
4. Verify input format using examples

---


**Last Updated:** 2026
**Version:** 1.0
