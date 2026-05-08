# TSP Optimization: AOA vs. AGWO Comparison

This repository contains a comprehensive implementation and comparative analysis of the **Arithmetic Optimization Algorithm (AOA)** and the **Adaptive Grey Wolf Optimizer (AGWO)** applied to the **Traveling Salesman Problem (TSP)**. The study evaluates how nature-inspired leadership hierarchies compare against mathematical operator-based exploration using the `berlin52` benchmark.

## 👥 Project Group: Oswalds
* **Institution:** University of Doha for Science and Technology
* **Course:** DSAI3203 - Fundamentals of AI
* **Instructor:** Dr. Somaiyeh M. Zadeh
* **Team Members:** Zahra Alwaili, Arlene Riona Devasahayarajan, Fariha Aslam Mahaldar, Syeda Maymona Mustafa, Mais Mardoum

---

## 🚀 Overview
The Traveling Salesman Problem is a classical NP-hard combinatorial optimization problem where the objective is to determine the shortest possible route that visits each city exactly once and returns to the starting point. This project compares two metaheuristic approaches:

* **Adaptive Grey Wolf Optimizer (AGWO):** An enhanced version of GWO inspired by the hierarchical leadership and cooperative hunting strategy of grey wolves. It uses an alpha, beta, and delta leadership structure to guide the search.
* **Arithmetic Optimization Algorithm (AOA):** A population-based optimization method inspired by basic arithmetic operations such as Addition, Subtraction, Multiplication, and Division to explore and exploit the search space.

---

## 📊 Experimental Results (berlin52)
Both algorithms were tested over 10 independent runs with 30 agents over 500 iterations, followed by a **2-opt local search** refinement.

| Metric | AOA | AGWO |
| :--- | :--- | :--- |
| **Best Distance** | 7,742.6 | 8,001.9 |
| **Avg. Distance** | 8,351.6 | 8,271.1 |
| **Std. Dev ($\sigma$)** | 250.4 | 201.4 |
| **Optimality Gap** | 2.66% | 6.10% |
| **Avg. Runtime (s)** | 7.87 | 11.92 |

*Note: The known optimal distance for the `berlin52` dataset is **7,542**.*

---

## 🛠️ Technical Stack 
* **Optimization:** Custom implementation of AOA and AGWO adapted for discrete permutation-based spaces using the **Largest Order Value (LOV)** method.
* **Local Search:** Integrated 2-opt heuristic applied at termination to eliminate path crossings and refine the final tour.

---

## 📈 Key Conclusions
1.  **AOA (Peak Performance):** Achieved the absolute lowest distance (2.66% optimality gap) but showed higher volatility between runs.
2.  **AGWO (Consistency):** Demonstrated more reliable results across all iterations with a better average distance and lower standard deviation.
3.  **Future Work:** Development of a **Hybrid AOA-GWO** framework to combine AOA's aggressive global exploration with GWO's stable local hunting mechanisms.

---
