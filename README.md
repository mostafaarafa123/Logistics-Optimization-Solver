#  Advanced Logistics Solver (ALNS)

This project is a high-performance solver designed for the Robin Logistics Environment. It implements state-of-the-art metaheuristic algorithms to solve the Vehicle Routing Problem (VRP) with complex inventory constraints.

##  Evolution of the Solver
The solver underwent 12 phases of development to achieve optimal efficiency:
- **Phase 1-2:** Basic BFS pathfinding and single-order fulfillment.
- **Phase 7:** Implemented robust inventory rollback logic for routing failure.
- **Phase 9:** Introduced Smart Batching using Nearest Neighbor (NN) clustering.
- **Phase 11-12 (Final):** Achieved global optimality using **Adaptive Large Neighborhood Search (ALNS)** and **Simulated Annealing (SA)**.

##  Key Algorithmic Features
- **Metaheuristic Core:** Uses an ALNS "Destroy and Repair" loop to escape local optima.
- **Performance:** Implemented an All-Pairs Shortest Path (APSP) cache for $O(1)$ distance lookups.
- **Optimization:** Utilizes Simulated Annealing for final route refinement.
- **Constraint Management:** Handles complex split-inventory fulfillment across multiple warehouses.

##  Tech Stack
- **Language:** Python
- **Algorithms:** ALNS, Simulated Annealing, Breadth-First Search (BFS), Nearest Neighbor.
- **Optimization:** Cost minimization and order fulfillment maximization.

##  How to Run
1. Install the required environment:
   ```bash
   pip install robin-logistics-env
