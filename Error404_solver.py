
"""
Contestant solver for the Robin Logistics Environment.
PHASE 12: The Root Solution (Adaptive Large Neighborhood Search - ALNS)

This solver uses a state-of-the-art metaheuristic to find a globally
optimal solution. It abandons the simple "Batch-then-Route" design and
instead operates on the *entire solution* at once.

It works in a "Destroy and Repair" loop:
1.  DESTROY: A random set of orders are removed from the solution.
2.  REPAIR:  The removed orders are re-inserted using a smart, low-cost heuristic.
3.  ACCEPT:  The new solution is accepted based on Simulated Annealing logic.

This process allows the solver to escape "local optima" (good solutions)
and find the "global optimum" (the best possible solution).
"""

import math
import random
import copy
from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------
# 1. APSP (ALL-PAIRS SHORTEST PATH) CACHE (FOR O(1) SPEED)
# -----------------------------------------------------------------

# Global cache: all_paths_cache[start][end] = (distance, [path_list])
all_paths_cache: Dict[int, Dict[int, Tuple[int, List[int]]]] = {}

def pre_calculate_all_paths(adj_list: Dict[int, List[int]]):
    """
    Fills the global `all_paths_cache` with all-pairs shortest paths using BFS
    from every single node. This is non-negotiable for high-speed solvers.
    """
    global all_paths_cache
    all_paths_cache = {}
    all_nodes = list(adj_list.keys())
    
    print(f"Pre-calculating all-pairs shortest paths for {len(all_nodes)} nodes...")

    for start_node in all_nodes:
        all_paths_cache[start_node] = {}
        queue = [(start_node, [start_node])]
        visited = {start_node}
        all_paths_cache[start_node][start_node] = (0, [start_node])

        while queue:
            (current_node, path) = queue.pop(0)
            if current_node not in adj_list: continue
            for neighbor in adj_list[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    distance = len(new_path) - 1
                    all_paths_cache[start_node][neighbor] = (distance, new_path)
                    queue.append((neighbor, new_path))
    print("... Pre-calculation complete.")

def get_dist_from_cache(start_node: int, end_node: int) -> float:
    """Gets a distance from the pre-computed cache. O(1) lookup."""
    try:
        return float(all_paths_cache[start_node][end_node][0])
    except KeyError:
        return float('inf')

def get_path_from_cache(start_node: int, end_node: int) -> Optional[List[int]]:
    """Gets a path list from the pre-computed cache. O(1) lookup."""
    try:
        return all_paths_cache[start_node][end_node][1]
    except KeyError:
        return None

# -----------------------------------------------------------------
# 2. ROUTE OPTIMIZATION HELPERS (NN & SA)
# -----------------------------------------------------------------

def get_tour_cost(tour: List[int]) -> float:
    """Calculates the total distance (cost) of a given tour list."""
    if not tour: return 0.0
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += get_dist_from_cache(tour[i], tour[i+1])
    return cost

def build_fast_nn_tour(start_node: int, stops: Set[int], end_node: int) -> List[int]:
    """Builds a good-enough tour using the fast Nearest Neighbor heuristic."""
    if not stops:
        return [start_node, end_node] if start_node != end_node else [start_node]
        
    tour = [start_node]
    current_node = start_node
    stops_to_visit = stops.copy()
    
    while stops_to_visit:
        # Find the nearest node from the *cache*
        min_dist = float('inf')
        best_target = None
        for target in stops_to_visit:
            dist = get_dist_from_cache(current_node, target)
            if dist < min_dist:
                min_dist = dist
                best_target = target
        
        if best_target is None:
            break # No reachable nodes left
            
        tour.append(best_target)
        current_node = best_target
        stops_to_visit.remove(best_target)
    
    tour.append(end_node)
    return tour


def optimize_route_simulated_annealing(tour: List[int]) -> List[int]:
    """
    Optimizes a *complete* tour (e.g., [home, A, B, C, home]) using SA.
    This is the "high-quality" optimizer, used only on the final solution.
    """
    if len(tour) < 4: # Not enough stops to optimize
        return tour

    # 1. Parameters
    initial_temp = 100.0
    min_temp = 0.1
    cooling_rate = 0.995 # Slow and steady
    
    # 2. Initial state
    current_solution = tour
    current_cost = get_tour_cost(current_solution)
    best_solution = current_solution[:]
    best_cost = current_cost
    temp = initial_temp

    while temp > min_temp:
        # 3. Create a neighbor solution (2-Opt swap)
        i, k = random.sample(range(1, len(current_solution) - 1), 2)
        if i > k: i, k = k, i
        
        new_solution = current_solution[:i]
        new_solution.extend(reversed(current_solution[i:k+1]))
        new_solution.extend(current_solution[k+1:])
        
        new_cost = get_tour_cost(new_solution)
        cost_diff = new_cost - current_cost

        # 4. Acceptance Logic
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
            current_solution = new_solution
            current_cost = new_cost
            if new_cost < best_cost:
                best_solution = new_solution[:]
                best_cost = new_cost
        
        temp *= cooling_rate

    return best_solution

# -----------------------------------------------------------------
# 3. ALNS STATE & HELPER FUNCTIONS
# -----------------------------------------------------------------

# The "Solution State" is a simple dictionary:
# state = {
#     "vehicle_routes": {
#         "vehicle_id_1": ["order_id_A", "order_id_B"],
#         "vehicle_id_2": ["order_id_C"],
#         ...
#     },
#     "unassigned_orders": {"order_id_D", "order_id_E"}
# }
#
# We also need a global cache for order details to avoid re-calculating.
order_details_cache = {}

def get_order_details(env, order_id: str, local_inventory: Dict) -> Optional[Dict]:
    """
    Gets all necessary info for an order (weight, volume, required pickups).
    Implements the 100% fulfillment Split-Inventory logic.
    Caches the result to avoid re-calculation.
    """
    if order_id in order_details_cache:
        return order_details_cache[order_id]

    try:
        requirements = env.get_order_requirements(order_id)
        dest_node = env.get_order_location(order_id)
        
        order_weight = 0.0
        order_volume = 0.0
        pickups = [] # List of (wh_id, wh_node, sku, quantity)
        
        for sku_id, quantity_needed in requirements.items():
            sku_details = env.get_sku_details(sku_id)
            sku_weight, sku_volume = sku_details['weight'], sku_details['volume']
            quantity_found = 0

            # Split-Inventory Logic
            for wh_id, inventory in local_inventory.items():
                if quantity_found == quantity_needed: break
                available_in_wh = inventory.get(sku_id, 0)
                
                if available_in_wh > 0:
                    quantity_to_take = min(available_in_wh, quantity_needed - quantity_found)
                    if quantity_to_take == 0: continue
                    
                    wh_node = env.get_warehouse_by_id(wh_id).location.id
                    pickups.append((wh_id, wh_node, sku_id, quantity_to_take))
                    
                    quantity_found += quantity_to_take
                    order_weight += sku_weight * quantity_to_take
                    order_volume += sku_volume * quantity_to_take

            if quantity_found < quantity_needed:
                return None # Order is unfulfillable

        details = {
            "id": order_id,
            "weight": order_weight,
            "volume": order_volume,
            "destination_node": dest_node,
            "pickups": pickups # (wh_id, wh_node, sku, qty)
        }
        order_details_cache[order_id] = details
        return details

    except Exception:
        return None

def build_initial_solution(env, all_order_ids: Set[str], vehicle_ids: List[str], local_inventory: Dict) -> Dict:
    """
    Creates a simple, greedy initial solution.
    It assigns orders to the first vehicle that has capacity.
    This is a "good enough" starting point for ALNS.
    """
    state = {
        "vehicle_routes": {v_id: [] for v_id in vehicle_ids},
        "unassigned_orders": set()
    }
    
    # Store vehicle capacities
    vehicle_caps = {}
    for v_id in vehicle_ids:
        v = env.get_vehicle_by_id(v_id)
        vehicle_caps[v_id] = {"max_w": v.capacity_weight, "max_v": v.capacity_volume, "cur_w": 0.0, "cur_v": 0.0}

    print("Building initial greedy solution...")
    for order_id in all_order_ids:
        details = get_order_details(env, order_id, local_inventory)
        if not details:
            state["unassigned_orders"].add(order_id)
            continue

        order_w, order_v = details["weight"], details["volume"]
        assigned = False

        for v_id in vehicle_ids:
            caps = vehicle_caps[v_id]
            if (caps["cur_w"] + order_w <= caps["max_w"] and
                caps["cur_v"] + order_v <= caps["max_v"]):
                
                # Assign to this vehicle
                state["vehicle_routes"][v_id].append(order_id)
                caps["cur_w"] += order_w
                caps["cur_v"] += order_v
                assigned = True
                break
        
        if not assigned:
            state["unassigned_orders"].add(order_id)
    
    print(f"Initial solution built. Unassigned: {len(state['unassigned_orders'])}")
    return state


def calculate_solution_cost(env, state: Dict) -> float:
    """
    This is the "fitness function". It calculates the total cost of a given state.
    It uses the FAST NN-tour heuristic, not the slow SA optimizer.
    This function is called thousands of times, so it *must* be fast.
    """
    total_cost = 0.0
    
    # Add a massive penalty for any unassigned orders
    total_cost += len(state["unassigned_orders"]) * 1000000.0 

    for v_id, order_list in state["vehicle_routes"].items():
        if not order_list:
            continue
            
        vehicle = env.get_vehicle_by_id(v_id)
        home_node = env.get_vehicle_home_warehouse(v_id)
        
        stops_to_visit = set()
        current_w, current_v = 0.0, 0.0
        
        for order_id in order_list:
            details = order_details_cache[order_id]
            current_w += details["weight"]
            current_v += details["volume"]
            stops_to_visit.add(details["destination_node"])
            for (wh_id, wh_node, sku, qty) in details["pickups"]:
                stops_to_visit.add(wh_node)
        
        # Add a massive penalty for violating capacity
        if current_w > vehicle.capacity_weight or current_v > vehicle.capacity_volume:
            total_cost += 1000000.0 # Invalid route
            
        # Build the FAST NN tour and get its cost
        nn_tour = build_fast_nn_tour(home_node, stops_to_visit, home_node)
        total_cost += get_tour_cost(nn_tour)
        
    return total_cost


def get_vehicle_load(v_id: str, order_list: List[str]) -> Tuple[float, float]:
    """Helper to get the current weight and volume of a vehicle's route."""
    cur_w, cur_v = 0.0, 0.0
    for order_id in order_list:
        details = order_details_cache[order_id]
        cur_w += details["weight"]
        cur_v += details["volume"]
    return cur_w, cur_v

# -----------------------------------------------------------------
# 4. ALNS DESTROY & REPAIR OPERATORS
# -----------------------------------------------------------------

def destroy_random_orders(state: Dict, num_to_remove: int) -> Tuple[Dict, List[str]]:
    """
    DESTROY Operator: Randomly selects 'num_to_remove' orders from the
    entire solution and moves them to the unassigned list.
    """
    # 1. Create a new state (deep copy)
    new_state = copy.deepcopy(state)
    
    # 2. Get a flat list of all currently assigned orders
    all_assigned_orders = []
    for v_id, order_list in new_state["vehicle_routes"].items():
        all_assigned_orders.extend(order_list)
        
    if not all_assigned_orders:
        return new_state, []

    # 3. Choose which orders to remove
    num_to_remove = min(num_to_remove, len(all_assigned_orders))
    orders_to_remove = set(random.sample(all_assigned_orders, num_to_remove))
    
    # 4. Remove them from vehicles and add to unassigned
    for v_id in new_state["vehicle_routes"]:
        new_state["vehicle_routes"][v_id] = [o for o in new_state["vehicle_routes"][v_id] if o not in orders_to_remove]
        
    new_state["unassigned_orders"].update(orders_to_remove)
    
    return new_state, list(orders_to_remove)


def repair_greedy_insertion(env, state: Dict) -> Dict:
    """
    REPAIR Operator: Tries to re-insert all unassigned orders using
    a "best-fit" (cheapest insertion) heuristic.
    """
    
    if not state["unassigned_orders"]:
        return state
        
    # Get vehicle capacities
    vehicle_caps = {}
    for v_id in state["vehicle_routes"]:
        v = env.get_vehicle_by_id(v_id)
        vehicle_caps[v_id] = {"max_w": v.capacity_weight, "max_v": v.capacity_volume}

    # Try to insert each unassigned order
    # We shuffle to add randomness
    orders_to_insert = list(state["unassigned_orders"])
    random.shuffle(orders_to_insert)
    
    for order_id in orders_to_insert:
        order_details = order_details_cache[order_id]
        if not order_details: continue

        best_vehicle_id = None
        best_cost_delta = float('inf')

        # Find the cheapest vehicle to insert this order into
        for v_id, order_list in state["vehicle_routes"].items():
            
            # a) Check capacity
            cur_w, cur_v = get_vehicle_load(v_id, order_list)
            if (cur_w + order_details["weight"] > vehicle_caps[v_id]["max_w"] or
                cur_v + order_details["volume"] > vehicle_caps[v_id]["max_v"]):
                continue # This vehicle doesn't have capacity

            # b) Calculate cost delta
            # Get cost *before* adding the order
            stops_before = set()
            for oid in order_list:
                stops_before.add(order_details_cache[oid]["destination_node"])
                for p in order_details_cache[oid]["pickups"]: stops_before.add(p[1])
            home_node = env.get_vehicle_home_warehouse(v_id)
            tour_before = build_fast_nn_tour(home_node, stops_before, home_node)
            cost_before = get_tour_cost(tour_before)

            # Get cost *after* adding the order
            stops_after = stops_before.copy()
            stops_after.add(order_details["destination_node"])
            for p in order_details["pickups"]: stops_after.add(p[1])
            tour_after = build_fast_nn_tour(home_node, stops_after, home_node)
            cost_after = get_tour_cost(tour_after)
            
            cost_delta = cost_after - cost_before

            if cost_delta < best_cost_delta:
                best_cost_delta = cost_delta
                best_vehicle_id = v_id
        
        # c) Perform the best insertion
        if best_vehicle_id is not None:
            state["vehicle_routes"][best_vehicle_id].append(order_id)
            state["unassigned_orders"].remove(order_id)
            
    return state

# -----------------------------------------------------------------
# 5. FINAL SOLUTION BUILDER
# -----------------------------------------------------------------

def build_final_solution_format(env, best_state: Dict) -> Dict:
    """
    Converts the optimized ALNS state into the final JSON format
    required by the environment.
    This is where we run the HIGH-QUALITY SA optimizer one last time
    on the final set of routes.
    """
    solution = {"routes": []}
    
    print("\nBuilding final solution from best state found...")
    
    for v_id, order_list in best_state["vehicle_routes"].items():
        if not order_list:
            continue
            
        print(f"  - Optimizing final route for vehicle '{v_id}'...")
        home_node = env.get_vehicle_home_warehouse(v_id)
        
        # 1. Get all stops (pickups and deliveries) for this route
        pickups_by_wh_node = {}
        deliveries_by_dest_node = {}
        all_stops_nodes = set()
        
        for order_id in order_list:
            details = order_details_cache[order_id]
            dest_node = details["destination_node"]
            all_stops_nodes.add(dest_node)
            
            # Group deliveries by destination
            if dest_node not in deliveries_by_dest_node:
                deliveries_by_dest_node[dest_node] = []
            order_reqs = env.get_order_requirements(order_id)
            for sku_id, qty in order_reqs.items():
                deliveries_by_dest_node[dest_node].append(
                    {'order_id': order_id, 'sku_id': sku_id, 'quantity': qty}
                )
            
            # Group pickups by warehouse
            for (wh_id, wh_node, sku, qty) in details["pickups"]:
                all_stops_nodes.add(wh_node)
                if wh_node not in pickups_by_wh_node:
                    pickups_by_wh_node[wh_node] = []
                pickups_by_wh_node[wh_node].append(
                    {'warehouse_id': wh_id, 'sku_id': sku, 'quantity': qty}
                )

        # 2. Build the HIGH-QUALITY tour
        # We optimize pickups and deliveries separately
        
        # Phase 2a: Optimize Pickup Tour
        pickup_stops = set(pickups_by_wh_node.keys())
        nn_pickup_tour = build_fast_nn_tour(home_node, pickup_stops, home_node)[1:-1] # Get stops, remove home
        optimized_pickup_tour = optimize_route_simulated_annealing([home_node] + nn_pickup_tour + [home_node])[1:-1]
        
        # Phase 2b: Optimize Delivery Tour
        last_pickup_node = optimized_pickup_tour[-1] if optimized_pickup_tour else home_node
        delivery_stops = set(deliveries_by_dest_node.keys())
        nn_delivery_tour = build_fast_nn_tour(last_pickup_node, delivery_stops, home_node)[1:-1]
        optimized_delivery_tour = optimize_route_simulated_annealing([last_pickup_node] + nn_delivery_tour + [home_node])[1:-1]
        
        # 3. Build the final `steps` list
        all_steps = [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
        current_node = home_node
        
        # Add pickup steps
        for wh_node in optimized_pickup_tour:
            path = get_path_from_cache(current_node, wh_node)
            for node in path[1:]: all_steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
            current_node = wh_node
            all_steps[-1]['pickups'].extend(pickups_by_wh_node[wh_node])
        
        # Add delivery steps
        for dest_node in optimized_delivery_tour:
            path = get_path_from_cache(current_node, dest_node)
            for node in path[1:]: all_steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
            current_node = dest_node
            all_steps[-1]['deliveries'].extend(deliveries_by_dest_node[dest_node])
            
        # Add return-to-home steps
        if current_node != home_node:
            path = get_path_from_cache(current_node, home_node)
            for node in path[1:]: all_steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
        
        solution["routes"].append({'vehicle_id': v_id, 'steps': all_steps})
    
    return solution

# -----------------------------------------------------------------
# 6. MAIN SOLVER FUNCTION (ALNS)
# -----------------------------------------------------------------

def my_solver(env) -> Dict:
    """
    Main ALNS solver function.
    """
    global all_paths_cache, order_details_cache
    all_paths_cache = {}
    order_details_cache = {}

    print("---  Starting Phase 12 Solver (ALNS) ---")

    # 1. Get all static data
    try:
        road_network = env.get_road_network_data()
        pre_calculate_all_paths(road_network.get("adjacency_list", {}))
    except Exception as e:
        print(f"ERROR during pre-computation: {e}")
        return {"routes": []}

    all_order_ids = set(env.get_all_order_ids())
    available_vehicles = env.get_available_vehicles()
    
    # Pre-cache all order details (handles split-inventory)
    local_inventory = {}
    for wh_id, warehouse in env.warehouses.items():
        local_inventory[wh_id] = warehouse.inventory.copy()
    
    print("Pre-caching all order details...")
    fulfillable_orders = set()
    for order_id in all_order_ids:
        if get_order_details(env, order_id, local_inventory):
            fulfillable_orders.add(order_id)
        else:
            print(f"Warning: Order '{order_id}' is unfulfillable. Not enough inventory.")

    # 2. ALNS Parameters
    # (Tune these for performance vs. quality)
    ITERATIONS = 5000     # 5,000 to 10,000 is a good start
    INITIAL_TEMP = 100.0
    MIN_TEMP = 0.1
    COOLING_RATE = 0.999   # Slower cooling for better exploration
    DESTROY_SIZE = 5       # How many orders to remove per iteration

    # 3. Build Initial Solution
    S_best = build_initial_solution(env, fulfillable_orders, available_vehicles, local_inventory)
    S_current = copy.deepcopy(S_best)
    
    best_cost = calculate_solution_cost(env, S_best)
    current_cost = best_cost
    
    print(f"Initial Solution Cost: {best_cost:.2f}")

    # 4. Start ALNS Optimization Loop
    temp = INITIAL_TEMP
    for i in range(ITERATIONS):
        
        # a) Create a new solution (Destroy + Repair)
        S_prime, removed = destroy_random_orders(copy.deepcopy(S_current), DESTROY_SIZE)
        S_new = repair_greedy_insertion(env, S_prime)
        new_cost = calculate_solution_cost(env, S_new)
        
        # b) Acceptance Criteria (SA)
        cost_diff = new_cost - current_cost
        
        if cost_diff < 0:
            # Better solution, always accept
            S_current = copy.deepcopy(S_new)
            current_cost = new_cost
            
            if new_cost < best_cost:
                S_best = copy.deepcopy(S_new)
                best_cost = new_cost
                print(f"  > Iter {i+1}: New Best Found! Cost: {best_cost:.2f}")
                
        elif random.random() < math.exp(-cost_diff / temp):
            # Worse solution, accept based on probability
            S_current = copy.deepcopy(S_new)
            current_cost = new_cost
            
        # c) Cool down
        temp = max(temp * COOLING_RATE, MIN_TEMP)
        
        if i % 500 == 0 and i > 0:
            print(f"  - Iter {i+1}: Current Cost: {current_cost:.2f} (Best: {best_cost:.2f})")

    # 5. Build Final Solution
    print(f"\nALNS loop finished. Best cost found: {best_cost:.2f}")
    
    # Run the final high-quality formatting and SA optimization
    final_solution = build_final_solution_format(env, S_best)

    unassigned = len(S_best["unassigned_orders"])
    print(f"\n---  Phase 12 Finished ---")
    print(f"Total routes created: {len(final_solution['routes'])}")
    print(f"Orders left unassigned: {unassigned}")

    return final_solution

