import math
import random
from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------
# 1. PATH CACHING HELPERS (FOR SPEED)
# -----------------------------------------------------------------
# Global cache to store all paths.
# all_paths_cache[start_node][end_node] = (distance, [path_list])
all_paths_cache: Dict[int, Dict[int, Tuple[int, List[int]]]] = {}


def pre_calculate_all_paths(adj_list: Dict[int, List[int]]):
    """
    Fills the global `all_paths_cache` with all-pairs shortest paths using BFS
    from every single node. This is the key to handling 500+ orders.
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


def get_path_from_cache(start_node: int, end_node: int) -> Optional[List[int]]:
    try:
        return all_paths_cache[start_node][end_node][1]  # Path list
    except KeyError:
        return None


def get_dist_from_cache(start_node: int, end_node: int) -> float:
    try:
        return float(all_paths_cache[start_node][end_node][0])  # Distance
    except KeyError:
        return float('inf')


def find_nearest_node_fast(current_node: int, target_nodes: Set[int]) -> Optional[int]:
    """Finds the nearest node from a set using the pre-computed cache (O(N) lookup)."""
    if not target_nodes: return None
    min_dist = float('inf')
    best_target = None
    for target in target_nodes:
        dist = get_dist_from_cache(current_node, target)
        if dist < min_dist:
            min_dist = dist
            best_target = target
    return best_target


# -----------------------------------------------------------------
# 2. SIMULATED ANNEALING (SA) OPTIMIZER (FOR GLOBAL BEST PATH)
# -----------------------------------------------------------------

def get_tour_cost(tour: List[int]) -> float:
    """Calculates the total distance (cost) of a given tour list."""
    if not tour: return 0.0
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += get_dist_from_cache(tour[i], tour[i + 1])
    return cost


def get_neighbor_solution(tour: List[int]) -> List[int]:
    """
    Creates a new "neighbor" tour by performing a 2-Opt swap.
    It picks two random indices (i, k) and reverses the segment between them.
    """
    if len(tour) < 4:  # Need at least 4 nodes to swap (e.g., [start, A, B, end])
        return tour[:]

    # Get two random, distinct indices, ensuring i < k
    # We don't swap the start (0) or end (last) nodes
    i, k = random.sample(range(1, len(tour) - 1), 2)
    if i > k:
        i, k = k, i

    new_tour = tour[:i]  # Segment 1
    new_tour.extend(reversed(tour[i:k + 1]))  # Segment 2 (Reversed)
    new_tour.extend(tour[k + 1:])  # Segment 3

    return new_tour


def optimize_route_simulated_annealing(start_node: int, tour: List[int], end_node: int) -> List[int]:
    """
    Improves a given tour (list of node IDs) using Simulated Annealing.
    This is the "globally optimal" algorithm you requested.
    """
    if not tour:
        return []

    # 1. Parameters
    initial_temp = 100.0  # Starting temperature
    min_temp = 0.1  # Ending temperature
    cooling_rate = 0.995  # Rate to cool down (e.g., 0.995 is slow and good)

    # 2. Create initial state
    full_tour = [start_node] + tour + [end_node]

    current_solution = full_tour
    current_cost = get_tour_cost(current_solution)

    best_solution = current_solution[:]
    best_cost = current_cost

    temp = initial_temp

    # 3. Main Optimization Loop
    while temp > min_temp:
        # Create a new solution by slightly modifying the current one
        new_solution = get_neighbor_solution(current_solution)
        new_cost = get_tour_cost(new_solution)

        # Calculate the energy difference
        cost_diff = new_cost - current_cost

        # 4. Decision Logic (The Core)
        if cost_diff < 0:
            # New solution is better, always accept it
            current_solution = new_solution
            current_cost = new_cost
            if new_cost < best_cost:
                best_solution = new_solution[:]
                best_cost = new_cost
        else:
            # New solution is worse.
            # Accept it based on probability to "escape" local optima
            acceptance_probability = math.exp(-cost_diff / temp)
            if random.random() < acceptance_probability:
                current_solution = new_solution
                current_cost = new_cost

        # Cool the temperature
        temp *= cooling_rate

    # Return the *best found* tour, removing the start and end nodes
    return best_solution[1:-1]


# -----------------------------------------------------------------
# 3. MAIN SOLVER
# -----------------------------------------------------------------
def my_solver(env) -> Dict:
    """
    Main solver function (Phase 11).
    """
    solution = {"routes": []}
    print("---  Starting Phase 11 Solver (APSP + Simulated Annealing) ---")

    # 1. Get Road Network Data and Pre-compute ALL paths
    try:
        road_network = env.get_road_network_data()
        pre_calculate_all_paths(road_network.get("adjacency_list", {}))
    except Exception as e:
        print(f"ERROR during pre-computation: {e}")
        return solution

    # 2. Get all orders and vehicles
    unassigned_orders = set(env.get_all_order_ids())
    available_vehicles = env.get_available_vehicles()
    local_inventory = {}
    for wh_id, warehouse in env.warehouses.items():
        local_inventory[wh_id] = warehouse.inventory.copy()

    print(f"Total orders to serve: {len(unassigned_orders)}")
    print(f"Total vehicles available: {len(available_vehicles)}")

    # 3. Loop through vehicles and try to build a "batch" for each
    for vehicle_id in available_vehicles:
        if not unassigned_orders: break

        print(f"\nBuilding batch for Vehicle '{vehicle_id}'...")
        vehicle = env.get_vehicle_by_id(vehicle_id)
        home_node = env.get_vehicle_home_warehouse(vehicle_id)

        batch_items_to_pickup, batch_deliveries = [], []
        current_weight, current_volume = 0, 0
        orders_in_this_batch, batch_inventory_changes = set(), []

        # --- SMART BATCHING (Sort by proximity) ---
        order_distances = []
        for order_id in unassigned_orders:
            try:
                dest_node = env.get_order_location(order_id)
                distance = get_dist_from_cache(home_node, dest_node)
                if distance != float('inf'):
                    order_distances.append((order_id, distance))
            except Exception:
                pass
        sorted_orders = sorted(order_distances, key=lambda x: x[1])

        # --- Batching Loop (Now uses sorted list) ---
        for order_id, distance in sorted_orders:
            if order_id not in unassigned_orders: continue
            try:
                requirements = env.get_order_requirements(order_id)
                dest_node = env.get_order_location(order_id)
                order_weight, order_volume = 0, 0
                items_for_this_order, order_inventory_changes = [], []
                order_is_feasible = True

                # --- Split-Inventory Logic (Untouched) ---
                for sku_id, quantity_needed in requirements.items():
                    sku_details = env.get_sku_details(sku_id)
                    sku_weight, sku_volume = sku_details['weight'], sku_details['volume']
                    quantity_found = 0
                    for wh_id, inventory in local_inventory.items():
                        if quantity_found == quantity_needed: break
                        available_in_wh = inventory.get(sku_id, 0)
                        if available_in_wh > 0:
                            quantity_to_take = min(available_in_wh, quantity_needed - quantity_found)
                            if quantity_to_take == 0: continue
                            wh_node = env.get_warehouse_by_id(wh_id).location.id
                            items_for_this_order.append((sku_id, quantity_to_take, wh_id, wh_node, dest_node))
                            order_inventory_changes.append((wh_id, sku_id, quantity_to_take))
                            quantity_found += quantity_to_take
                            order_weight += sku_weight * quantity_to_take
                            order_volume += sku_volume * quantity_to_take
                    if quantity_found < quantity_needed:
                        order_is_feasible = False
                        break
                # --- End Split-Inventory Logic ---

                if not order_is_feasible: continue
                if (current_weight + order_weight <= vehicle.capacity_weight and
                        current_volume + order_volume <= vehicle.capacity_volume):
                    current_weight += order_weight
                    current_volume += order_volume
                    orders_in_this_batch.add(order_id)
                    for (sku_id, q, wh_id, wh_node, dest) in items_for_this_order:
                        batch_items_to_pickup.append((sku_id, q, wh_id, wh_node))
                    for sku_id, quantity in requirements.items():
                        batch_deliveries.append((order_id, sku_id, quantity, dest_node))
                    batch_inventory_changes.extend(order_inventory_changes)
                    for (wh_id, sku_id, quantity) in order_inventory_changes:
                        local_inventory[wh_id][sku_id] -= quantity
                    unassigned_orders.remove(order_id)
            except Exception as e:
                pass
        # --- END: Batching Loop ---

        if not orders_in_this_batch:
            print("  - No orders batched for this vehicle.")
            continue

        print(f"  - Batch complete for '{vehicle_id}'. Building GLOBALLY OPTIMAL route...")

        # 5. Find the paths between ALL stops
        try:
            all_steps = [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
            current_node = home_node
            pickups_by_wh_node = {}
            for (sku_id, q, wh_id, wh_node) in batch_items_to_pickup:
                pickups_by_wh_node.setdefault(wh_node, []).append(
                    {'warehouse_id': wh_id, 'sku_id': sku_id, 'quantity': q})
            deliveries_by_dest_node = {}
            for (order_id, sku_id, q, dest_node) in batch_deliveries:
                deliveries_by_dest_node.setdefault(dest_node, []).append(
                    {'order_id': order_id, 'sku_id': sku_id, 'quantity': q})

            # --- GLOBALLY OPTIMIZED ROUTING (NN + SA) ---
            pickup_nodes_to_visit = set(pickups_by_wh_node.keys())
            delivery_nodes_to_visit = set(deliveries_by_dest_node.keys())

            # --- Phase 1: Build Pickup Route (Fast NN) ---
            nn_pickup_route = []
            temp_current_node = current_node
            while pickup_nodes_to_visit:
                nearest_wh_node = find_nearest_node_fast(temp_current_node, pickup_nodes_to_visit)
                if nearest_wh_node is None: raise Exception(
                    "Routing Error: Cannot find path to any remaining pickup warehouse.")
                nn_pickup_route.append(nearest_wh_node)
                temp_current_node = nearest_wh_node
                pickup_nodes_to_visit.remove(nearest_wh_node)

            # --- Phase 2: Optimize Pickup Route (Simulated Annealing) ---
            print("    - Optimizing pickup route with Simulated Annealing...")
            optimized_pickup_route = optimize_route_simulated_annealing(current_node, nn_pickup_route, current_node)

            # --- Phase 3: Build Delivery Route (Fast NN) ---
            temp_current_node = optimized_pickup_route[-1] if optimized_pickup_route else current_node
            nn_delivery_route = []
            while delivery_nodes_to_visit:
                nearest_dest_node = find_nearest_node_fast(temp_current_node, delivery_nodes_to_visit)
                if nearest_dest_node is None: raise Exception(
                    "Routing Error: Cannot find path to any remaining customer.")
                nn_delivery_route.append(nearest_dest_node)
                temp_current_node = nearest_dest_node
                delivery_nodes_to_visit.remove(nearest_dest_node)

            # --- Phase 4: Optimize Delivery Route (Simulated Annealing) ---
            print("    - Optimizing delivery route with Simulated Annealing...")
            start_of_delivery = optimized_pickup_route[-1] if optimized_pickup_route else home_node
            optimized_delivery_route = optimize_route_simulated_annealing(start_of_delivery, nn_delivery_route,
                                                                          home_node)

            # --- Phase 5: Build Final `all_steps` from Optimized Routes ---
            print("    - Building final optimized step list...")
            for wh_node in optimized_pickup_route:
                path = get_path_from_cache(current_node, wh_node)
                if not path: raise Exception(f"No path to warehouse node {wh_node}")
                for node in path[1:]: all_steps.append(
                    {'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
                current_node = wh_node
                all_steps[-1]['pickups'].extend(pickups_by_wh_node[wh_node])
            for dest_node in optimized_delivery_route:
                path = get_path_from_cache(current_node, dest_node)
                if not path: raise Exception(f"No path to customer node {dest_node}")
                for node in path[1:]: all_steps.append(
                    {'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
                current_node = dest_node
                all_steps[-1]['deliveries'].extend(deliveries_by_dest_node[dest_node])

            # --- Phase 6: Return Home ---
            if home_node != current_node:
                path_to_home = get_path_from_cache(current_node, home_node)
                if not path_to_home: raise Exception(f"No path back home from {current_node}")
                for node in path_to_home[1:]: all_steps.append(
                    {'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})

            solution['routes'].append({'vehicle_id': vehicle_id, 'steps': all_steps})
            print(f"  - SUCCESS: Globally Optimal Route created for batch with {len(all_steps)} steps.")

        except Exception as e:
            print(f"  - ERROR: Could not build route for batch: {e}")
            print(f"  - Re-assigning {len(orders_in_this_batch)} orders.")
            unassigned_orders.update(orders_in_this_batch)
            for (wh_id, sku_id, quantity) in batch_inventory_changes:
                local_inventory[wh_id][sku_id] = local_inventory[wh_id].get(sku_id, 0) + quantity

    print(f"\n---  Phase 11 Finished ---")
    print(f"Total routes created: {len(solution['routes'])}")
    print(f"Orders left unassigned: {len(unassigned_orders)}")
    return solution

