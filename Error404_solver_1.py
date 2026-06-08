
from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple


def find_shortest_path(adj_list: Dict[int, List[int]], start_node: int, end_node: int) -> Optional[List[int]]:
    """
    Find shortest path between two nodes using BFS.
    Returns list of node IDs or None if no path exists.
    """
    if start_node == end_node:
        return [start_node]

    queue = [(start_node, [start_node])]
    visited = {start_node}

    while queue:
        current_node, path = queue.pop(0)

        if current_node not in adj_list:
            continue

        for neighbor in adj_list[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]

                if neighbor == end_node:
                    return new_path

                queue.append((neighbor, new_path))

    return None


def calculate_path_length(adj_list: Dict[int, List[int]], start: int, end: int) -> float:
    """Calculate shortest path length between nodes. Returns inf if no path."""
    path = find_shortest_path(adj_list, start, end)
    return len(path) - 1 if path else float('inf')


def find_optimal_warehouse_for_order(env, order_id: int, local_inventory: Dict,
                                     adj_list: Dict) -> Tuple[float, Optional[int]]:
    """
    Find the closest warehouse that can fulfill the entire order.
    Returns: (distance_to_order, warehouse_id) or (inf, None)
    """
    requirements = env.get_order_requirements(order_id)
    dest_node = env.get_order_location(order_id)

    candidates = []

    for wh_id, inventory in local_inventory.items():
        # Check if this warehouse can fulfill all SKUs
        can_fulfill = all(
            inventory.get(sku_id, 0) >= quantity
            for sku_id, quantity in requirements.items()
        )

        if can_fulfill:
            wh_node = env.get_warehouse_by_id(wh_id).location.id
            distance = calculate_path_length(adj_list, wh_node, dest_node)
            candidates.append((distance, wh_id))

    if not candidates:
        return (float('inf'), None)

    candidates.sort()
    return candidates[0]


def compute_order_profile(env, order_id: int, local_inventory: Dict,
                          adj_list: Dict, vehicle_home_node: int) -> Optional[Dict]:
    """
    Compute comprehensive metrics for an order including weight, volume,
    optimal warehouse, and routing efficiency.
    """
    try:
        requirements = env.get_order_requirements(order_id)
        dest_node = env.get_order_location(order_id)

        # Calculate cargo metrics
        total_weight = 0
        total_volume = 0

        for sku_id, quantity in requirements.items():
            sku_details = env.get_sku_details(sku_id)
            total_weight += sku_details['weight'] * quantity
            total_volume += sku_details['volume'] * quantity

        # Find optimal warehouse
        dist_to_dest, best_wh_id = find_optimal_warehouse_for_order(
            env, order_id, local_inventory, adj_list
        )

        if best_wh_id is None:
            return None

        wh_node = env.get_warehouse_by_id(best_wh_id).location.id

        # Calculate routing metrics
        dist_vehicle_to_wh = calculate_path_length(adj_list, vehicle_home_node, wh_node)
        total_route_dist = dist_vehicle_to_wh + dist_to_dest

        # Efficiency: lower is better (distance per unit cargo)
        cargo_units = max(total_weight, 1)
        efficiency = total_route_dist / cargo_units

        return {
            'order_id': order_id,
            'weight': total_weight,
            'volume': total_volume,
            'best_warehouse_id': best_wh_id,
            'warehouse_node': wh_node,
            'destination_node': dest_node,
            'dist_wh_to_dest': dist_to_dest,
            'total_route_distance': total_route_dist,
            'requirements': requirements,
            'efficiency_score': efficiency
        }
    except Exception:
        return None


def select_orders_knapsack(vehicle, order_profiles: List[Dict],
                           local_inventory: Dict) -> List[Dict]:
    """
    Select optimal subset of orders for vehicle using greedy knapsack approach.
    Prioritizes efficiency while respecting capacity constraints.
    """
    # Sort by efficiency (lower distance per cargo unit is better)
    sorted_profiles = sorted(order_profiles, key=lambda x: x['efficiency_score'])

    selected = []
    cumulative_weight = 0
    cumulative_volume = 0

    for profile in sorted_profiles:
        # Check capacity feasibility
        if (cumulative_weight + profile['weight'] <= vehicle.capacity_weight and
                cumulative_volume + profile['volume'] <= vehicle.capacity_volume):

            # Verify inventory availability
            wh_id = profile['best_warehouse_id']
            inventory_available = all(
                local_inventory[wh_id].get(sku_id, 0) >= qty
                for sku_id, qty in profile['requirements'].items()
            )

            if inventory_available:
                selected.append(profile)
                cumulative_weight += profile['weight']
                cumulative_volume += profile['volume']

    return selected


def build_route_for_batch(env, vehicle_id: int, selected_orders: List[Dict],
                          adj_list: Dict) -> Dict:
    """
    Construct a complete route with pickups and deliveries for selected orders.
    """
    vehicle_home_node = env.get_vehicle_home_warehouse(vehicle_id)

    # Initialize route
    all_steps = [{
        'node_id': vehicle_home_node,
        'pickups': [],
        'deliveries': [],
        'unloads': []
    }]
    current_node = vehicle_home_node

    # Group pickups by warehouse node - ensure no duplicates
    pickups_by_node = {}
    for order in selected_orders:
        wh_id = order['best_warehouse_id']
        wh_node = order['warehouse_node']

        if wh_node not in pickups_by_node:
            pickups_by_node[wh_node] = []

        # Add each SKU pickup for this order
        for sku_id, quantity in order['requirements'].items():
            pickups_by_node[wh_node].append({
                'warehouse_id': wh_id,
                'sku_id': sku_id,
                'quantity': quantity
            })

    # Group deliveries by destination node - ensure complete orders
    deliveries_by_node = {}
    for order in selected_orders:
        dest_node = order['destination_node']
        order_id = order['order_id']

        if dest_node not in deliveries_by_node:
            deliveries_by_node[dest_node] = []

        # Add all SKUs for this order together
        for sku_id, quantity in order['requirements'].items():
            deliveries_by_node[dest_node].append({
                'order_id': order_id,
                'sku_id': sku_id,
                'quantity': quantity
            })

    # Execute pickups
    for wh_node, pickup_actions in pickups_by_node.items():
        if wh_node != current_node:
            path = find_shortest_path(adj_list, current_node, wh_node)
            if not path:
                raise Exception(f"No path to warehouse at node {wh_node}")

            for node in path[1:]:
                all_steps.append({
                    'node_id': node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
            current_node = wh_node

        all_steps[-1]['pickups'].extend(pickup_actions)

    # Execute deliveries
    for dest_node, delivery_actions in deliveries_by_node.items():
        if dest_node != current_node:
            path = find_shortest_path(adj_list, current_node, dest_node)
            if not path:
                raise Exception(f"No path to destination at node {dest_node}")

            for node in path[1:]:
                all_steps.append({
                    'node_id': node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
            current_node = dest_node

        all_steps[-1]['deliveries'].extend(delivery_actions)

    # Return to home warehouse
    if current_node != vehicle_home_node:
        path_home = find_shortest_path(adj_list, current_node, vehicle_home_node)
        if not path_home:
            raise Exception(f"No path back to home warehouse from node {current_node}")

        for node in path_home[1:]:
            all_steps.append({
                'node_id': node,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })

    return {
        'vehicle_id': vehicle_id,
        'steps': all_steps
    }


def my_solver(env) -> Dict:
    """
    Main solver function for MWVRP.
    Optimizes for cost minimization and high order fulfillment.
    """
    solution = {"routes": []}

    print("--- Starting Solver ---")

    # Load road network
    try:
        road_network = env.get_road_network_data()
        adj_list = road_network.get("adjacency_list", {})
    except Exception as e:
        print(f"Error loading road network: {e}")
        return solution

    # Initialize tracking
    unassigned_orders = set(env.get_all_order_ids())
    available_vehicles = env.get_available_vehicles()

    # Create local inventory copy for tracking
    local_inventory = {
        wh_id: warehouse.inventory.copy()
        for wh_id, warehouse in env.warehouses.items()
    }

    print(f"Total orders: {len(unassigned_orders)}")
    print(f"Total vehicles: {len(available_vehicles)}")

    # Process each vehicle
    for vehicle_id in available_vehicles:
        if not unassigned_orders:
            break

        print(f"\n=== Processing Vehicle: {vehicle_id} ===")

        vehicle = env.get_vehicle_by_id(vehicle_id)
        vehicle_home_node = env.get_vehicle_home_warehouse(vehicle_id)

        # Compute profiles for all unassigned orders
        order_profiles = []
        for order_id in unassigned_orders:
            profile = compute_order_profile(
                env, order_id, local_inventory, adj_list, vehicle_home_node
            )
            if profile:
                order_profiles.append(profile)

        if not order_profiles:
            print("  No feasible orders for this vehicle")
            continue

        # Select optimal batch using knapsack approach
        selected_orders = select_orders_knapsack(
            vehicle, order_profiles, local_inventory
        )

        if not selected_orders:
            print("  No orders selected after knapsack optimization")
            continue

        print(f"  Selected {len(selected_orders)} orders")

        # Track inventory changes for potential rollback
        inventory_changes = []
        orders_in_batch = set()

        # Reserve inventory for selected orders
        for order in selected_orders:
            wh_id = order['best_warehouse_id']

            for sku_id, quantity in order['requirements'].items():
                inventory_changes.append((wh_id, sku_id, quantity))
                local_inventory[wh_id][sku_id] -= quantity

            orders_in_batch.add(order['order_id'])
            unassigned_orders.remove(order['order_id'])

        # Build and validate route
        try:
            route = build_route_for_batch(
                env, vehicle_id, selected_orders, adj_list
            )

            # Verify route structure before adding
            if not route or 'vehicle_id' not in route or 'steps' not in route:
                raise Exception("Invalid route structure")

            if not route['steps'] or len(route['steps']) == 0:
                raise Exception("Route has no steps")

            # Verify first and last steps are at home warehouse
            home_node = env.get_vehicle_home_warehouse(vehicle_id)
            if route['steps'][0]['node_id'] != home_node:
                raise Exception(f"Route doesn't start at home warehouse")
            if route['steps'][-1]['node_id'] != home_node:
                raise Exception(f"Route doesn't end at home warehouse")

            solution['routes'].append(route)
            print(f"   Route created with {len(route['steps'])} steps")

        except Exception as e:
            # Route construction failed - rollback
            print(f"  ✗ Route construction failed: {e}")
            unassigned_orders.update(orders_in_batch)

            for (wh_id, sku_id, quantity) in inventory_changes:
                if sku_id in local_inventory[wh_id]:
                    local_inventory[wh_id][sku_id] += quantity
                else:
                    local_inventory[wh_id][sku_id] = quantity

    print(f"\n--- Solver Complete ---")
    print(f"Routes created: {len(solution['routes'])}")
    print(f"Unassigned orders: {len(unassigned_orders)}")

    return solution

