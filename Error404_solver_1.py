from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Set


def find_shortest_path(adj_list: Dict[int, List[int]], start_node: int, end_node: int) -> Optional[List[int]]:
    """
    Finds the shortest path between start_node and end_node using BFS.
    Returns a list of node IDs representing the path, or None if no path exists.
    """
    if start_node == end_node:
        return [start_node]

    # queue will store (current_node, path_list)
    queue: List = [(start_node, [start_node])]
    visited = {start_node}

    while queue:
        (current_node, path) = queue.pop(0)

        # Check if the current_node is in the adjacency list
        if current_node not in adj_list:
            continue

        # Look at all neighbors of the current node
        for neighbor in adj_list[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)

                new_path = path + [neighbor]

                if neighbor == end_node:
                    return new_path  # We found the path

                # Add the neighbor to the queue to explore later
                queue.append((neighbor, new_path))

    return None  # No path was found


def find_nearest_node(adj_list: Dict[int, List[int]],
                      current_node: int,
                      target_nodes: Set[int]) -> Optional[int]:
    """
    Finds the node in target_nodes that is closest to current_node.
    Uses BFS to find the shortest path to *any* target.
    """
    if not target_nodes:
        return None

    if current_node in target_nodes:
        return current_node

    queue: List = [(current_node, [current_node])]
    visited = {current_node}

    while queue:
        (node, path) = queue.pop(0)

        if node not in adj_list:
            continue

        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)

                # Check if this neighbor is one of our targets
                if neighbor in target_nodes:
                    return neighbor  # Found the nearest target

                queue.append((neighbor, path + [neighbor]))

    return None  # No path found to any target


def my_solver(env) -> Dict:
    """
    Main solver function.
    PHASE 9: Implements:
    1. Split-Inventory (for 100% fulfillment)
    2. Smart Batching (Sort orders by proximity to home)
    3. Optimal Routing (Nearest Neighbor for pickups, then Nearest Neighbor for deliveries)
    """

    solution = {"routes": []}

    print("---  Starting Phase 9 Solver (NN Routing + Smart Batching) ---")

    # 1. Get Road Network Data
    try:
        road_network = env.get_road_network_data()
        adj_list = road_network.get("adjacency_list", {})
    except Exception as e:
        return solution

    # 2. Get all orders and vehicles
    unassigned_orders = set(env.get_all_order_ids())
    available_vehicles = env.get_available_vehicles()

    # Create a *local copy* of the inventory to track changes
    local_inventory = {}
    for wh_id, warehouse in env.warehouses.items():
        local_inventory[wh_id] = warehouse.inventory.copy()

    print(f"Total orders to serve: {len(unassigned_orders)}")
    print(f"Total vehicles available: {len(available_vehicles)}")

    # 3. Loop through vehicles and try to build a "batch" for each
    for vehicle_id in available_vehicles:

        if not unassigned_orders:
            break

        print(f"\nBuilding batch for Vehicle '{vehicle_id}'...")
        vehicle = env.get_vehicle_by_id(vehicle_id)
        home_node = env.get_vehicle_home_warehouse(vehicle_id)

        batch_items_to_pickup = []
        batch_deliveries = []

        current_weight = 0
        current_volume = 0
        orders_in_this_batch = set()

        batch_inventory_changes = []

        # --- IMPROVEMENT 1: SMART BATCHING ---
        # Sort unassigned orders by distance from the vehicle's home to the customer
        # This helps create geographically clustered batches.

        order_distances = []
        for order_id in unassigned_orders:
            try:
                dest_node = env.get_order_location(order_id)
                path = find_shortest_path(adj_list, home_node, dest_node)
                distance = len(path) - 1 if path else float('inf')
                if distance != float('inf'):
                    order_distances.append((order_id, distance))
            except Exception:
                pass  # Order might be invalid

        # Sort by distance (ascending)
        sorted_orders = sorted(order_distances, key=lambda x: x[1])

        print(f"  - Analyzing {len(sorted_orders)} reachable orders for batching...")

        # --- Batching Loop (Now uses sorted list) ---
        for order_id, distance in sorted_orders:

            # The order might have been assigned to another vehicle in the meantime
            if order_id not in unassigned_orders:
                continue

            try:
                requirements = env.get_order_requirements(order_id)
                dest_node = env.get_order_location(order_id)

                order_weight = 0
                order_volume = 0
                items_for_this_order = []
                order_inventory_changes = []
                order_is_feasible = True

                # --- Split-Inventory Logic (from Phase 8) ---
                for sku_id, quantity_needed in requirements.items():
                    sku_details = env.get_sku_details(sku_id)
                    sku_weight = sku_details['weight']
                    sku_volume = sku_details['volume']
                    quantity_found = 0

                    for wh_id, inventory in local_inventory.items():
                        if quantity_found == quantity_needed:
                            break

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
                        # Note: We don't print failure here, as it's too noisy
                        break
                        # --- End Split-Inventory Logic ---

                if not order_is_feasible:
                    continue

                # Check capacity
                if (current_weight + order_weight <= vehicle.capacity_weight and
                        current_volume + order_volume <= vehicle.capacity_volume):

                    print(f"  + Adding Order '{order_id}' to batch.")
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

        print(
            f"  - Batch complete for '{vehicle_id}'. Building OPTIMAL route for {len(orders_in_this_batch)} orders...")

        # 5. Find the paths between ALL stops
        try:
            all_steps = [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
            current_node = home_node

            # Group pickups by warehouse node
            pickups_by_wh_node = {}
            for (sku_id, quantity, warehouse_id, warehouse_node) in batch_items_to_pickup:
                if warehouse_node not in pickups_by_wh_node:
                    pickups_by_wh_node[warehouse_node] = []
                pickups_by_wh_node[warehouse_node].append(
                    {'warehouse_id': warehouse_id, 'sku_id': sku_id, 'quantity': quantity}
                )

            # Group deliveries by destination node
            deliveries_by_dest_node = {}
            for (order_id, sku_id, quantity, dest_node) in batch_deliveries:
                if dest_node not in deliveries_by_dest_node:
                    deliveries_by_dest_node[dest_node] = []
                deliveries_by_dest_node[dest_node].append(
                    {'order_id': order_id, 'sku_id': sku_id, 'quantity': quantity}
                )

            # --- IMPROVEMENT 2: OPTIMAL ROUTING (NEAREST NEIGHBOR) ---

            pickup_nodes_to_visit = set(pickups_by_wh_node.keys())
            delivery_nodes_to_visit = set(deliveries_by_dest_node.keys())

            # --- Phase 1: Pickups (Nearest Neighbor) ---
            print(f"    - Routing: Finding nearest pickups...")
            while pickup_nodes_to_visit:
                # Find the *closest* warehouse that still needs to be visited
                nearest_wh_node = find_nearest_node(adj_list, current_node, pickup_nodes_to_visit)

                if nearest_wh_node is None:
                    raise Exception("Routing Error: Cannot find path to any remaining pickup warehouse.")

                if nearest_wh_node != current_node:
                    path = find_shortest_path(adj_list, current_node, nearest_wh_node)
                    if not path:
                        raise Exception(f"No path to warehouse node {nearest_wh_node}")
                    for node in path[1:]:
                        all_steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
                    current_node = nearest_wh_node

                # Perform all pickups at this (now current) node
                all_steps[-1]['pickups'].extend(pickups_by_wh_node[nearest_wh_node])
                pickup_nodes_to_visit.remove(nearest_wh_node)

            # --- Phase 2: Deliveries (Nearest Neighbor) ---
            print(f"    - Routing: Finding nearest deliveries...")
            while delivery_nodes_to_visit:
                # Find the *closest* customer that still needs to be visited
                nearest_dest_node = find_nearest_node(adj_list, current_node, delivery_nodes_to_visit)

                if nearest_dest_node is None:
                    raise Exception("Routing Error: Cannot find path to any remaining customer.")

                if nearest_dest_node != current_node:
                    path = find_shortest_path(adj_list, current_node, nearest_dest_node)
                    if not path:
                        raise Exception(f"No path to customer node {nearest_dest_node}")
                    for node in path[1:]:
                        all_steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
                    current_node = nearest_dest_node

                # Perform all deliveries at this (now current) node
                all_steps[-1]['deliveries'].extend(deliveries_by_dest_node[nearest_dest_node])
                delivery_nodes_to_visit.remove(nearest_dest_node)

            # --- END: OPTIMAL ROUTING LOGIC ---

            # --- Phase 3: Return Home ---
            if home_node != current_node:
                path_to_home = find_shortest_path(adj_list, current_node, home_node)
                if not path_to_home:
                    raise Exception(f"No path back home from {current_node}")
                for node in path_to_home[1:]:
                    all_steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})

            route = {
                'vehicle_id': vehicle_id,
                'steps': all_steps
            }
            solution['routes'].append(route)
            print(f"  - SUCCESS: Route created for batch with {len(all_steps)} steps.")

        except Exception as e:
            # Routing failed, revert all changes for this batch
            print(f"  - ERROR: Could not build route for batch: {e}")
            print(f"  - Re-assigning {len(orders_in_this_batch)} orders.")
            unassigned_orders.update(orders_in_this_batch)

            # Revert inventory
            for (wh_id, sku_id, quantity) in batch_inventory_changes:
                if sku_id in local_inventory[wh_id]:
                    local_inventory[wh_id][sku_id] += quantity
                else:
                    local_inventory[wh_id][sku_id] = quantity

    print(f"\n---  Phase 9 Finished ---")
    print(f"Total routes created: {len(solution['routes'])}")
    print(f"Orders left unassigned: {len(unassigned_orders)}")

    return solution
