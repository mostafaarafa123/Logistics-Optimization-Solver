from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional


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


def my_solver(env) -> Dict:
    """
    Main solver function.
    PHASE 7: Fixing the inventory revert logic on routing failure.
    """

    solution = {"routes": []}

    print("---  Starting Phase 7 Solver (Bug Fix) ---")

    # 1. Get Road Network Data
    try:
        road_network = env.get_road_network_data()
        adj_list = road_network.get("adjacency_list", {})
    except Exception as e:
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

        if not unassigned_orders:
            break

        print(f"\nBuilding batch for Vehicle '{vehicle_id}'...")
        vehicle = env.get_vehicle_by_id(vehicle_id)

        batch_items_to_pickup = []
        batch_deliveries = []

        current_weight = 0
        current_volume = 0
        orders_in_this_batch = set()

        # --- FIX #1: Create a master list for *all* inventory changes in this batch ---
        batch_inventory_changes = []
        # --- END FIX #1 ---

        # --- Batching Loop ---
        for order_id in list(unassigned_orders):

            try:
                requirements = env.get_order_requirements(order_id)
                dest_node = env.get_order_location(order_id)

                order_weight = 0
                order_volume = 0
                items_for_this_order = []

                order_is_feasible = True

                # --- FIX #2: Create a *temporary* list for *this order only* ---
                order_inventory_changes = []
                # --- END FIX #2 ---

                for sku_id, quantity in requirements.items():
                    sku_details = env.get_sku_details(sku_id)
                    order_weight += sku_details['weight'] * quantity
                    order_volume += sku_details['volume'] * quantity

                    found_warehouse = False
                    for wh_id, inventory in local_inventory.items():
                        if inventory.get(sku_id, 0) >= quantity:
                            warehouse_node = env.get_warehouse_by_id(wh_id).location.id
                            items_for_this_order.append((sku_id, quantity, wh_id, warehouse_node, dest_node))

                            # Add change to the *order's* temp list
                            order_inventory_changes.append((wh_id, sku_id, quantity))
                            found_warehouse = True
                            break

                    if not found_warehouse:
                        order_is_feasible = False
                        break

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
                        batch_deliveries.append((order_id, sku_id, q, dest))

                    # --- Apply and Store inventory changes ---
                    # Add this order's changes to the master *batch* list
                    batch_inventory_changes.extend(order_inventory_changes)

                    # Apply changes to local inventory
                    for (wh_id, sku_id, quantity) in order_inventory_changes:
                        local_inventory[wh_id][sku_id] -= quantity
                    # --- END Apply and Store ---

                    unassigned_orders.remove(order_id)

            except Exception as e:
                pass
        # --- END: Batching Loop ---

        if not orders_in_this_batch:
            print("  - No orders batched for this vehicle.")
            continue

        print(f"  - Batch complete for '{vehicle_id}'. Building route for {len(orders_in_this_batch)} orders...")

        # 5. Find the paths between ALL stops (Logic remains the same)
        try:
            home_node = env.get_vehicle_home_warehouse(vehicle_id)
            all_steps = [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
            current_node = home_node

            pickups_by_wh_node = {}
            for (sku_id, quantity, warehouse_id, warehouse_node) in batch_items_to_pickup:
                if warehouse_node not in pickups_by_wh_node:
                    pickups_by_wh_node[warehouse_node] = []
                pickups_by_wh_node[warehouse_node].append(
                    {'warehouse_id': warehouse_id, 'sku_id': sku_id, 'quantity': quantity}
                )

            deliveries_by_dest_node = {}
            for (order_id, sku_id, quantity, dest_node) in batch_deliveries:
                if dest_node not in deliveries_by_dest_node:
                    deliveries_by_dest_node[dest_node] = []
                deliveries_by_dest_node[dest_node].append(
                    {'order_id': order_id, 'sku_id': sku_id, 'quantity': quantity}
                )

            for wh_node, pickup_actions in pickups_by_wh_node.items():
                if wh_node != current_node:
                    path = find_shortest_path(adj_list, current_node, wh_node)
                    if not path:
                        raise Exception(f"No path to warehouse node {wh_node}")
                    for node in path[1:]:
                        all_steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
                    current_node = wh_node
                all_steps[-1]['pickups'].extend(pickup_actions)

            for dest_node, delivery_actions in deliveries_by_dest_node.items():
                if dest_node != current_node:
                    path = find_shortest_path(adj_list, current_node, dest_node)
                    if not path:
                        raise Exception(f"No path to customer node {dest_node}")
                    for node in path[1:]:
                        all_steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
                    current_node = dest_node
                all_steps[-1]['deliveries'].extend(delivery_actions)

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
            print(f"  - ERROR: Could not build route for batch: {e}")
            print(f"  - Re-assigning {len(orders_in_this_batch)} orders.")
            unassigned_orders.update(orders_in_this_batch)

            # --- FIX #3: Revert inventory using the MASTER batch list ---
            for (wh_id, sku_id, quantity) in batch_inventory_changes:
                if sku_id in local_inventory[wh_id]:
                    local_inventory[wh_id][sku_id] += quantity
                else:
                    local_inventory[wh_id][sku_id] = quantity  # (Just in case)
            # --- END FIX #3 ---

    print(f"\n---  Phase 7 Finished ---")
    print(f"Total routes created: {len(solution['routes'])}")
    print(f"Orders left unassigned: {len(unassigned_orders)}")

    return solution
