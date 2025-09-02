import os
import shutil

# Path to your root results folder
root_dir = "results"

removed_routes = []

# Traverse through results/Town#_Scenario#
for town_dir in os.listdir(root_dir):
    town_path = os.path.join(root_dir, town_dir)
    if not os.path.isdir(town_path):
        continue

    # Traverse each route folder
    for route_dir in os.listdir(town_path):
        route_path = os.path.join(town_path, route_dir)
        if not os.path.isdir(route_path):
            continue

        # Flag for removal
        remove_route = False

        # Check all subfolders (depth, rgb, semantics, topdown, label_raw, measurements)
        for subfolder in os.listdir(route_path):
            subfolder_path = os.path.join(route_path, subfolder)

            if os.path.isdir(subfolder_path) and not os.listdir(subfolder_path):
                print(f"Empty subfolder found: {subfolder_path}")
                remove_route = True
                break  # no need to check further, one empty is enough

        # Remove route folder if any empty subfolder was found
        if remove_route:
            print(f"Removing route folder: {route_path}")
            # shutil.rmtree(route_path)
            # removed_routes.append(route_path)

print(f"\nDone. Removed {len(removed_routes)} route folders.")
