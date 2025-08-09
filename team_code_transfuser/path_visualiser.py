import matplotlib.pyplot as plt
import numpy as np

def visualise_from_tensor(waypoints):
    cmap = plt.cm.get_cmap('tab20')  # 20 distinct colors
    num_traj_sets = len(waypoints[0])
    print("roke")
    for i, traj_set in enumerate(waypoints[0]):
        np_traj_set = traj_set.cpu().detach().numpy()
        x = []
        y = []
        for coord in np_traj_set:
            x.append(coord[0])
            y.append(coord[1])
        plt.plot(x, y, color=cmap(i % cmap.N))  # Pick color per trajectory

    print("shit broke")
    plt.show()
