import pickle

import matplotlib.pyplot as plt
import numpy as np

def heat_map_time_done():
    with open("time_done_results.pkl", "rb") as fp:
        time_done_dict = pickle.load(fp)
        print(time_done_dict.keys())

    layer_names = ["input", "1", "3", "6", "10", "12"]
    core_names = []
    time_done_per_core = []
    n_cores_per_layer = []

    for layer in layer_names:
        time_done_per_layer = time_done_dict[layer]
        cores = time_done_dict[layer].keys()
        for core in cores:
            core_names.append(core)
            time_done_per_core.append(time_done_per_layer[core])
        n_cores_per_layer.append(len(cores))

    time_done_per_core = np.array(time_done_per_core)

    # only consider the first 100 steps
    time_done_per_core = time_done_per_core[:,:100]

    layer_limits = np.cumsum(n_cores_per_layer)
    layer_limits = [0] + layer_limits.tolist()
    layer_limits = np.asarray(layer_limits)
    layer_centers = (layer_limits[:-1] + layer_limits[1:])/2
    print(layer_limits)

    plt.figure()
    plt.imshow(time_done_per_core, origin="lower")

    xmin = -0.5
    xmax = time_done_per_core.shape[1] - 0.5
    ymax = time_done_per_core.shape[0] - 0.5
    plt.hlines(layer_limits-0.5, xmin=xmin, xmax=xmax, colors="gray")
    for center, name in zip(layer_centers, layer_names):
        plt.text(xmax + 2, center, name, va="center_baseline")
    plt.text(xmax + 2, ymax + 4, "layer", va="center_baseline", fontweight="demibold")
    plt.colorbar(pad=0.15, shrink=0.7, label="time done [rel.]")
    plt.xlabel("time step")
    plt.ylabel("core")
    plt.title("time done per timestep and core")
    plt.savefig("time_done_heat_plot.pdf")

def heat_map_performance_levels():
    with open("performance_level_results.pkl", "rb") as fp:
        pl_dict = pickle.load(fp)
        print(pl_dict.keys())

    layer_names = ["1", "3", "6", "10", "12"]
    core_names = []
    pl_per_core = []
    n_cores_per_layer = []

    for layer in layer_names:
        pl_per_layer = pl_dict[layer]
        cores = pl_dict[layer].keys()
        for core in cores:
            core_names.append(core)
            pl_per_core.append(pl_per_layer[core])
        n_cores_per_layer.append(len(cores))

    pl_per_core = np.array(pl_per_core)

    # only consider the first 100 steps
    pl_per_core = pl_per_core[:,:100]

    layer_limits = np.cumsum(n_cores_per_layer)
    layer_limits = [0] + layer_limits.tolist()
    layer_limits = np.asarray(layer_limits)
    layer_centers = (layer_limits[:-1] + layer_limits[1:])/2
    print(layer_limits)

    plt.figure()
    plt.imshow(pl_per_core, origin="lower")

    xmin = -0.5
    xmax = pl_per_core.shape[1] - 0.5
    ymax = pl_per_core.shape[0] - 0.5
    plt.hlines(layer_limits-0.5, xmin=xmin, xmax=xmax, colors="gray")
    for center, name in zip(layer_centers, layer_names):
        plt.text(xmax + 2, center, name, va="center_baseline")
    plt.text(xmax + 2, ymax + 4, "layer", va="center_baseline", fontweight="demibold")
    # plt.colorbar(pad=0.15, shrink=0.7, label="performance level [rel.]")
    plt.xlabel("time step")
    plt.ylabel("core")
    plt.title("performance level per timestep and core")
    plt.savefig("pl_heat_plot.pdf")

if __name__ == "__main__":
    sys_tick_in_s=10e-3

    heat_map_performance_levels()
    heat_map_time_done()
    plt.show()
