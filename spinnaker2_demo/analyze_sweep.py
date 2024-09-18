import os
import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from spinnaker2 import helpers
from plot_time_done import plot_time_done_vs_n_packets

sys_tick_in_s=2.0e-3
# load 

n_samples = 50
sweep_dir = "results_subset_mapping_2_dvfs/"
all_time_done = [] # list[{layer : {PE : np.array}}}
all_n_packets = [] # list[{layer : {PE : np.array}}}
for i in range(n_samples):
    base_path = os.path.join(sweep_dir, f"sample_{i}")
    with open(os.path.join(base_path, "time_done_results.pkl"), "rb") as fp:
        time_done_dict = pickle.load(fp)
        all_time_done.append(time_done_dict)
    with open(os.path.join(base_path, "n_packets_results.pkl"), "rb") as fp:
        n_packets_dict = pickle.load(fp)
        all_n_packets.append(n_packets_dict)

#####################################
# max time done for each layer type #
#####################################

all_layer_names = all_time_done[0].keys()
max_time_done = {layer:0 for layer in all_layer_names}

for result in all_time_done:
    for layer, map_pe_times in result.items():
        for times in map_pe_times.values():
            max_t = times.max()
            max_time_done[layer] = max(max_time_done[layer], max_t)

# turn into seconds and pretty print
max_time_done_in_s = {key:value*sys_tick_in_s for key,value in max_time_done.items()}
sorted_layer_names = ["input", "1", "3", "6", "10", "12"]
assert(set(sorted_layer_names) == set(all_layer_names))

for layer in sorted_layer_names:
    print(layer, ":",  max_time_done_in_s[layer], "s")


##################################
# time done vs number of packets #
##################################

# time done
flat_time_done = {} # {layer : {PE : [n.array]}
for result in all_time_done:
    for layer, map_pe_times in result.items():
        if not layer in flat_time_done:
            flat_time_done[layer] = {}
        for PE, times in map_pe_times.items():
            if not PE in flat_time_done[layer]:
                flat_time_done[layer][PE] = []
            flat_time_done[layer][PE].append(times)

flat_time_done_arr = {} # {layer : {PE : [n.array]}
for layer, map_pe_times in flat_time_done.items():
    flat_time_done_arr[layer] = {}
    for PE, times in map_pe_times.items():
        flat_time_done_arr[layer][PE] = np.concatenate(flat_time_done[layer][PE])

# n_packets
flat_n_packets = {} # {layer : {PE : [n.array]}
for result in all_n_packets:
    for layer, map_pe_times in result.items():
        if not layer in flat_n_packets:
            flat_n_packets[layer] = {}
        for PE, times in map_pe_times.items():
            if not PE in flat_n_packets[layer]:
                flat_n_packets[layer][PE] = []
            flat_n_packets[layer][PE].append(times)

flat_n_packets_arr = {} # {layer : {PE : [n.array]}
for layer, map_pe_times in flat_n_packets.items():
    flat_n_packets_arr[layer] = {}
    for PE, times in map_pe_times.items():
        flat_n_packets_arr[layer][PE] = np.concatenate(flat_n_packets[layer][PE])

for pop in ["1", "3", "6", "10", "12"]:
    plot_time_done_vs_n_packets(flat_time_done_arr[pop], flat_n_packets_arr[pop], sys_tick_in_s, pop)

# print(len(flat_time_done_arr[pop]["PE(1, 1, 0)"]))
# print(len(flat_n_packets_arr[pop]["PE(1, 1, 0)"]))
# print(flat_time_done[pop]["PE(1, 1, 0)"])

# join results using the same neuron type
pops_lif_conv2d = ["1", "3", "6"]
pops_lif_neuron = ["10", "12"]
flat_time_done_arr_per_neuron_type = {"lif_conv2d":{}, "lif_neuron":{}}
flat_n_packets_arr_per_neuron_type = {"lif_conv2d":{}, "lif_neuron":{}}

for pop, data in flat_time_done_arr.items():
    if pop in pops_lif_conv2d:
        flat_time_done_arr_per_neuron_type["lif_conv2d"].update(data)
    elif pop in pops_lif_neuron:
        flat_time_done_arr_per_neuron_type["lif_neuron"].update(data)

for pop, data in flat_n_packets_arr.items():
    if pop in pops_lif_conv2d:
        flat_n_packets_arr_per_neuron_type["lif_conv2d"].update(data)
    elif pop in pops_lif_neuron:
        flat_n_packets_arr_per_neuron_type["lif_neuron"].update(data)


for neuron_type in ["lif_conv2d", "lif_neuron"]:
    plot_time_done_vs_n_packets(
            flat_time_done_arr_per_neuron_type[neuron_type],
            flat_n_packets_arr_per_neuron_type[neuron_type],
            sys_tick_in_s,
            neuron_type)
