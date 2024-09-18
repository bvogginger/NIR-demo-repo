#!/usr/bin/env python
# coding: utf-8

# # From NIR to SpiNNaker2
# 
# *(this notebook is based on the [Speck demo](https://github.com/Jegp/NIR-demo-repo/blob/main/speck_demo/demo.ipynb) by Felix Bauer)*
# 
# In this notebook we will show how a NIR model can be deployed onto the SpiNNaker2 chip. 
# 
# [py-spinnaker2](https://gitlab.com/spinnaker2/py-spinnaker2), the high-level software interface for running spiking neural networks on SpiNNaker2, provides an API similar to [PyNN](http://neuralensemble.org/docs/PyNN/) and allows to define populations (groups of neurons with the same neuron model) and projections (group of synapses between two populations).

# In[1]:

import time
start_time = time.time()

# Import statements
import os
from matplotlib import pyplot as plt
import nir
import numpy as np
import pickle
import tonic
import torch
from tqdm.notebook import tqdm
from spinnaker2 import brian2_sim, hardware, helpers, s2_nir, snn
import spinnaker2.neuron_models.lif_neuron
import spinnaker2.neuron_models.lif_conv2d
from spinnaker2.neuron_models.common import DVFSParams, PmgtMode

# set custom DVFS params
if True:
    dvfs_lif_conv2d = DVFSParams(mode=PmgtMode.PL_MODE_LOW, pl_threshold=30)
    dvfs_lif_neuron = DVFSParams(mode=PmgtMode.PL_MODE_LOW, pl_threshold=200)
    spinnaker2.neuron_models.lif_neuron.LIFNoDelayApplication.dvfs_params = dvfs_lif_neuron
    spinnaker2.neuron_models.lif_conv2d.LIFConv2dApplication.dvfs_params = dvfs_lif_conv2d
sys_tick_in_s = 2.0e-3

# Matplotlib settings
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 20


# ## Load NIR model from disk

# In[2]:


model_path = "scnn_mnist.nir"
nir_graph = nir.read(model_path)

# make sure all nodes have necessary shape information
nir_graph.infer_types()
s2_nir.model_summary(nir_graph)


# ## Convert NIR graph into py-spinnaker2 network
# 
# Let's convert the nir graph into a `spinnaker2.snn.Network()`.

# In[3]:


# Configuration for converting NIR graph to SpiNNaker2
conversion_cfg = s2_nir.ConversionConfig()
conversion_cfg.output_record = ["v", "spikes", "time_done", "n_packets"]
conversion_cfg.dt = 0.0001
conversion_cfg.conn_delay = 0
conversion_cfg.scale_weights = True # Scale weights to dynamic range on chip
conversion_cfg.reset = s2_nir.ResetMethod.ZERO # Reset voltage to zero at spike
conversion_cfg.integrator = s2_nir.IntegratorMethod.FORWARD # Euler-Forward

net, inp, outp = s2_nir.from_nir(nir_graph, conversion_cfg)


# ### Load the neuromorphic MNIST dataset
# 
# Let's quickly run the test set through the sinabs model to make sure everything works as it should.

# In[4]:


# load dataset
to_frame = tonic.transforms.ToFrame(
    sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1e3
)
dataset = tonic.datasets.NMNIST(".", transform=to_frame, train=False)

# Only use every 200th sample
indices = torch.arange(50) * 200
subset = torch.utils.data.Subset(dataset, indices)


# ## Deploy the model onto SpiNNaker2
# 
# Deploying a model onto SpiNNaker2 looks as follows:
# 
# ### Customize neurons per core

# In[5]:


# Customize neurons per core per population
for pop in net.populations:
    if pop.name == "input":
        pop.record = ["time_done"]
    if pop.name == "1":
        pop.record = ["time_done", "n_packets"]
    if pop.name == "3":
        pop.set_max_atoms_per_core(256)
        pop.record = ["time_done", "n_packets"]
    if pop.name == "6":
        pop.set_max_atoms_per_core(64)
        pop.record = ["time_done", "n_packets"]
    if pop.name == "10":
        pop.set_max_atoms_per_core(8)
        pop.record = ["time_done", "n_packets"]


# ### Convert input data to spikes
# While the torch dataset uses tensors, py-spinnaker2's input populations of type `spike_list` require spike times as input. Here's the conversion function that also considers flattening of the 3-dimensional frames

# In[6]:


def convert_input(x):
    d = {}
    # T = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    for c in range(C):
        for h in range(H):
            for w in range(W):
                d[c * H * W + h * W + w] = x[:, c, h, w].nonzero()[0].tolist()
    return d


# Function to run one sample on the chip and return voltages and spikes of output layer.

# In[7]:


def run_single(hw, net, inp, outp, x):
    """run single sample on SpiNNaker2
    
    Args:
      hw: spinnaker2.Hardware instance
      net: spinnaker2 Network
      inpu: list of input populations
      outp: list of output populations
      x: input sample of shape (T,C,H,W)
      
    Returns:
      tuple (voltages, spikes, time_done_dict, n_packets_dict): voltages and
      spikes of output layer, time_done and n_packets dict of all layers.
    """
    input_spikes = convert_input(x)
    inp[0].params = input_spikes

    timesteps = x.shape[0] + 1
    net.reset()
    print(f"Run for {timesteps} time steps")
    hw.run(net, timesteps, sys_tick_in_s=sys_tick_in_s, debug=False)
    voltages = outp[0].get_voltages()
    spikes = outp[0].get_spikes()


    time_done_times_dict = {}
    time_done_times = outp[0].get_time_done_times()
    time_done_times_dict[outp[0].name] = time_done_times

    for pop in net.populations:
        if pop.name in ["input", "1", "3", "6", "10"]:
            time_done_times = pop.get_time_done_times()
            time_done_times_dict[pop.name] = time_done_times

    n_packets_dict = {}
    n_packets = outp[0].get_n_packets()
    n_packets_dict[outp[0].name] = n_packets

    for pop in net.populations:
        if pop.name in ["1", "3", "6", "10"]:
            n_packets = pop.get_n_packets()
            n_packets_dict[pop.name] = n_packets

    return voltages, spikes, time_done_times_dict, n_packets_dict


# ### Some helper functions

# In[8]:


# This will help us choose samples of a given target
targets = np.array([y for x,y in subset])
target_indices = {idx: np.where(targets == idx)[0] for idx in range(10)}

def plot_hist(output_spikes, target):
    # spike count
    spike_counts = np.zeros(10)
    for idx, spikes in output_spikes.items():
        spike_counts[idx] = len(spikes)
    prediction = np.argmax(spike_counts)


    # Draw histogram
    fig, ax = plt.subplots()
    bins = np.arange(11)-0.5
    # N, bins, patches = ax.hist(features, bins=bins, edgecolor='white', linewidth=1)
    patches = ax.bar(np.arange(10), spike_counts, edgecolor='white', linewidth=1)
    plt.title(f"Prediction: {prediction} ({f'target: {target}' if target!=prediction else 'correct'})")
    plt.ylabel("Event count")
    plt.xlabel("Feature")
    plt.xticks(np.arange(10));
    
    # Set bar colors according to prediction and target
    for i, patch in enumerate(patches):
        if i == prediction and i == target:
            patch.set_facecolor('g')
        elif i == prediction:
            patch.set_facecolor('r')
        elif i == target:
            patch.set_facecolor('k')
    
    # Make xtick label of prediciton bold
    ax.xaxis.get_major_ticks()[prediction].label1.set_fontweight("bold")

    return prediction

def test_sample(target):
    index = np.random.choice(target_indices[target])
    sample, tgt = subset[index]
    assert(target == tgt)
    image = sample.sum((0, 1))
    plt.imshow(image)
    plt.title("Input")
    
    # run on SpiNNaker 2
    hw = hardware.SpiNNaker2Chip(eth_ip="192.168.2.33")
    # hw = brian2_sim.Brian2Backend()

    voltages, spikes, time_done_dict, n_packets_dict = run_single(hw, net, inp, outp, sample)
    del hw
    
    prediction = plot_hist(spikes, target)
    print(f"SpiNNaker2 prediction: {prediction}")

    with open("time_done_results.pkl", "wb") as fp:
        pickle.dump(time_done_dict, fp)

    with open("n_packets_results.pkl", "wb") as fp:
        pickle.dump(n_packets_dict, fp)



# ## Live demo
# 
# Now let's run the example on the SpiNNaker2 chip:

# In[9]:


test_sample(2)



# ## Send test data to the chip and read out its prediction
# To get some quantitative idea about how well the on-chip model does, we can use the test data from above and run it through the chip. **Note: this will take more than 6 minutes!!!**

# In[10]:


def run_subset():
    correct = 0
    predictions = []
    result_dir = "results_subset_mapping_2_dvfs"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for i, (sample, target) in enumerate(tqdm(subset, total=len(subset))):

        if not os.path.exists(f"{result_dir}/sample_{i}"):
            os.makedirs(f"{result_dir}/sample_{i}")

        # run on SpiNNaker 2
        hw = hardware.SpiNNaker2Chip(eth_ip="192.168.2.33")
        # hw = brian2_sim.Brian2Backend()

        voltages, spikes, time_done_dict, n_packets_dict = run_single(hw, net, inp, outp, sample)
        del hw

        spike_counts = np.zeros(10)
        for idx, spike_times in spikes.items():
            spike_counts[idx] = len(spike_times)
        prediction = np.argmax(spike_counts)

        correct += (prediction == target)
        predictions.append(prediction)

        # save time done and n packets
        with open(f"{result_dir}/sample_{i}/time_done_results.pkl", "wb") as fp:
            pickle.dump(time_done_dict, fp)

        with open(f"{result_dir}/sample_{i}/n_packets_results.pkl", "wb") as fp:
            pickle.dump(n_packets_dict, fp)

    accuracy = correct / len(subset)
    print(f"Test accuracy on SpiNNaker2: {accuracy:.2%}")


# In[11]:


# run_subset()

end_time = time.time()
print("Duration: ", end_time - start_time)

