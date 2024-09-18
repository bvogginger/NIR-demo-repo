# Dynamic power management demo on SpiNNaker2

Here we use the N-MNIST example to showcase the dynamic power management with DVFS on SpiNNaker2.

## Dependencies
- py-spinnaker2 branch [54-integrate-dvfs-for-snn-simulation](https://gitlab.com/spinnaker2/py-spinnaker2/-/tree/54-integrate-dvfs-for-snn-simulation?ref_type=heads)
- s2-sim2lab-app branch [dvfs-for-snn](https://gitlab.com/spinnaker2/s2-sim2lab-app/-/tree/dvfs-for-snn?ref_type=heads)

## Instructions

1. Run `run_subset()` in [demo.py](demo.py) to measure the number of spike packets and time needed per timestep for all cores.
    - It is recommended to pick a rather high systick duration (e.g., `sys_tick_in_s = 10.0e-3`) to make sure all spikes are processed.
    - Disable DVFS --> use `PL_MODE_LOW` all over the place
2. Analyse the recorded number of packets and time done records with script [analyze_sweep.py](analyze_sweep.py).
    - Check for the maximum time_done of all simulations to choose your standard systick without DVFS and the lower performance mode.
    - For each of the neuron types - here `lif_neuron` and `lif_conv2d` - manually choose a `pl_threshold`, defined by the number of packets from which the high performance level (double clock frequency) is chosen per time step.
    - Run the model again with power management enabled and check the systick duration is never exceeded.
3. Power measurements:
    - See [power2.py](power2.py) for power measurement. 
    - See [plot_power.py](plot_power.py) for plotting.
