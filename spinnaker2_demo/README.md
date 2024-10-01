# Dynamic power management demo on SpiNNaker2

Here we use the N-MNIST example to showcase the dynamic power management with DVFS on SpiNNaker2.

## Dependencies
- py-spinnaker2 branch [54-integrate-dvfs-for-snn-simulation](https://gitlab.com/spinnaker2/py-spinnaker2/-/tree/54-integrate-dvfs-for-snn-simulation?ref_type=heads)
- s2-sim2lab-app branch [dvfs-for-snn](https://gitlab.com/spinnaker2/s2-sim2lab-app/-/tree/dvfs-for-snn?ref_type=heads)

## Relevant files

- `demo.py`
- `plot_time_done.py`
- `analyze_sweep.py`
- [plot_power.py](plot_power.py): outdated code for plotting power

## Instructions

1. Run [demo.py](demo.py) with `dvfs_mode=calibration` to measure the number of spike packets and time needed per timestep for all cores.
    - DVFS is disabled, instead the low performance mode is used.
    - It uses a long simulation tick of 10ms
2. Analyse the recorded number of packets and time done records with script [analyze_sweep.py](analyze_sweep.py).
    - Check for the maximum time_done of all simulations to choose your standard systick without DVFS and the lower performance mode.
    - For each of the neuron types - here `lif_neuron` and `lif_conv2d` - manually choose a `pl_threshold`, defined by the number of packets from which the high performance level (double clock frequency) is chosen per time step.
    - Choose half of the standard systick when running in `auto` or `high` mode.
3. Run the model in 3 different power management modes in `demo.py` with power measurement enabled.
    - Modes: `auto`, `low`, `high`.
    - Energy and accuracy are stored in file `all_results.json` in subfolder `results_subset_{mode}`
