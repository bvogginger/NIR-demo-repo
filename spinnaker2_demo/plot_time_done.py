import pickle

import matplotlib.pyplot as plt
import numpy as np

from spinnaker2 import helpers


def plot_time_done_vs_n_packets(time_done, n_packets, sys_tick_in_s, layer=None):
    # plot time done vs. number of received packets
    #plt.plot(np.stack(list(n_packets.values())), np.stack(list(time_done.values())) * sys_tick_in_s, ".", label=list(n_packets.keys()))
    plt.axhline(sys_tick_in_s, ls="-", c="0.5",label="1.0 tick")
    plt.axhline(sys_tick_in_s*0.5, ls="--", c="0.5", label="0.5 tick")
    for pe in n_packets:
        plt.plot(n_packets[pe], time_done[pe]*sys_tick_in_s, ".", label=pe)
    plt.xlabel("n_packets")
    plt.ylabel("time done [s]")
    plt.legend()
    plt.title(layer)
    plt.show()

if __name__ == "__main__":
    sys_tick_in_s=10e-3

    """
    for pop in ["input", "1", "3", "6", "10", "output"][:0]:
        filepath = f"time_done_{pop}.npz"
        npzfile = np.load(filepath, allow_pickle=True)
        pes = npzfile["pes"]
        times = npzfile["times"]
        #print(times.shape)
        print("\n", pes)
        print("Max:", times.max())
        print("Min:", times.min())
        print("Mean:", times.mean())

        helpers.plot_times_done_multiple_pes_one_plot_vertical(filepath, sys_tick_in_s)
    """

    with open("time_done_results.pkl", "rb") as fp:
        time_done_dict = pickle.load(fp)
        print(time_done_dict.keys())

    with open("n_packets_results.pkl", "rb") as fp:
        n_packets_dict = pickle.load(fp)
        print(n_packets_dict.keys())

    for pop in ["1", "3", "6", "10", "12"][:]:
        plot_time_done_vs_n_packets(time_done_dict[pop], n_packets_dict[pop], sys_tick_in_s, pop)
