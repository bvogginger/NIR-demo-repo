import numpy as np
import pickle
from spinnaker2 import helpers


for pop in ["input", "1", "3", "6", "10", "output"]:
    filepath = f"time_done_{pop}.npz"
    npzfile = np.load(filepath, allow_pickle=True)
    pes = npzfile["pes"]
    times = npzfile["times"]
    #print(times.shape)
    print("\n", pes)
    print("Max:", times.max())
    print("Min:", times.min())
    print("Mean:", times.mean())

    helpers.plot_times_done_multiple_pes_one_plot_vertical(filepath)

with open("time_done_results.pkl", "rb") as fp:
    d = pickle.load(fp)
    print(d.keys())
