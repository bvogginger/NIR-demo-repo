from mmap import mmap
import re
from datetime import datetime, timezone
import json

import numpy as np
import matplotlib.pyplot as plt

R_PL08_SHUNT     = 0.008 #Ohm (R570)
R_PL05_SHUNT     = 0.03  #Ohm (R569)
R_VDDPLL08_SHUNT = 0.05  #Ohm (R131)
R_VDDIO18_SHUNT  = 0.05  #Ohm (R572)
R_VDDQ11_SHUNT   = 0.05  #Ohm (R571)

def get_shunt_V_t(filepath):
    """
    returns shunt voltages of PL05 and PL08 in uV over time
    """
    f = open(filepath, "r+")
    data = mmap(f.fileno(), 0)
    pl08_data = np.array([int(i) for i in re.findall(b'0.8V:.(-*[0-9]+) uV', data)])
    pl05_data = np.array([int(i) for i in re.findall(b'0.5V:.(-*[0-9]+) uV', data)])
    pll_data  = np.array([int(i) for i in re.findall(b'PLL:.(-*[0-9]+) uV', data)])
    io_data   = np.array([int(i) for i in re.findall(b'IO:.(-*[0-9]+) uV', data)])
    q_data    = np.array([int(i) for i in re.findall(b'Q:.(-*[0-9]+) uV', data)])

    return pl05_data, pl08_data, pll_data, io_data, q_data


def get_shunt_I_t(filepath):
    """
    returns shunt currents of PL05 and PL08 in mA over time
    """
    pl05_data, pl08_data, pll_data, io_data, q_data = get_shunt_V_t(filepath)
    return (0.001*pl05_data/R_PL05_SHUNT, 0.001*pl08_data/R_PL08_SHUNT, 0.001*pll_data/R_VDDPLL08_SHUNT,
            0.001*io_data/R_VDDIO18_SHUNT, 0.001*q_data/R_VDDQ11_SHUNT)


def get_shunt_P_t(filepath, bus_volt):
    """
    returns power of PL05 and PL08 in mW over time
    """
    pl05_data, pl08_data, pll_data, io_data, q_data = get_shunt_I_t(filepath)
    return pl05_data*bus_volt[0], pl08_data*bus_volt[1], pll_data*bus_volt[2], io_data*bus_volt[3], q_data*bus_volt[4]

def get_unix_timestamps(filepath):
    with open(filepath, "r+") as f:
        data = mmap(f.fileno(), 0)
        t_data = np.array([int(i) for i in re.findall(b'T:.(-*[0-9]+)', data)])
    return t_data

def get_t_start(filepath):
    with open(filepath, "r+") as f:
        data = mmap(f.fileno(), 0)
        t_data = np.array([int(i) for i in re.findall(b't_start = .(-*[0-9]+)', data)])
    return t_data

def plot_P_t(filepath, rdout_delay, bus_volt, t_start=None, t_end=None, plot_all=False):
    """
    """
    t_record_start = get_t_start(filepath)[0]

    # times = get_unix_timestamps(filepath)
    # print(times)
    # diff = np.diff(times)
    # print(diff)
    pl05_data, pl08_data, pll_data, io_data, q_data = get_shunt_P_t(filepath, bus_volt)

    t = np.arange(len(pl05_data))*(rdout_delay/1000) #s
    times = t * 1e9 + t_record_start
    t = times

    plt.plot(t, pl05_data, marker='o', label='PL05')
    plt.plot(t, pl08_data, marker='o', label='PL08')
    if plot_all:
        plt.plot(t, pll_data,  marker='o', label='PLL')
        plt.plot(t, io_data,   marker='o', label='IO')
        plt.plot(t, q_data,    marker='o', label='Q')

    if t_start:
        index_start = next(i for i,t in enumerate(times) if t > t_start)
        plt.axvline(t[index_start], label="start")
    if t_end:
        index_end = next(i for i,t in enumerate(times) if t > t_end)
        plt.axvline(t[index_end], label="end")

    if t_start and t_end:
        total_power = 0.0
        for data in pl05_data, pl08_data, pll_data, io_data, q_data:
            total_power += np.mean(data[index_start:index_end])
        print("Average total power:", total_power, "mW")
        t1 = datetime.fromtimestamp(t_start/1e9, tz=timezone.utc)
        t2 = datetime.fromtimestamp(t_end/1e9, tz=timezone.utc)
        duration = (t2 - t1).total_seconds()
        energy = total_power/1e3 * duration
        print("Duration:", duration, "seconds")
        print("Energy:", energy, "Joule")
        #t2 = datetime.fromtimestamp(t_end)
        print(t1)
        #print(duration)
        result_str = f"Duration: {duration:.3f} s\nEnergy: {energy:.3f} J"
        props = dict(boxstyle='round', facecolor='wheat')
        plt.text(.01, .99, result_str, ha="left", va="top", transform=plt.gca().transAxes, bbox=props)

    plt.ylabel('shunt power [mW]')
    plt.xlabel('time [s]')
    plt.legend(loc='upper right')
    plt.savefig("power_measurement.pdf")
    plt.show()


if __name__ == "__main__":
    filepath = "power_record_PL_high.log"
    filepath = "power_record_PL_auto.log"
    filepath = "power_record_PL_low.log"
    volt = [0.5, 0.8, 0.8, 1.8, 1.1] # vdd05, vdd08, vddpll, vddio, vddq (WIP)
    # PL high
    t_start = 227602195710525
    t_end = 227602827971502
    # PL auto
    t_start = 231685970330642
    t_end = 231686602442347
    # PL auto
    t_start = 232069315874527
    t_end = 232069942164073

    filepath = "my_power_record.log"
    with open("results.json", "r") as f:
        results = json.load(f)
    profiling = results["profiling"]
    t_start = profiling["t_experiment_start"]
    t_end = profiling["t_experiment_done"]
    plot_P_t(filepath, rdout_delay=100, bus_volt=volt, t_start=t_start, t_end=t_end, plot_all=True)

