import os
import sys
import time
from subprocess import Popen
from multiprocessing import Process, Pipe
import numpy as np

# import measurement functions
HELPER_SCRIPT_PATH = "../../s2-sim2lab-app/host/stm_meas_helper/"

sys.path.append(HELPER_SCRIPT_PATH)
from read_power import Serial, ReadLineWrapper

def process_readout(conn, file_path, rdout_delay, ser_path='/dev/ttyACM0', get_all=False):
    """
    """
    print(ser_path)
    ser = Serial(ser_path)
    r = ReadLineWrapper(ser)

    stdout = sys.stdout
    sys.stdout = open(file_path, "w")

    # read out shunt voltages from all INA223AIDGS in while loop
    if get_all:
        write_cmd = ('PA' + str(rdout_delay).zfill(6) + '\n').encode() 
        n_lines = 5
    else:
        write_cmd = ('Pa' + str(rdout_delay).zfill(6) + '\n').encode() 
        n_lines = 2

    ser.write(write_cmd)

    # conn.send("init")

    # time.sleep(2.0)

    start_time_ns = time.monotonic_ns()
    start_time = time.monotonic()
    n_samples = 0
    delay_in_s = rdout_delay/1000
    while True:
        if conn.poll():
            if conn.recv() == "stop":
                break
        else:
            # print("T:", time.monotonic_ns())
            for i in range(n_lines):
                print((r.readline()).decode().rstrip())
            n_samples += 1
            next_time = start_time + n_samples*delay_in_s
            wait_time = next_time - time.monotonic()
            time.sleep(max((wait_time, 0)))

    end_time_ns = time.monotonic_ns()
    print("t_start = ", start_time_ns)
    print("t_end = ", end_time_ns)
    print("n_samples = ", n_samples)

    # stop continous stm32 readout
    ser.write('\n'.encode())
    time.sleep(0.5)
    ser.write('\n'.encode())
    ser.close()
    ser.__del__()

# num_dpoints=100
rdout_delay = 100
get_all = True
volt = [0.5, 0.8, 0.8, 1.8, 1.1] # vdd05, vdd08, vddpll, vddio, vddq (WIP)
freq_cfg = [1, 300, 2, 300, 0x002d8402, 6, 6, 6, 6]
ser_path="/dev/ttyS2_192_168_2_33"
data_file = "power_record.log"
log_file = "main.log"

# proc_arr = np.array(['sleep', '10'])
proc_arr = np.array(['python', 'demo.py'])
#if FLAGS.fpga_ip            is not None: proc_arr = np.append(proc_arr, ['-f', FLAGS.fpga_ip           ])
print('Execute the following: ' + ' '.join(proc_arr))


parent_conn, child_conn = Pipe()

p = Process(
	target=process_readout,
	args=(
        child_conn,
		data_file,
		rdout_delay,
		ser_path,
		get_all
	),
)
p.start()

"""
while True:
    if parent_conn.recv() == "init":
        break
"""

stdout = sys.stdout
sys.stdout = open(log_file, 'a')
with open(log_file, 'w') as f:
	proc1 = Popen(proc_arr, stdout=f, stderr=f)
proc1.wait()

parent_conn.send("stop")
p.join()

sys.stdout = stdout
