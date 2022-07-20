import numpy as np
import matplotlib.pylab as plt
import os


root_dir = "/home/joe/projects/theseus/theseus/optimizer/gbp/outputs"
err_files1 = [
    "gbp_problem-21-11315-pre.txt",
    "levenberg_marquardt_problem-21-11315-pre.txt",
]
err_files2 = [
    "gbp_problem-50-20431-pre.txt",
    "levenberg_marquardt_problem-50-20431-pre.txt",
]


err_files = err_files1

for err_files in [err_files1, err_files2]:

    gbp_err = np.loadtxt(os.path.join(root_dir, err_files[0]))
    lm_err = np.loadtxt(os.path.join(root_dir, err_files[1]))

    plt.plot(gbp_err, label="GBP")
    plt.plot(lm_err, label="Levenberg Marquardt")
    plt.xscale("log")
    plt.title(err_files[0][4:])
    plt.xlabel("Iterations")
    plt.ylabel("Total Energy")
    plt.legend()
    plt.show()
