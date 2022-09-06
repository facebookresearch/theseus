import numpy as np
import matplotlib.pylab as plt
import os


"""
Nesterov experiments
"""


def nesterov_plots():

    root_dir = (
        "/home/joe/projects/mpSLAM/theseus/theseus/optimizer/gbp/outputs/nesterov/"
    )
    exp_dir = root_dir + "bal/"

    err_normal = np.loadtxt(exp_dir + "0/error_history.txt")
    err_nesterov_normalize = np.loadtxt(exp_dir + "normalize/error_history.txt")
    err_nesterov_tp = np.loadtxt(exp_dir + "tangent_plane/error_history.txt")

    plt.plot(err_normal, label="Normal GBP")
    plt.plot(err_nesterov_normalize, label="Nesterov acceleration - normalize")
    plt.plot(err_nesterov_tp, label="Nesterov acceleration - lie algebra")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()


"""
Comparing GBP to Levenberg Marquardt
"""


def gbp_vs_lm():

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


if __name__ == "__main__":

    nesterov_plots()
