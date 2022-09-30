import numpy as np
import os
import json

import matplotlib.pylab as plt


def plot_timing_memory(root):
    dirs = os.listdir(root)
    dirs.remove("figs")

    timings = {}
    memory = {}

    for direc in dirs:

        with open(os.path.join(root, direc, "timings.txt"), "r") as f:
            timings[direc] = json.load(f)
        with open(os.path.join(root, direc, "memory.txt"), "r") as f:
            memory[direc] = json.load(f)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))
    fig.subplots_adjust(hspace=0.0, wspace=0.3)

    exps = ["full", "implicit", "truncated_5", "truncated_10"]
    labels = ["Unroll", "Implicit", "Trunc-5", "Trunc-10"]
    colors = ["C0", "C1", "C2", "C3"]
    markers = [".", "v", "o", "s"]
    inner_iters = [25, 50, 100, 150, 200, 500]

    for i, exp in enumerate(exps):

        fwd_times = []
        bwd_times = []
        fwd_memory = []
        bwd_memory = []
        for iters in inner_iters:
            key = f"{str(iters)}_{exp}"
            fwd_times.append(np.mean(timings[key]["fwd"]) / 1e3)
            bwd_times.append(np.mean(timings[key]["bwd"]) / 1e3)
            fwd_memory.append(np.mean(memory[key]["fwd"]))
            bwd_memory.append(np.mean(memory[key]["bwd"]))

        col = colors[i]
        m = markers[i]
        ax[0].plot(inner_iters, fwd_times, color=col, marker=m, label=labels[i])
        ax[1].plot(inner_iters, bwd_times, color=col, marker=m)
        ax[2].plot(inner_iters, fwd_memory, color=col, marker=m)
        ax[3].plot(inner_iters, bwd_memory, color=col, marker=m)

    title_fontsize = 11
    ax[0].title.set_text("Forward time")
    ax[1].title.set_text("Backward time")
    ax[2].title.set_text("Forward memory")
    ax[3].title.set_text("Backward memory")
    ax[0].title.set_size(title_fontsize)
    ax[1].title.set_size(title_fontsize)
    ax[2].title.set_size(title_fontsize)
    ax[3].title.set_size(title_fontsize)

    ax[0].set_xlabel("Inner loop iterations")
    ax[1].set_xlabel("Inner loop iterations")
    ax[2].set_xlabel("Inner loop iterations")
    ax[3].set_xlabel("Inner loop iterations")
    ax[0].set_ylabel("Time (seconds)")
    ax[1].set_ylabel("Time (seconds)")
    ax[2].set_ylabel("Memory (MBs)")
    ax[3].set_ylabel("Memory (MBs)")

    ax[0].legend(
        loc="lower center",
        bbox_to_anchor=(2.5, -0.5),
        fancybox=True,
        ncol=4,
        fontsize=10,
    )

    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()


def plot_loss_traj(root, ref_loss=None):

    exps = ["full", "implicit", "truncated_5", "truncated_10"]
    labels = ["Unroll", "Implicit", "Trunc-5", "Trunc-10"]
    colors = ["C0", "C1", "C2", "C3"]
    inner_iters = [150, 200, 500]  # [25, 50, 100, 150, 200, 500]

    fig_loss, ax_loss = plt.subplots(nrows=1, ncols=len(inner_iters), figsize=(20, 3))
    fig_loss.subplots_adjust(hspace=0.0, wspace=0.5)

    fig_loss_t, ax_loss_t = plt.subplots(
        nrows=1, ncols=len(inner_iters), figsize=(20, 3)
    )
    fig_loss_t.subplots_adjust(hspace=0.0, wspace=0.5)

    fig_traj, ax_traj = plt.subplots(nrows=len(inner_iters), ncols=4, figsize=(20, 15))
    fig_traj.subplots_adjust(hspace=0.75, wspace=0.4)

    for i, iters in enumerate(inner_iters):

        for j, exp in enumerate(exps):
            direc = f"{str(iters)}_{exp}"

            # plot sweep curves
            if j == 0:
                sweep_radii = np.loadtxt(os.path.join(root, direc, "sweep_radius.txt"))
                sweep_loss = np.loadtxt(os.path.join(root, direc, "sweep_loss.txt"))
                for k in range(len(exps)):
                    ax_traj[i, k].plot(sweep_radii, sweep_loss)
                    ax_traj[i, k].title.set_text(labels[k])

            # plot trajectory over epochs
            loss_traj = np.loadtxt(os.path.join(root, direc, "optim_loss.txt"))
            radius_traj = np.loadtxt(os.path.join(root, direc, "optim_radius.txt"))
            ax_traj[i, j].scatter(
                radius_traj,
                loss_traj,
                c=range(len(loss_traj)),
                cmap=plt.get_cmap("viridis"),
            )

            # plot loss over epochs or over total time
            label = labels[j] if i == 0 else None
            if ref_loss is not None:
                loss_traj = np.array(loss_traj) * ref_loss + ref_loss
            with open(os.path.join(root, direc, "timings.txt"), "r") as f:
                timings = json.load(f)
            step_times = [
                timings["fwd"][i] + timings["bwd"][i]
                for i in range(len(timings["fwd"]))
            ]
            step_times = np.array(step_times) / 1000
            cum_times = np.cumsum(step_times)
            ax_loss[i].plot(loss_traj, color=colors[j], marker=None, label=label)
            ax_loss_t[i].plot(
                cum_times, loss_traj, color=colors[j], marker=None, label=label
            )

    for ax in [ax_loss, ax_loss_t]:
        ax[0].legend(
            loc="lower center",
            bbox_to_anchor=(2.0, -0.5),  # (3.5, -0.5)
            fancybox=True,
            ncol=4,
            fontsize=10,
        )

    # align y axes
    ymin, ymax = 1e10, -1e10
    for ax in ax_loss:
        ymin = min(ymin, ax.get_ylim()[0])
        ymax = max(ymax, ax.get_ylim()[1])
    for ax in ax_loss:
        ax.set_ylim((ymin, ymax))
    for ax in ax_loss_t:
        ax.set_ylim((ymin, ymax))

    title_fontsize = 11
    for i, iters in enumerate(inner_iters):
        for ax in [ax_loss, ax_loss_t]:
            ax[i].title.set_text(f"Train loss ({iters} inner GBP steps)")
            ax[i].title.set_size(title_fontsize)
            ax[i].set_ylabel("Camera Loss")
        ax_loss[i].set_xlabel("Epoch")
        ax_loss_t[i].set_xlabel("Time (seconds)")

        for j in range(4):
            ax_traj[i, j].set_xlabel("Huber loss radius")
            if j == 0:
                ax_traj[i, j].set_ylabel(f"{iters} inner steps\n\n\nCamera Loss")
            else:
                ax_traj[i, j].set_ylabel("Camera Loss")

    fig_loss.subplots_adjust(bottom=0.3)
    fig_loss_t.subplots_adjust(bottom=0.3)
    plt.show()


if __name__ == "__main__":

    root = (
        "/home/joe/projects/theseus/theseus/optimizer/gbp/"
        + "outputs/loss_radius_exp/backward_analysis/"
    )

    # plot_timing_memory(root)

    plot_loss_traj(root, ref_loss=None)
