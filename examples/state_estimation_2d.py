# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import theseus as th

device = "cpu"
torch.manual_seed(0)
path_length = 50
state_size = 2
batch_size = 4
learning_method = "leo"  # "default", "leo"

vis_flag = True
plt.ion()


# --------------------------------------------------- #
# --------------------- Utilities ------------------- #
# --------------------------------------------------- #
def plot_path(optimizer_path, groundtruth_path):
    plt.cla()
    plt.gca().axis("equal")

    plt.xlim(-250, 250)
    plt.ylim(-100, 400)

    batch_idx = 0
    plt.plot(
        optimizer_path[batch_idx, :, 0],
        optimizer_path[batch_idx, :, 1],
        linewidth=2,
        linestyle="-",
        color="tab:orange",
        label="optimizer",
    )
    plt.plot(
        groundtruth_path[batch_idx, :, 0],
        groundtruth_path[batch_idx, :, 1],
        linewidth=2,
        linestyle="-",
        color="tab:green",
        label="groundtruth",
    )

    plt.show()
    plt.pause(1e-12)


def generate_path_data(
    batch_size_,
    num_measurements_,
    generator=None,
):
    vel_ = torch.ones(batch_size, 2)
    path_ = [torch.zeros(batch_size, 2)]
    for _ in range(1, num_measurements_):
        new_state_ = path_[-1] + vel_
        path_.append(new_state_)
        vel_ += 0.75 * torch.randn(batch_size, 2, generator=generator)
    return path_


class SimpleNN(nn.Module):
    def __init__(self, in_size, out_size, hid_size=30, use_offset=False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, out_size),
        )

    def forward(self, state_):
        return self.fc(state_)


def run_model(
    mode_,
    cost_weights_model_,
    current_inputs_,
    path_length_,
    print_stuff=False,
):
    weights_ = get_weights_dict_from_model(
        mode_,
        cost_weights_model_,
        current_inputs_,
        path_length_,
        print_stuff=print_stuff,
    )
    theseus_inputs_ = {}
    theseus_inputs_.update(current_inputs_)
    theseus_inputs_.update(weights_)

    return theseus_inputs_


def get_weights_dict_from_model(
    mode_, cost_weights_model_, values_, path_length_, print_stuff=False
):
    if mode_ == "constant":
        weights_dict = {}
        unique_weight_ = cost_weights_model_()
        for i in range(path_length_):
            weights_dict[f"scale_gps_{i}"] = unique_weight_[0, 0].view(1, 1)
            if i < path_length_ - 1:
                weights_dict[f"scale_between_{i}"] = unique_weight_[0, 1].view(1, 1)
    else:
        weights_dict = {}
        all_states_ = []
        # will compute weight for all cost weights in the path
        for i in range(path_length_):
            all_states_.append(values_[f"pose_{i}"])
        model_input_ = torch.cat(all_states_, dim=0)

        weights_ = cost_weights_model_(model_input_)
        for i in range(path_length_):
            weights_dict[f"scale_gps_{i}"] = weights_[i - 1, 0].view(1, 1)
            if i < path_length_ - 1:
                weights_dict[f"scale_between_{i}"] = weights_[i - 1, 1].view(1, 1)

    if print_stuff:
        with torch.no_grad():
            print("scale5", weights_dict["scale_gps_5"].item())
            print("scale45", weights_dict["scale_gps_45"].item())
            print("btwn5", weights_dict["scale_between_5"].item())
            print("btwn45", weights_dict["scale_between_45"].item())

    return weights_dict


def get_initial_inputs(gps_targets_):
    inputs_ = {}
    for i, _ in enumerate(gps_targets_):
        inputs_[f"pose_{i}"] = gps_targets_[i] + torch.randn(1)
    return inputs_


def get_path_from_values(batch_size_, values_, path_length_):
    path = torch.empty(batch_size_, path_length_, 2, device=device)
    for i in range(path_length_):
        path[:, i, :2] = values_[f"pose_{i}"]
    return path


def get_values_from_path(path_):
    """
    :param path_: tensor of dim batch_size_ x path_length_ x 2
    :return: values: dict of (x,y) pos values
    """
    [batch_size_, path_length_, dim] = path_.shape
    values = {}
    for i in range(path_length_):
        values[f"pose_{i}"] = path_[:, i, :2]
    return values


def get_average_sample_cost(x_samples, cost_weights_model, objective, mode_):
    cost_opt = None
    n_samples = x_samples.shape[-1]
    for sidx in range(0, n_samples):
        x_sample_vals = get_values_from_path(
            x_samples[:, :, sidx].reshape(x_samples.shape[0], -1, 2)
        )
        theseus_inputs = run_model(
            mode_,
            cost_weights_model,
            x_sample_vals,
            path_length,
            print_stuff=False,
        )
        objective.update(theseus_inputs)
        if cost_opt is not None:
            cost_opt = cost_opt + torch.sum(objective.error(), dim=1)
        else:
            cost_opt = torch.sum(objective.error(), dim=1)
    cost_opt = cost_opt / n_samples

    return cost_opt


# ------------------------------------------------------------- #
# --------------------------- Learning ------------------------ #
# ------------------------------------------------------------- #
def run_learning(mode_, path_data_, gps_targets_, measurements_):
    # first input is scale for GPS costs, second is scale for Between costs
    if mode_ == "constant":
        model_params = nn.Parameter(10 * torch.rand(1, 2, device=device))

        def cost_weights_model():
            return model_params * torch.ones(1)

        model_optimizer = torch.optim.Adam([model_params], lr=5e-2)
    else:
        cost_weights_model = SimpleNN(state_size, 2, hid_size=100, use_offset=False).to(
            device
        )
        model_optimizer = torch.optim.Adam(
            cost_weights_model.parameters(),
            lr=7e-5,
        )

    # GPS and Between cost weights
    gps_cost_weights = []
    between_cost_weights = []
    for i in range(path_length):
        gps_cost_weights.append(
            th.ScaleCostWeight(th.Variable(torch.ones(1, 1), name=f"scale_gps_{i}"))
        )
        if i < path_length - 1:
            between_cost_weights.append(
                th.ScaleCostWeight(
                    th.Variable(torch.ones(1, 1), name=f"scale_between_{i}")
                )
            )

    # ## Variables
    poses = []
    for i in range(path_length):
        poses.append(th.Point2(name=f"pose_{i}"))

    # ## Cost functions
    cost_functions: List[th.CostFunction] = []

    # ### GPS and between cost functions
    for i in range(path_length):
        cost_functions.append(
            th.Difference(
                poses[i],
                th.Point2(tensor=gps_targets_[i]),
                gps_cost_weights[i],
                name=f"gps_{i}",
            )
        )
        if i < path_length - 1:
            cost_functions.append(
                (
                    th.Between(
                        poses[i],
                        poses[i + 1],
                        th.Point2(tensor=measurements_[i]),
                        between_cost_weights[i],
                        name=f"between_{i}",
                    )
                )
            )

    # # Create Theseus layer and initial values for variables
    objective = th.Objective()
    for cost_function in cost_functions:
        objective.add(cost_function)
    optimizer = th.GaussNewton(
        objective,
        th.CholeskyDenseSolver,
        max_iterations=1,
        step_size=0.9,
    )
    state_estimator = th.TheseusLayer(optimizer)
    state_estimator.to(device)

    # ## Learning loop
    best_loss = 1000.0
    inner_loop_iters = 3
    groundtruth_path = torch.stack(path_data_).permute(1, 0, 2)
    best_solution = None
    losses = []
    for epoch in range(500):
        model_optimizer.zero_grad()

        theseus_inputs = get_initial_inputs(gps_targets_)
        theseus_inputs = run_model(
            mode_,
            cost_weights_model,
            theseus_inputs,
            path_length,
            print_stuff=False,
        )
        objective.update(theseus_inputs)
        with torch.no_grad():
            if epoch % 10 == 0:
                print("Initial error:", objective.error_metric().mean().item())

        for i in range(inner_loop_iters):
            theseus_inputs, _ = state_estimator.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": epoch % 10 == 0,
                },
            )
            theseus_inputs = run_model(
                mode_,
                cost_weights_model,
                theseus_inputs,
                path_length,
                print_stuff=epoch % 10 == 0 and i == 0,
            )

        optimizer_path = get_path_from_values(
            objective.batch_size, theseus_inputs, path_length
        )
        mse_loss = F.mse_loss(optimizer_path, groundtruth_path)

        # LEO (Sodhi et al., https://arxiv.org/abs/2108.02274) is a method to learn
        # models end-to-end within second-order optimizers. The main difference is that
        # instead of unrolling the optimizer and minimizing the MSE tracking loss,
        # it uses a NLL energy-based loss that does not backpropagate through the optimizer.
        if learning_method == "leo":
            x_samples = state_estimator.compute_samples(
                optimizer.linear_solver, n_samples=10, temperature=1.0
            )  # batch_size x n_vars x n_samples
            # When x_samples is None, this defaults to a perceptron loss
            # using the mean trajectory solution from the optimizer.
            if x_samples is None:
                x_opt_dict = {key: val.detach() for key, val in theseus_inputs.items()}
                x_samples = get_path_from_values(
                    objective.batch_size, x_opt_dict, path_length
                )
                x_samples = x_samples.reshape(x_samples.shape[0], -1).unsqueeze(
                    -1
                )  # batch_size x n_vars x 1
            cost_opt = get_average_sample_cost(
                x_samples, cost_weights_model, objective, mode_
            )
            x_gt = get_values_from_path(groundtruth_path)
            theseus_inputs_gt = run_model(
                mode_,
                cost_weights_model,
                x_gt,
                path_length,
                print_stuff=False,
            )
            objective.update(theseus_inputs_gt)
            cost_gt = torch.sum(objective.error(), dim=1)
            loss = cost_gt - cost_opt
        else:
            loss = mse_loss

        loss = torch.mean(loss, dim=0)
        loss.backward()
        model_optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_solution = optimizer_path.detach()

        if epoch % 10 == 0:
            if vis_flag:
                plot_path(
                    optimizer_path.detach().cpu().numpy(),
                    groundtruth_path.detach().cpu().numpy(),
                )
            print("Loss: ", loss.item())
            print("MSE error: ", mse_loss.item())
            print(f" ---------------- END EPOCH {epoch} -------------- ")

    return best_solution, losses


path_data = generate_path_data(batch_size, 50)
gps_targets = []
measurements = []
for i in range(path_length):
    gps_noise = 0.075 * path_data[i][1].abs() * torch.randn(batch_size, 2)
    gps_target = (path_data[i] + gps_noise).view(batch_size, 2)
    gps_targets.append(gps_target)

    if i < path_length - 1:
        measurement = (path_data[i + 1] - path_data[i]).view(batch_size, 2)
        measurement_noise = 0.005 * torch.randn(batch_size, 2).view(batch_size, 2)
        measurements.append(measurement + measurement_noise)

constant_solution, constant_losses = run_learning(
    "constant", path_data, gps_targets, measurements
)
