# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import theseus as th

device = "cpu"
torch.manual_seed(0)
path_length = 50
state_size = 2
batch_size = 4


# --------------------------------------------------- #
# --------------------- Utilities ------------------- #
# --------------------------------------------------- #
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


# ------------------------------------------------------------- #
# --------------------------- Learning ------------------------ #
# ------------------------------------------------------------- #
def run_learning(mode_, path_data_, gps_targets_, measurements_):

    # first input is scale for GPS costs, second is scale for Between costs
    if mode_ == "constant":
        model_params = nn.Parameter(10 * torch.rand(1, 2, device=device))

        def cost_weights_model():
            return model_params * torch.ones(1)

        model_optimizer = torch.optim.Adam([model_params], lr=3e-2)
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
            th.eb.VariableDifference(
                poses[i],
                gps_cost_weights[i],
                th.Point2(data=gps_targets_[i]),
                name=f"gps_{i}",
            )
        )
        if i < path_length - 1:
            cost_functions.append(
                (
                    th.eb.Between(
                        poses[i],
                        poses[i + 1],
                        between_cost_weights[i],
                        th.Point2(data=measurements_[i]),
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
    path_tensor = torch.stack(path_data_).permute(1, 0, 2)
    best_loss = 1000.0
    best_solution = None
    losses = []
    for epoch in range(200):
        model_optimizer.zero_grad()

        inner_loop_iters = 3
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
                print("Initial error:", objective.error_squared_norm().mean().item())

        for i in range(inner_loop_iters):
            theseus_inputs, info = state_estimator.forward(
                theseus_inputs,
                track_best_solution=True,
                verbose=epoch % 10 == 0,
            )
            theseus_inputs = run_model(
                mode_,
                cost_weights_model,
                theseus_inputs,
                path_length,
                print_stuff=epoch % 10 == 0 and i == 0,
            )

        solution_path = get_path_from_values(
            objective.batch_size, theseus_inputs, path_length
        )

        loss = F.mse_loss(solution_path, path_tensor)
        loss.backward()
        model_optimizer.step()
        loss_value = loss.item()
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_solution = solution_path.detach()

        if epoch % 10 == 0:
            print("TOTAL LOSS: ", loss.item())
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

mlp_solution, mlp_losses = run_learning("mlp", path_data, gps_targets, measurements)
print(" -------------------------------------------------------------- ")
constant_solution, constant_losses = run_learning(
    "constant", path_data, gps_targets, measurements
)
