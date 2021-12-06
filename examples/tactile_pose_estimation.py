# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import theseus as th
import theseus.utils.examples as theg

# To run this example, you will need a tactile pushing dataset available at
# https://dl.fbaipublicfiles.com/theseus/tactile_pushing_data.tar.gz
#
# The steps below should let you run the example.
# From the root project folder do:
#   mkdir expts
#   cd expts
#   wget https://dl.fbaipublicfiles.com/theseus/tactile_pushing_data.tar.gz
#   tar -xzvf tactile_pushing_data.tar.gz
#   cd ..
#   python examples/tactile_pose_estimation.py
EXP_PATH = pathlib.Path.cwd() / "expts" / "tactile-pushing"
torch.set_default_dtype(torch.double)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

plt.ion()


# In this example, the goal is to track 2D object poses (x,y,theta) during planar
# pushing from tactile image measurements for single episode.
# This can solved as an optimization problem where the variables being estimated
# are object poses over time.
#
# We formulate the optimization using following cost terms:
#   * Quasi-static pushing planar: Penalizes deviation from quasi-static dynamics.
#     Uses velocity-only quasi-static model for sticking contact.
#   * Tactile measurements: Penalizes deviations from predicted relative pose
#     from tactile image feature pairs.
#   * Object-effector intersection: Penalizes intersections between object and
#     end-effector.
#   * End-effector priors: Penalizes deviation from end-effector poses captured from
#     motion capture.
#   * Boundary conditions: Penalizes deviation of first object pose from a global
#     pose prior.
#
# Based on the method described in,
# Sodhi et al. Learning Tactile Models for Factor Graph-based Estimation,
# 2021 (https://arxiv.org/abs/1705.10664)


def run_learning_loop(cfg):
    dataset_file = EXP_PATH / "datasets" / f"{cfg.dataset_name}.json"
    dataset = theg.load_tactile_dataset_from_file(
        filename=dataset_file, device=device, episode=cfg.episode
    )

    time_steps = np.minimum(cfg.max_steps, len(dataset["obj_poses"]))

    min_win_mf, max_win_mf, step_win_mf = (
        cfg.tactile_cost.min_win_mf,
        cfg.tactile_cost.max_win_mf,
        cfg.tactile_cost.step_win_mf,
    )

    # -------------------------------------------------------------------- #
    # Creating optimization variables
    # -------------------------------------------------------------------- #
    # The optimization variables for this problem are SE2 object and end effector
    # poses over time.
    obj_poses, eff_poses = [], []
    for i in range(time_steps):
        obj_poses.append(th.SE2(name=f"obj_pose_{i}", dtype=torch.double))
        eff_poses.append(th.SE2(name=f"eff_pose_{i}", dtype=torch.double))

    # -------------------------------------------------------------------- #
    # Creating auxiliary variables
    # -------------------------------------------------------------------- #
    #  - obj_start_pose: target for boundary cost functions
    #  - motion_captures: priors on the end-effector poses
    #  - nn_measurements: tactile measurement prediction from image features
    #  - sdf_data, sdf_cell_size, sdf_origin: signed distance field data,
    #    cell_size and origin
    obj_start_pose = th.SE2(
        x_y_theta=dataset["obj_poses"][0].unsqueeze(0), name="obj_start_pose"
    )

    motion_captures = []
    for i in range(time_steps):
        motion_captures.append(
            th.SE2(
                x_y_theta=dataset["eff_poses"][i].unsqueeze(0),
                name=f"motion_capture_{i}",
            )
        )

    nn_measurements = []
    for i in range(min_win_mf, time_steps):
        for offset in range(min_win_mf, np.minimum(i, max_win_mf), step_win_mf):
            nn_measurements.append(th.SE2(name=f"nn_measurement_{i-offset}_{i}"))

    sdf_tensor, sdf_cell_size, sdf_origin = theg.load_tactile_sdf_from_file(
        filename=EXP_PATH / "sdfs" / f"{cfg.sdf_name}.json",
        device=device,
    )

    sdf_data = th.Variable(sdf_tensor, name="sdf_data")
    sdf_cell_size = th.Variable(sdf_cell_size, name="sdf_cell_size")
    sdf_origin = th.Variable(sdf_origin, name="sdf_origin")
    eff_radius = th.Variable(torch.zeros(1, 1), name="eff_radius")

    # -------------------------------------------------------------------- #
    # Creating cost weights
    # -------------------------------------------------------------------- #
    #  - qsp_weight: diagonal cost weight shared across all quasi-static cost
    #    functions.
    #  - mf_between_weight: diagonal cost weight shared across all moving factor
    #    cost functions.
    #  - intersect_weight: scalar cost weight shared across all object-effector
    #    intersection cost functions.
    #  - motion_capture_weight: diagonal cost weight shared across all end-effector
    #    priors cost functions.
    qsp_weight = th.DiagonalCostWeight(th.Variable(torch.ones(1, 3), name="qsp_weight"))
    mf_between_weight = th.DiagonalCostWeight(
        th.Variable(torch.ones(1, 3), name="mf_between_weight")
    )
    intersect_weight = th.ScaleCostWeight(
        th.Variable(torch.ones(1, 1), name="intersect_weight")
    )
    motion_capture_weight = th.DiagonalCostWeight(
        th.Variable(torch.ones(1, 3), name="mc_weight")
    )

    # -------------------------------------------------------------------- #
    # Creating cost functions
    # -------------------------------------------------------------------- #
    #  - VariableDifference: Penalizes deviation between first object pose from
    #    a global pose prior.
    #  - QuasiStaticPushingPlanar: Penalizes deviation from velocity-only
    #    quasi-static dynamics model QuasiStaticPushingPlanar
    #  - MovingFrameBetween: Penalizes deviation between relative end effector poses
    #    in object frame against a measurement target. Measurement target
    #    `nn_measurements` is obtained from a network prediction.
    #  - EffectorObjectContactPlanar: Penalizes intersections between object and end
    #    effector based on the object sdf.
    #  - VariableDifference: Penalizes deviations of end-effector poses from motion
    #    capture readings

    # Loop over and add all cost functions,
    # cost weights, and their auxiliary variables
    objective = th.Objective()
    nn_meas_idx = 0
    c_square = (np.sqrt(cfg.shape.rect_len_x ** 2 + cfg.shape.rect_len_y ** 2)) ** 2
    for i in range(time_steps):
        if i == 0:
            objective.add(
                th.eb.VariableDifference(
                    obj_poses[i],
                    motion_capture_weight,
                    obj_start_pose,
                    name=f"obj_priors_{i}",
                )
            )

        if i < time_steps - 1:
            objective.add(
                th.eb.QuasiStaticPushingPlanar(
                    obj_poses[i],
                    obj_poses[i + 1],
                    eff_poses[i],
                    eff_poses[i + 1],
                    qsp_weight,
                    c_square,
                    name=f"qsp_{i}",
                )
            )
        if i >= min_win_mf:
            for offset in range(min_win_mf, np.minimum(i, max_win_mf), step_win_mf):
                objective.add(
                    th.eb.MovingFrameBetween(
                        obj_poses[i - offset],
                        obj_poses[i],
                        eff_poses[i - offset],
                        eff_poses[i],
                        mf_between_weight,
                        nn_measurements[nn_meas_idx],
                        name=f"mf_between_{i - offset}_{i}",
                    )
                )
                nn_meas_idx = nn_meas_idx + 1

        objective.add(
            th.eb.EffectorObjectContactPlanar(
                obj_poses[i],
                eff_poses[i],
                intersect_weight,
                sdf_origin,
                sdf_data,
                sdf_cell_size,
                eff_radius,
                name=f"intersect_{i}",
            )
        )

        objective.add(
            th.eb.VariableDifference(
                eff_poses[i],
                motion_capture_weight,
                motion_captures[i],
                name=f"eff_priors_{i}",
            )
        )

    # -------------------------------------------------------------------- #
    # Creating TheseusLayer
    # -------------------------------------------------------------------- #
    # Wrap the objective and inner-loop optimizer into a `TheseusLayer`.
    # Inner-loop optimizer here is the Levenberg-Marquardt nonlinear optimizer
    # coupled with a dense linear solver based on Cholesky decomposition.
    nl_optimizer = th.LevenbergMarquardt(
        objective, th.CholeskyDenseSolver, max_iterations=3
    )
    theseus_layer = th.TheseusLayer(nl_optimizer)
    theseus_layer.to(device=device, dtype=torch.double)

    # -------------------------------------------------------------------- #
    # Creating parameters to learn
    # -------------------------------------------------------------------- #
    # Use theseus_layer in an outer learning loop to learn different cost
    # function parameters:
    if cfg.train.mode == "weights_only":
        qsp_model = theg.TactileWeightModel(
            device, wt_init=torch.tensor([[50.0, 50.0, 50.0]])
        ).to(device)
        mf_between_model = theg.TactileWeightModel(
            device, wt_init=torch.tensor([[0.0, 0.0, 10.0]])
        ).to(device)
        measurements_model = None

        learning_rate = 5
        learnable_params = list(qsp_model.parameters()) + list(
            mf_between_model.parameters()
        )
    elif cfg.train.mode == "weights_and_measurement_nn":
        qsp_model = theg.TactileWeightModel(
            device, wt_init=torch.tensor([[5.0, 5.0, 5.0]])
        ).to(device)
        mf_between_model = theg.TactileWeightModel(
            device, wt_init=torch.tensor([[0.0, 0.0, 5.0]])
        ).to(device)
        measurements_model = theg.TactileMeasModel(2 * 2 * 4, 4)
        if cfg.tactile_cost.init_pretrained_model is True:
            measurements_model = theg.init_tactile_model_from_file(
                model=measurements_model,
                filename=EXP_PATH / "models" / "transform_prediction_keypoints.pt",
            )
        measurements_model.to(device)

        learning_rate = 1e-3
        cfg.train.eps_tracking_loss = 5e-4  # early stopping
        learnable_params = (
            list(measurements_model.parameters())
            + list(qsp_model.parameters())
            + list(mf_between_model.parameters())
        )
    else:
        print("Learning mode {cfg.train.mode} not found")

    outer_optim = optim.Adam(learnable_params, lr=learning_rate)
    batch_size = cfg.train.batch_size
    num_batches = cfg.train.num_batches
    measurements = [
        (
            dataset["img_feats"][0:time_steps].unsqueeze(0).repeat(batch_size, 1, 1),
            dataset["eff_poses"][0:time_steps].unsqueeze(0).repeat(batch_size, 1, 1),
            dataset["obj_poses"][0:time_steps].unsqueeze(0).repeat(batch_size, 1, 1),
        )
        for _ in range(num_batches)
    ]

    obj_poses_gt = dataset["obj_poses"][0:time_steps, :].clone().requires_grad_(True)
    eff_poses_gt = dataset["eff_poses"][0:time_steps, :].clone().requires_grad_(True)

    theseus_inputs = {}
    for epoch in range(cfg.train.num_epochs):
        losses = []
        for batch_idx, batch in enumerate(measurements):
            theseus_inputs.update(
                theg.get_tactile_nn_measurements_inputs(
                    batch=batch,
                    device=device,
                    class_label=cfg.class_label,
                    num_classes=cfg.num_classes,
                    min_win_mf=min_win_mf,
                    max_win_mf=max_win_mf,
                    step_win_mf=step_win_mf,
                    time_steps=time_steps,
                    model=measurements_model,
                )
            )
            theseus_inputs.update(
                theg.get_tactile_motion_capture_inputs(batch, device, time_steps)
            )
            theseus_inputs.update(
                theg.get_tactile_cost_weight_inputs(qsp_model, mf_between_model)
            )
            theseus_inputs.update(
                theg.get_tactile_initial_optim_vars(batch, device, time_steps)
            )

            theseus_inputs["sdf_data"] = (
                (sdf_tensor.data).repeat(batch_size, 1, 1).to(device)
            )

            theseus_inputs, _ = theseus_layer.forward(theseus_inputs, verbose=True)

            obj_poses_opt = theg.get_tactile_poses_from_values(
                batch_size=batch_size,
                values=theseus_inputs,
                time_steps=time_steps,
                device=device,
                key="obj_pose",
            )
            eff_poses_opt = theg.get_tactile_poses_from_values(
                batch_size=batch_size,
                values=theseus_inputs,
                time_steps=time_steps,
                device=device,
                key="eff_pose",
            )

            loss = F.mse_loss(obj_poses_opt[batch_idx, :], obj_poses_gt)
            loss.backward()

            nn.utils.clip_grad_norm_(qsp_model.parameters(), 100, norm_type=2)
            nn.utils.clip_grad_norm_(mf_between_model.parameters(), 100, norm_type=2)

            with torch.no_grad():
                for name, param in qsp_model.named_parameters():
                    print(name, param.data)
                for name, param in mf_between_model.named_parameters():
                    print(name, param.data)

                print("    grad qsp", qsp_model.param.grad.norm().item())
                print("    grad mfb", mf_between_model.param.grad.norm().item())

            outer_optim.step()

            with torch.no_grad():
                for param in qsp_model.parameters():
                    param.data.clamp_(0)
                for param in mf_between_model.parameters():
                    param.data.clamp_(0)

            losses.append(loss.item())

        if cfg.options.vis_traj:
            theg.visualize_tactile_push2d(
                obj_poses=obj_poses_opt[0, :],
                eff_poses=eff_poses_opt[0, :],
                obj_poses_gt=obj_poses_gt,
                eff_poses_gt=eff_poses_gt,
                rect_len_x=cfg.shape.rect_len_x,
                rect_len_y=cfg.shape.rect_len_y,
            )

        print(f"AVG. LOSS: {np.mean(losses)}")

        if np.mean(losses) < cfg.train.eps_tracking_loss:
            break


@hydra.main(config_path="./configs/", config_name="tactile_pose_estimation")
def main(cfg):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_learning_loop(cfg)


if __name__ == "__main__":
    main()
