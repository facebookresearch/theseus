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
    dataset_path = EXP_PATH / "datasets" / f"{cfg.dataset_name}.json"
    sdf_path = EXP_PATH / "sdfs" / f"{cfg.sdf_name}.json"
    dataset = theg.TactilePushingDataset(dataset_path, sdf_path, cfg.episode, device)

    # -------------------------------------------------------------------- #
    # Create pose estimator (which wraps a TheseusLayer)
    # -------------------------------------------------------------------- #
    pose_estimator = theg.TactilePoseEstimator(
        dataset,
        max_steps=cfg.max_steps,
        min_window_moving_frame=cfg.tactile_cost.min_win_mf,
        max_window_moving_frame=cfg.tactile_cost.max_win_mf,
        step_window_moving_frame=cfg.tactile_cost.step_win_mf,
        rectangle_shape=(cfg.shape.rect_len_x, cfg.shape.rect_len_y),
        device=device,
    )
    time_steps = pose_estimator.time_steps

    # -------------------------------------------------------------------- #
    # Creating parameters to learn
    # -------------------------------------------------------------------- #
    if cfg.tactile_cost.init_pretrained_model is True:
        measurements_model_path = (
            EXP_PATH / "models" / "transform_prediction_keypoints.pt"
        )
    else:
        measurements_model_path = None
    (
        measurements_model,
        qsp_model,
        mf_between_model,
        learnable_params,
        hyperparameters,
    ) = theg.create_tactile_models(
        cfg.train.mode, device, measurements_model_path=measurements_model_path
    )
    if "eps_tracking_loss" in hyperparameters:
        cfg.train.eps_tracking_loss = 5e-4  # early stopping
    outer_optim = optim.Adam(learnable_params, lr=hyperparameters["learning_rate"])

    # -------------------------------------------------------------------- #
    # Main learning loop
    # -------------------------------------------------------------------- #
    # Use theseus_layer in an outer learning loop to learn different cost
    # function parameters:
    measurements = dataset.get_measurements(
        cfg.train.batch_size, cfg.train.num_batches, time_steps
    )
    obj_poses_gt = dataset.obj_poses[0:time_steps, :].clone().requires_grad_(True)
    eff_poses_gt = dataset.eff_poses[0:time_steps, :].clone().requires_grad_(True)
    theseus_inputs = {}
    for _ in range(cfg.train.num_epochs):
        losses = []
        for batch_idx, batch in enumerate(measurements):
            theg.update_tactile_pushing_inputs(
                dataset=dataset,
                batch=batch,
                measurements_model=measurements_model,
                qsp_model=qsp_model,
                mf_between_model=mf_between_model,
                device=device,
                cfg=cfg,
                time_steps=time_steps,
                theseus_inputs=theseus_inputs,
            )

            theseus_inputs, _ = pose_estimator.forward(
                theseus_inputs, optimizer_kwargs={"verbose": True}
            )

            obj_poses_opt, eff_poses_opt = theg.get_tactile_poses_from_values(
                values=theseus_inputs, time_steps=time_steps
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
