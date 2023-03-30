# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import random
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

import theseus as th
import theseus.utils.examples as theg

# To run this example, you will need a motion planning dataset available at
# https://dl.fbaipublicfiles.com/theseus/motion_planning_dataset.tar.gz
#
# The steps below should let you run the example.
# From the root project folder do:
#   mkdir expts
#   cd expts
#   wget https://dl.fbaipublicfiles.com/theseus/motion_planning_data.tar.gz
#   tar -xzvf motion_planning_data.tar.gz
#   cd ..
#   python examples/motion_planning_2d.py
DATASET_DIR = pathlib.Path.cwd() / "expts" / "motion-planning-2d" / "dataset"


def plot_and_save_trajectories(
    epoch: int,
    batch: Dict[str, torch.Tensor],
    initial_trajectory: torch.Tensor,
    solution_trajectory: torch.Tensor,
    robot_radius: float,
    include_expert: bool = False,
):
    sdf = th.eb.SignedDistanceField2D(
        th.Point2(batch["sdf_origin"]),
        th.Variable(batch["cell_size"]),
        th.Variable(batch["sdf_data"]),
    )

    trajectories = [initial_trajectory.cpu(), solution_trajectory.cpu()]
    if include_expert:
        trajectories.append(batch["expert_trajectory"])
    figures = theg.generate_trajectory_figs(
        batch["map_tensor"], sdf, trajectories, robot_radius=robot_radius
    )

    dir_name = pathlib.Path.cwd() / f"output_maps/{epoch:02d}"
    dir_name.mkdir(exist_ok=True, parents=True)
    for fig_idx, fig in enumerate(figures):
        map_id = batch["file_id"][fig_idx]
        fig.savefig(dir_name / f"output_{map_id}_im.png")
        plt.close(fig)


def run_learning_loop(cfg):
    dataset = theg.TrajectoryDataset(
        True, cfg.train_data_size, DATASET_DIR, cfg.map_type
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, cfg.batch_size, shuffle=cfg.shuffle_each_epoch
    )

    motion_planner = theg.MotionPlanner(
        cfg.optim_params.init,
        map_size=cfg.img_size,
        epsilon_dist=cfg.obs_params.safety_dist + cfg.robot_radius,
        total_time=cfg.total_time,
        collision_weight=cfg.obs_params.weight,
        Qc_inv=cfg.gp_params.Qc_inv,
        num_time_steps=cfg.num_time_steps,
        use_single_collision_weight=True,
        device=cfg.device,
    )

    if cfg.model_type == "scalar_collision_weight":
        learnable_model = theg.ScalarCollisionWeightModel()
    elif cfg.model_type == "scalar_collision_weight_and_cost_eps":
        learnable_model = theg.ScalarCollisionWeightAndCostEpstModel(cfg.robot_radius)
    elif cfg.model_type == "initial_trajectory_model":
        learnable_model = theg.InitialTrajectoryModel(motion_planner)

    learnable_model.to(cfg.device)
    model_optimizer = torch.optim.Adam(
        learnable_model.parameters(),
        lr=cfg.model_lr,
        amsgrad=cfg.amsgrad,
        weight_decay=cfg.model_wd,
    )

    if not cfg.do_learning:
        cfg.num_epochs = 1

    for epoch in range(cfg.num_epochs):
        print(f"Starting epoch {epoch}")
        epoch_total_loss = 0
        epoch_mean_objective_loss = 0
        epoch_gp_loss = 0
        epoch_collision_loss = 0
        epoch_imitation_loss = 0
        epoch_grad_norm = 0
        epoch_params_norm = 0
        for batch in data_loader:
            model_optimizer.zero_grad()

            start = batch["expert_trajectory"][:, :2, 0]
            goal = batch["expert_trajectory"][:, :2, -1]
            planner_inputs = {
                "sdf_origin": batch["sdf_origin"].to(cfg.device),
                "start": start.to(cfg.device),
                "goal": goal.to(cfg.device),
                "cell_size": batch["cell_size"].to(cfg.device),
                "sdf_data": batch["sdf_data"].to(cfg.device),
            }
            motion_planner.objective.update(planner_inputs)

            if not cfg.do_learning or cfg.model_type != "initial_trajectory_model":
                # This method updates the dictionary so that optimization variables
                # are initialized with a straight line trajectory
                planner_inputs.update(
                    motion_planner.get_variable_values_from_straight_line(
                        planner_inputs["start"], planner_inputs["goal"]
                    )
                )

            # This returns a dictionary of auxiliary variable names to torch tensors,
            # with values learned by a model upstream of the motion planner layer
            if cfg.do_learning:
                learnable_inputs = learnable_model.forward(batch)
                planner_inputs.update(learnable_inputs)

            # Get the initial trajectory (to use for logging)
            motion_planner.objective.update(planner_inputs)
            initial_trajectory = motion_planner.get_trajectory()
            with torch.no_grad():
                batch_error = motion_planner.objective.error_metric().mean()
                print(f"Planner MSE optim first: {batch_error.item() : 10.2f}")

            _, info = motion_planner.layer.forward(
                planner_inputs,
                optimizer_kwargs={
                    **{
                        "track_best_solution": True,
                        "verbose": cfg.verbose,
                    },
                    **cfg.optim_params.kwargs,
                },
            )
            if cfg.do_learning and cfg.include_imitation_loss:
                solution_trajectory = motion_planner.get_trajectory()
            else:
                solution_trajectory = motion_planner.get_trajectory(
                    values_dict=info.best_solution
                )

            with torch.no_grad():
                batch_error = motion_planner.objective.error_metric().mean()
                print(f"Planner MSE optim final: {batch_error.item() : 10.2f}")

            if cfg.do_learning:
                gp_error, collision_error = motion_planner.get_total_squared_errors()
                loss = 0
                if cfg.use_mean_objective_as_loss:
                    loss = motion_planner.objective.error_metric().mean()
                    loss /= motion_planner.objective.dim()
                    loss *= cfg.obj_loss_weight
                    epoch_mean_objective_loss += loss.item()
                else:
                    loss = (
                        cfg.gp_loss_weight * gp_error
                        + cfg.collision_loss_weight * collision_error
                    )
                if cfg.include_imitation_loss:
                    imitation_loss = F.mse_loss(
                        solution_trajectory, batch["expert_trajectory"].to(cfg.device)
                    )
                    loss += imitation_loss
                    epoch_imitation_loss += imitation_loss.item()
                loss.backward()

                with torch.no_grad():
                    grad_norm = 0.0
                    params_norm = 0.0
                    for p in learnable_model.parameters():
                        grad_norm += p.grad.data.norm(2).item()
                        params_norm += p.data.norm(float("inf")).item()
                    epoch_grad_norm += grad_norm / cfg.num_epochs
                    epoch_params_norm += params_norm / cfg.num_epochs
                if cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        learnable_model.parameters(), cfg.clip_grad_norm
                    )
                model_optimizer.step()

                epoch_total_loss += loss.item()
                epoch_gp_loss += gp_error.item()
                epoch_collision_loss += collision_error.item()

            if cfg.plot_trajectories:
                plot_and_save_trajectories(
                    epoch,
                    batch,
                    initial_trajectory.cpu().detach().clone(),
                    solution_trajectory.cpu().detach().clone(),
                    cfg.robot_radius,
                    include_expert=cfg.include_imitation_loss,
                )

        collision_w = (
            motion_planner.objective.aux_vars["collision_w"].tensor.mean().item()
        )
        cost_eps = motion_planner.objective.aux_vars["cost_eps"].tensor.mean().item()
        print("collision weight", collision_w)
        print("cost_eps", cost_eps)
        print("OBJECTIVE MEAN LOSS", epoch_mean_objective_loss)
        print("GP LOSS: ", epoch_gp_loss)
        print("COLLISION LOSS: ", epoch_collision_loss)
        print("IMITATION LOSS: ", epoch_imitation_loss)
        print("TOTAL LOSS: ", epoch_total_loss)
        print(f" ---------------- END EPOCH {epoch} -------------- ")


@hydra.main(config_path="./configs/", config_name="motion_planning_2d")
def main(cfg):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_learning_loop(cfg)


if __name__ == "__main__":
    main()
