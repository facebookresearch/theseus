# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pathlib
import random
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import theseus as th
import theseus.utils.examples as theg

# Logger
logger = logging.getLogger(__name__)

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


def pack_batch_results(
    theseus_outputs: Dict[str, torch.Tensor],
    qsp_state_dict: torch.Tensor,
    mfb_state_dict: torch.Tensor,
    meas_state_dict: torch.Tensor,
    info: th.optimizer.OptimizerInfo,
    loss_value: float,
    forward_time: float,
    backward_time: float,
) -> Dict:
    def _clone(t_):
        return t_.detach().cpu().clone()

    return {
        "theseus_outputs": dict((s, _clone(t)) for s, t in theseus_outputs.items()),
        "qsp_state_dict": qsp_state_dict,
        "mfb_state_dict": mfb_state_dict,
        "meas_state_dict": meas_state_dict,
        "err_history": info.err_history,  # type: ignore
        "loss": loss_value,
        "forward_time": forward_time,
        "backward_time": backward_time,
    }


def run_learning_loop(cfg):
    root_path = pathlib.Path(os.getcwd())
    dataset_path = EXP_PATH / "datasets" / f"{cfg.dataset_name}.json"
    sdf_path = EXP_PATH / "sdfs" / f"{cfg.sdf_name}.json"
    dataset = theg.TactilePushingDataset(
        dataset_path,
        sdf_path,
        cfg.episode_length,
        cfg.train.batch_size,
        cfg.max_episodes,
        device,
    )

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
        optimizer_cls=getattr(th, cfg.inner_optim.optimizer),
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
        regularization_w=cfg.inner_optim.reg_w,
    )
    time_steps = pose_estimator.time_steps

    # -------------------------------------------------------------------- #
    # Creating parameters to learn
    # -------------------------------------------------------------------- #
    if cfg.tactile_cost.init_pretrained_model:
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
    ) = theg.create_tactile_models(
        cfg.train.mode, device, measurements_model_path=measurements_model_path
    )
    eps_tracking_loss = cfg.train.eps_tracking_loss
    outer_optim = optim.Adam(learnable_params, lr=cfg.train.lr)

    # -------------------------------------------------------------------- #
    # Main learning loop
    # -------------------------------------------------------------------- #
    # Use theseus_layer in an outer learning loop to learn different cost
    # function parameters:
    measurements = dataset.get_measurements(time_steps)
    results = {}
    for epoch in range(cfg.train.num_epochs):
        results[epoch] = {}
        logger.info(f" ********************* EPOCH f{epoch} *********************")
        losses = []
        image_idx = 0
        for batch_idx, batch in enumerate(measurements):
            pose_and_motion_batch = dataset.get_start_pose_and_motion_for_batch(
                batch_idx, time_steps
            )  # x_y_theta format
            pose_estimator.update_start_pose_and_motion_from_batch(
                pose_and_motion_batch
            )
            theseus_inputs = {}
            # Updates the above with measurement factor data
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

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            theseus_outputs, info = pose_estimator.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "verbose": True,
                    "track_err_history": True,
                    "backward_mode": getattr(
                        th.BackwardMode, cfg.inner_optim.backward_mode
                    ),
                    "__keep_final_step_size__": cfg.inner_optim.keep_step_size,
                },
            )
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event)
            logger.info(f"Forward pass took {forward_time} ms.")

            obj_poses_opt, eff_poses_opt = theg.get_tactile_poses_from_values(
                values=theseus_outputs, time_steps=time_steps
            )
            obj_poses_gt, eff_poses_gt = dataset.get_gt_data_for_batch(
                batch_idx, time_steps
            )

            se2_opt = th.SE2(x_y_theta=obj_poses_opt.view(-1, 3))
            se2_gt = th.SE2(x_y_theta=obj_poses_gt.view(-1, 3))
            loss = se2_opt.local(se2_gt).norm()
            start_event.record()
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            backward_time = start_event.elapsed_time(end_event)
            logger.info(f"Backward pass took {backward_time} ms.")

            nn.utils.clip_grad_norm_(qsp_model.parameters(), 100, norm_type=2)
            nn.utils.clip_grad_norm_(mf_between_model.parameters(), 100, norm_type=2)
            nn.utils.clip_grad_norm_(measurements_model.parameters(), 100, norm_type=2)

            with torch.no_grad():
                for name, param in qsp_model.named_parameters():
                    logger.info(f"{name} {param.data}")
                for name, param in mf_between_model.named_parameters():
                    logger.info(f"{name} {param.data}")

                def _print_grad(msg_, param_):
                    logger.info(f"{msg_} {param_.grad.norm().item()}")

                _print_grad("    grad qsp", qsp_model.param)
                _print_grad("    grad mfb", mf_between_model.param)
                _print_grad("    grad nn_weight", measurements_model.fc1.weight)
                _print_grad("    grad nn_bias", measurements_model.fc1.bias)

            outer_optim.step()

            with torch.no_grad():
                for param in qsp_model.parameters():
                    param.data.clamp_(0)
                for param in mf_between_model.parameters():
                    param.data.clamp_(0)

            losses.append(loss.item())

            if cfg.save_all:
                results[epoch][batch_idx] = pack_batch_results(
                    theseus_outputs,
                    qsp_model.state_dict(),
                    mf_between_model.state_dict(),
                    measurements_model.state_dict(),
                    info,
                    loss.item(),
                    forward_time,
                    backward_time,
                )
                torch.save(results, root_path / "results.pt")

            if cfg.options.vis_traj:
                for i in range(len(obj_poses_gt)):
                    save_dir = root_path / f"img_{image_idx}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_fname = save_dir / f"epoch{epoch}.png"
                    theg.visualize_tactile_push2d(
                        obj_poses=obj_poses_opt[i],
                        eff_poses=eff_poses_opt[i],
                        obj_poses_gt=obj_poses_gt[i],
                        eff_poses_gt=eff_poses_gt[i],
                        rect_len_x=cfg.shape.rect_len_x,
                        rect_len_y=cfg.shape.rect_len_y,
                        save_fname=save_fname,
                    )
                    image_idx += 1

        logger.info(f"AVG. LOSS: {np.mean(losses)}")

        if np.mean(losses) < eps_tracking_loss:
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
