# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pathlib
import random
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
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


class TactilePushingTrainer:
    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg
        dataset_path = EXP_PATH / "datasets" / f"{cfg.dataset_name}.json"
        sdf_path = EXP_PATH / "sdfs" / f"{cfg.sdf_name}.json"
        self.dataset_train = theg.TactilePushingDataset(
            str(dataset_path),
            str(sdf_path),
            cfg.episode_length,
            cfg.train.batch_size,
            cfg.max_episodes,
            cfg.max_steps,
            device,
            split_episodes=cfg.split_episodes,
            data_mode="train",
        )
        self.dataset_val = theg.TactilePushingDataset(
            str(dataset_path),
            str(sdf_path),
            cfg.episode_length,
            cfg.train.batch_size,
            cfg.max_episodes,
            cfg.max_steps,
            device,
            split_episodes=cfg.split_episodes,
            data_mode="val",
        )

        # -------------------------------------------------------------------- #
        # Create pose estimator (which wraps a TheseusLayer)
        # -------------------------------------------------------------------- #
        # self.dataset_train is used to created the correct SDF tensor shapes
        # and also get the number of time steps
        self.pose_estimator = theg.TactilePoseEstimator(
            self.dataset_train,
            min_window_moving_frame=cfg.tactile_cost.min_win_mf,
            max_window_moving_frame=cfg.tactile_cost.max_win_mf,
            step_window_moving_frame=cfg.tactile_cost.step_win_mf,
            rectangle_shape=(cfg.shape.rect_len_x, cfg.shape.rect_len_y),
            device=device,
            optimizer_cls=getattr(th, cfg.inner_optim.optimizer),
            max_iterations=cfg.inner_optim.max_iters,
            step_size=cfg.inner_optim.step_size,
            regularization_w=cfg.inner_optim.reg_w,
            force_max_iters=cfg.inner_optim.force_max_iters,
        )

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
            self.measurements_model,
            self.qsp_model,
            self.mf_between_model,
            learnable_params,
        ) = theg.create_tactile_models(
            cfg.train.mode, device, measurements_model_path=measurements_model_path
        )
        self.eps_tracking_loss = cfg.train.eps_tracking_loss
        self.outer_optim = optim.Adam(learnable_params, lr=cfg.train.lr)

    def get_batch_data(
        self,
        batch: Dict[str, torch.Tensor],
        dataset: theg.TactilePushingDataset,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # Initialize inputs dictionary
        theseus_inputs = self.pose_estimator.get_start_pose_and_motion_capture_dict(
            batch
        )
        # Update the above dictionary with measurement factor data
        theg.update_tactile_pushing_inputs(
            dataset=dataset,
            batch=batch,
            measurements_model=self.measurements_model,
            qsp_model=self.qsp_model,
            mf_between_model=self.mf_between_model,
            device=device,
            cfg=self.cfg,
            theseus_inputs=theseus_inputs,
        )
        # Get ground truth data to use for the outer loss
        obj_poses_gt = batch["obj_poses_gt"]
        eff_poses_gt = batch["eff_poses_gt"]
        return theseus_inputs, obj_poses_gt, eff_poses_gt

    def _update(self, loss: torch.Tensor) -> float:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        loss.backward()
        end_event.record()
        torch.cuda.synchronize()
        backward_time = start_event.elapsed_time(end_event)
        logger.info(f"Backward pass took {backward_time} ms.")

        nn.utils.clip_grad_norm_(self.qsp_model.parameters(), 100, norm_type=2)
        nn.utils.clip_grad_norm_(self.mf_between_model.parameters(), 100, norm_type=2)
        nn.utils.clip_grad_norm_(self.measurements_model.parameters(), 100, norm_type=2)

        with torch.no_grad():
            for name, param in self.qsp_model.named_parameters():
                logger.info(f"{name} {param.data}")
            for name, param in self.mf_between_model.named_parameters():
                logger.info(f"{name} {param.data}")

            def _print_grad(msg_, param_):
                logger.info(f"{msg_} {param_.grad.norm().item()}")

            _print_grad("    grad qsp", self.qsp_model.param)
            _print_grad("    grad mfb", self.mf_between_model.param)
            _print_grad("    grad nn_weight", self.measurements_model.fc1.weight)
            _print_grad("    grad nn_bias", self.measurements_model.fc1.bias)

        self.outer_optim.step()

        with torch.no_grad():
            for param in self.qsp_model.parameters():
                param.data.clamp_(0)
            for param in self.mf_between_model.parameters():
                param.data.clamp_(0)

        return backward_time

    def compute_loss(
        self, update: bool = True
    ) -> Tuple[
        List[torch.Tensor], Dict[int, Dict[str, Any]], Dict[str, List[torch.Tensor]]
    ]:
        if update:
            dataset = self.dataset_train
        else:
            dataset = self.dataset_val

        results = {}
        losses = []
        image_data: Dict[str, List[torch.Tensor]] = dict(
            (name, []) for name in ["obj_opt", "eff_opt", "obj_gt", "eff_gt"]
        )

        # Set different number of max iterations for validation loop
        self.pose_estimator.theseus_layer.optimizer.set_params(  # type: ignore
            max_iterations=self.cfg.inner_optim.max_iters
            if update
            else self.cfg.inner_optim.val_iters
        )
        for batch_idx in range(dataset.num_batches):
            # ---------- Read data from batch ----------- #
            batch = dataset.get_batch(batch_idx)
            theseus_inputs, obj_poses_gt, eff_poses_gt = self.get_batch_data(
                batch, dataset, device
            )

            # ---------- Forward pass ----------- #
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            theseus_outputs, info = self.pose_estimator.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "verbose": True,
                    "track_err_history": True,
                    "backward_mode": getattr(
                        th.BackwardMode, self.cfg.inner_optim.backward_mode
                    ),
                    "__keep_final_step_size__": self.cfg.inner_optim.keep_step_size,
                },
            )
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event)
            logger.info(f"Forward pass took {forward_time} ms.")

            # ---------- Backward pass and update ----------- #
            obj_poses_opt, eff_poses_opt = theg.get_tactile_poses_from_values(
                values=theseus_outputs, time_steps=dataset.time_steps
            )
            se2_opt = th.SE2(x_y_theta=obj_poses_opt.view(-1, 3))
            se2_gt = th.SE2(x_y_theta=obj_poses_gt.view(-1, 3))
            loss = se2_opt.local(se2_gt).norm()

            backward_time = -1.0
            if update:
                backward_time = self._update(loss)

            # ---------- Pack results ----------- #
            losses.append(loss.item())
            results[batch_idx] = pack_batch_results(
                theseus_outputs,
                self.qsp_model.state_dict(),
                self.mf_between_model.state_dict(),
                self.measurements_model.state_dict(),
                info,
                loss.item(),
                forward_time,
                backward_time,
            )
            image_data["obj_opt"].extend([p for p in obj_poses_opt])
            image_data["eff_opt"].extend([p for p in eff_poses_opt])
            image_data["obj_gt"].extend([p for p in obj_poses_gt])
            image_data["eff_gt"].extend([p for p in eff_poses_gt])

        return losses, results, image_data


def pack_batch_results(
    theseus_outputs: Dict[str, torch.Tensor],
    qsp_state_dict: Dict[str, torch.Tensor],
    mfb_state_dict: Dict[str, torch.Tensor],
    meas_state_dict: Dict[str, torch.Tensor],
    info: th.optimizer.OptimizerInfo,
    loss_value: float,
    forward_time: float,
    backward_time: float,
) -> Dict[str, Any]:
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
    logger.info(f"LOGGING TO {str(root_path)}")
    trainer = TactilePushingTrainer(cfg)

    # -------------------------------------------------------------------- #
    # Main learning loop
    # -------------------------------------------------------------------- #
    # Use theseus_layer in an outer learning loop to learn different cost
    # function parameters:
    results_train = {}
    results_val = {}
    for epoch in range(cfg.train.num_epochs):
        logger.info(f" ********************* EPOCH {epoch} *********************")
        logger.info(" -------------- TRAINING --------------")
        train_losses, results_train[epoch], _ = trainer.compute_loss()
        logger.info(f"AVG. TRAIN LOSS: {np.mean(train_losses)}")
        torch.save(results_train, root_path / "results_train.pt")

        logger.info(" -------------- VALIDATION --------------")
        val_losses, results_val[epoch], image_data = trainer.compute_loss(update=False)
        logger.info(f"AVG. VAL LOSS: {np.mean(val_losses)}")
        torch.save(results_val, root_path / "results_val.pt")

        if cfg.options.vis_traj:
            for i in range(len(image_data["obj_opt"])):
                save_dir = root_path / f"img_{i}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_fname = save_dir / f"epoch{epoch}.png"
                theg.visualize_tactile_push2d(
                    obj_poses=image_data["obj_opt"][i],
                    eff_poses=image_data["eff_opt"][i],
                    obj_poses_gt=image_data["obj_gt"][i],
                    eff_poses_gt=image_data["eff_gt"][i],
                    rect_len_x=cfg.shape.rect_len_x,
                    rect_len_y=cfg.shape.rect_len_y,
                    save_fname=save_fname,
                )


@hydra.main(config_path="./configs/", config_name="tactile_pose_estimation")
def main(cfg):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_learning_loop(cfg)


if __name__ == "__main__":
    main()
