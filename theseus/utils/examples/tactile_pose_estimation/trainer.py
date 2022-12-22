# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import pathlib
from typing import Any, Dict, List, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim

import theseus as th

from .misc import TactilePushingDataset
from .models import (
    create_tactile_models,
    get_tactile_poses_from_values,
    update_tactile_pushing_inputs,
)
from .pose_estimator import TactilePoseEstimator

# Logger
logger = logging.getLogger(__name__)


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
    def __init__(
        self, cfg: omegaconf.DictConfig, exp_path: pathlib.Path, device: th.DeviceType
    ):
        self.cfg = cfg
        self.device = device
        dataset_path = exp_path / "datasets" / f"{cfg.dataset_name}.json"
        sdf_path = exp_path / "sdfs" / f"{cfg.sdf_name}.json"
        self.dataset_train = TactilePushingDataset(
            str(dataset_path),
            str(sdf_path),
            cfg.episode_length,
            cfg.train.batch_size,
            cfg.max_episodes,
            cfg.max_steps,
            device,
            split_episodes=cfg.split_episodes,
            val_ratio=cfg.train.val_ratio,
            seed=cfg.seed,
            data_mode="train",
        )
        self.dataset_val = TactilePushingDataset(
            str(dataset_path),
            str(sdf_path),
            cfg.episode_length,
            cfg.train.batch_size,
            cfg.max_episodes,
            cfg.max_steps,
            device,
            split_episodes=cfg.split_episodes,
            val_ratio=cfg.train.val_ratio,
            seed=cfg.seed,
            data_mode="val",
        )

        # -------------------------------------------------------------------- #
        # Create pose estimator (which wraps a TheseusLayer)
        # -------------------------------------------------------------------- #
        # self.dataset_train is used to created the correct SDF tensor shapes
        # and also get the number of time steps
        self.pose_estimator = TactilePoseEstimator(
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
                exp_path / "models" / "transform_prediction_keypoints.pt"
            )
        else:
            measurements_model_path = None
        (
            self.measurements_model,
            self.qsp_model,
            self.mf_between_model,
            learnable_params,
        ) = create_tactile_models(
            cfg.train.mode, device, measurements_model_path=measurements_model_path
        )
        self.outer_optim = optim.Adam(learnable_params, lr=cfg.train.lr)

    def get_batch_data(
        self,
        batch: Dict[str, torch.Tensor],
        dataset: TactilePushingDataset,
        device: th.DeviceType,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # Initialize inputs dictionary
        theseus_inputs = self.pose_estimator.get_start_pose_and_motion_capture_dict(
            batch
        )
        # Update the above dictionary with measurement factor data
        update_tactile_pushing_inputs(
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

    def _update(self, loss: torch.Tensor) -> Tuple[float, float]:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        torch.cuda.reset_max_memory_allocated()
        loss.backward()
        end_event.record()
        torch.cuda.synchronize()
        backward_time = start_event.elapsed_time(end_event)
        backward_mem = torch.cuda.max_memory_allocated() / 1025 / 1024
        logger.info(f"Backward pass took {backward_time} ms.")
        logger.info(f"Backward pass used {backward_mem} MBs.")

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

        return backward_time, backward_mem

    def _resolve_backward_mode(self, epoch: int) -> th.BackwardMode:
        if epoch >= self.cfg.inner_optim.force_implicit_by_epoch - 1:  # 0-indexing
            logger.info("Forcing IMPLICIT backward mode.")
            return th.BackwardMode.IMPLICIT
        else:
            return self.cfg.inner_optim.backward_mode

    def compute_loss(
        self, epoch: int, update: bool = True
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
        inner_opt_cfg = self.cfg.inner_optim
        self.pose_estimator.theseus_layer.optimizer.set_params(  # type: ignore
            max_iterations=inner_opt_cfg.max_iters
            if update or inner_opt_cfg.val_iters < 1
            else inner_opt_cfg.val_iters
        )
        for batch_idx in range(dataset.num_batches):
            # ---------- Read data from batch ----------- #
            batch = dataset.get_batch(batch_idx)
            theseus_inputs, obj_poses_gt, eff_poses_gt = self.get_batch_data(
                batch, dataset, self.device
            )

            # ---------- Forward pass ----------- #
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            torch.cuda.reset_max_memory_allocated()
            theseus_outputs, info = self.pose_estimator.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "verbose": True,
                    "track_err_history": True,
                    "backward_mode": self._resolve_backward_mode(epoch),
                    "backward_num_iterations": inner_opt_cfg.backward_num_iterations,
                    "dlm_epsilon": inner_opt_cfg.dlm_epsilon,
                    "__keep_final_step_size__": inner_opt_cfg.keep_step_size,
                },
            )
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event)
            forward_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            logger.info(f"Forward pass took {forward_time} ms.")
            logger.info(f"Forward pass used {forward_mem} MBs.")

            # ---------- Backward pass and update ----------- #
            obj_poses_opt, eff_poses_opt = get_tactile_poses_from_values(
                values=theseus_outputs, time_steps=dataset.time_steps
            )
            se2_opt = th.SE2(x_y_theta=obj_poses_opt.view(-1, 3))
            se2_gt = th.SE2(x_y_theta=obj_poses_gt.view(-1, 3))
            loss = se2_opt.local(se2_gt).norm()

            backward_time, backward_mem = -1.0, -1.0
            if update:
                backward_time, backward_mem = self._update(loss)

            # ---------- Pack results ----------- #
            losses.append(loss.item())
            results[batch_idx] = TactilePushingTrainer.pack_batch_results(
                theseus_outputs,
                self.qsp_model.state_dict(),
                self.mf_between_model.state_dict(),
                self.measurements_model.state_dict(),
                info,
                loss.item(),
                forward_time,
                backward_time,
                forward_mem,
                backward_mem,
            )
            image_data["obj_opt"].extend([p for p in obj_poses_opt])
            image_data["eff_opt"].extend([p for p in eff_poses_opt])
            image_data["obj_gt"].extend([p for p in obj_poses_gt])
            image_data["eff_gt"].extend([p for p in eff_poses_gt])

        return losses, results, image_data

    @staticmethod
    def pack_batch_results(
        theseus_outputs: Dict[str, torch.Tensor],
        qsp_state_dict: Dict[str, torch.Tensor],
        mfb_state_dict: Dict[str, torch.Tensor],
        meas_state_dict: Dict[str, torch.Tensor],
        info: th.optimizer.OptimizerInfo,
        loss_value: float,
        forward_time: float,
        backward_time: float,
        forward_mem: float,
        backward_mem: float,
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
            "forward_mem": forward_mem,
            "backward_mem": backward_mem,
        }
