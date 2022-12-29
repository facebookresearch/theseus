import collections
import pathlib
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import omegaconf
import torch
import torch.nn as nn

import theseus as th

from .misc import TactilePushingDataset


class TactileMeasModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, k: torch.Tensor):
        x = torch.cat([x1, x2], dim=1)

        k1_ = k.unsqueeze(1)  # b x 1 x cl
        x1_ = x.unsqueeze(-1)  # b x dim x 1
        x = torch.mul(x1_, k1_)  # b x dim x cl

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x


def init_tactile_model_from_file(model: nn.Module, filename: pathlib.Path):
    model_saved = torch.jit.load(filename)
    state_dict_saved = model_saved.state_dict()
    state_dict_new = collections.OrderedDict()
    state_dict_new["fc1.weight"] = state_dict_saved["model.fc1.weight"]
    state_dict_new["fc1.bias"] = state_dict_saved["model.fc1.bias"]

    model.load_state_dict(state_dict_new)

    return model


# Set some parameters for the cost weights
class TactileWeightModel(nn.Module):
    def __init__(
        self,
        device: th.DeviceType,
        dim: int = 3,
        wt_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        wt_init_ = torch.rand(1, dim)
        if wt_init is not None:
            wt_init_ = wt_init
        self.param = nn.Parameter(wt_init_.to(device))
        self.to(device)

    def forward(self):
        return self.param.clone()


def create_tactile_models(
    model_type: str,
    device: th.DeviceType,
    measurements_model_path: Optional[pathlib.Path] = None,
) -> Tuple[
    Optional[TactileMeasModel],
    TactileWeightModel,
    TactileWeightModel,
    List[nn.Parameter],
]:
    if model_type == "weights_only":
        qsp_model = TactileWeightModel(
            device, wt_init=torch.tensor([[50.0, 50.0, 50.0]])
        )
        mf_between_model = TactileWeightModel(
            device, wt_init=torch.tensor([[0.0, 0.0, 10.0]])
        )
        measurements_model = None

        learnable_params = list(qsp_model.parameters()) + list(
            mf_between_model.parameters()
        )
    elif model_type == "weights_and_measurement_nn":
        qsp_model = TactileWeightModel(device, wt_init=torch.tensor([[5.0, 5.0, 5.0]]))
        mf_between_model = TactileWeightModel(
            device, wt_init=torch.tensor([[0.0, 0.0, 5.0]])
        )
        measurements_model = TactileMeasModel(2 * 2 * 4, 4)
        if measurements_model_path is not None:
            measurements_model = init_tactile_model_from_file(
                model=measurements_model,
                filename=measurements_model_path,
            )
        measurements_model.to(device)

        learnable_params = (
            list(measurements_model.parameters())
            + list(qsp_model.parameters())
            + list(mf_between_model.parameters())
        )
    else:
        print("Learning mode {cfg.train.mode} not found")

    return (
        measurements_model,
        qsp_model,
        mf_between_model,
        learnable_params,
    )


# ----------------------------------------------------------------------------------- #
# ------------------------------- Theseus Model Interface --------------------------- #
# ----------------------------------------------------------------------------------- #


def get_tactile_nn_measurements_inputs(
    batch: Dict[str, torch.Tensor],
    device: th.DeviceType,
    class_label: int,
    num_classes: int,
    min_win_mf: int,
    max_win_mf: int,
    step_win_mf: int,
    time_steps: int,
    model: Optional[nn.Module] = None,
):
    inputs = {}

    if model is not None:
        images_feat_meas = batch["img_feats"].to(device)
        class_label_vec = (
            nn.functional.one_hot(torch.tensor(class_label), torch.tensor(num_classes))
            .view(1, -1)
            .to(device)
        )

        meas_model_input_1_list: List[torch.Tensor] = []
        meas_model_input_2_list: List[torch.Tensor] = []
        for i in range(min_win_mf, time_steps):
            for offset in range(min_win_mf, np.minimum(i, max_win_mf), step_win_mf):
                meas_model_input_1_list.append(images_feat_meas[:, i - offset, :])
                meas_model_input_2_list.append(images_feat_meas[:, i, :])

        meas_model_input_1 = torch.cat(meas_model_input_1_list, dim=0)
        meas_model_input_2 = torch.cat(meas_model_input_2_list, dim=0)
        num_measurements = meas_model_input_1.shape[0]
        model_measurements = model.forward(
            meas_model_input_1, meas_model_input_2, class_label_vec
        ).reshape(
            -1, num_measurements, 4
        )  # data format (x, y, cos, sin)
    else:  # use oracle model
        eff_poses = batch["eff_poses"]
        obj_poses = batch["obj_poses"]
        model_measurements = []
        for i in range(min_win_mf, time_steps):
            for offset in range(min_win_mf, np.minimum(i, max_win_mf), step_win_mf):
                eff_pose_1 = th.SE2(x_y_theta=eff_poses[:, i - offset])
                obj_pose_1 = th.SE2(x_y_theta=obj_poses[:, i - offset])
                eff_pose_1__obj = obj_pose_1.between(eff_pose_1)

                eff_pose_2 = th.SE2(x_y_theta=eff_poses[:, i])
                obj_pose_2 = th.SE2(x_y_theta=obj_poses[:, i])
                eff_pose_2__obj = obj_pose_2.between(eff_pose_2)

                meas_pose_rel = cast(th.SE2, eff_pose_1__obj.between(eff_pose_2__obj))
                model_measurements.append(
                    torch.cat(
                        (
                            meas_pose_rel.xy().tensor,
                            meas_pose_rel.theta().cos(),
                            meas_pose_rel.theta().sin(),
                        ),
                        dim=1,
                    )
                )  # data format (x, y, cos, sin)

        num_measurements = len(model_measurements)
        model_measurements = torch.cat(model_measurements, dim=0).reshape(
            -1, num_measurements, 4
        )
        model_measurements = model_measurements.to(device)

    # set MovingFrameBetween measurements from the NN output
    nn_meas_idx = 0
    for i in range(min_win_mf, time_steps):
        for offset in range(min_win_mf, np.minimum(i, max_win_mf), step_win_mf):
            meas_xycs_ = torch.stack(
                [
                    model_measurements[:, nn_meas_idx, 0],
                    model_measurements[:, nn_meas_idx, 1],
                    model_measurements[:, nn_meas_idx, 2],
                    model_measurements[:, nn_meas_idx, 3],
                ],
                dim=1,
            )
            inputs[f"nn_measurement_{i-offset}_{i}"] = meas_xycs_
            nn_meas_idx = nn_meas_idx + 1

    return inputs


def get_tactile_motion_capture_inputs(
    batch: Dict[str, torch.Tensor], device: th.DeviceType, time_steps: int
):
    inputs = {}
    captures = batch["eff_poses"].to(device)
    for step in range(time_steps):
        capture = captures[:, step, :]
        cature_xycs = torch.stack(
            [capture[:, 0], capture[:, 1], capture[:, 2].cos(), capture[:, 2].sin()],
            dim=1,
        )
        inputs[f"motion_capture_{step}"] = cature_xycs
    return inputs


def get_tactile_cost_weight_inputs(qsp_model, mf_between_model):
    return {"qsp_weight": qsp_model(), "mf_between_weight": mf_between_model()}


def get_tactile_initial_optim_vars(
    batch: Dict[str, torch.Tensor],
    device: th.DeviceType,
    time_steps: int,
):
    inputs = {}
    eff_captures = batch["eff_poses"].to(device)
    obj_captures = batch["obj_poses"].to(device)
    for step in range(time_steps):
        inputs[f"obj_pose_{step}"] = th.SE2(x_y_theta=obj_captures[:, 0].clone()).tensor
        inputs[f"eff_pose_{step}"] = th.SE2(x_y_theta=eff_captures[:, 0].clone()).tensor

    return inputs


def update_tactile_pushing_inputs(
    dataset: TactilePushingDataset,
    batch: Dict[str, torch.Tensor],
    measurements_model: nn.Module,
    qsp_model: nn.Module,
    mf_between_model: nn.Module,
    device: th.DeviceType,
    cfg: omegaconf.DictConfig,
    theseus_inputs: Dict[str, torch.Tensor],
):
    time_steps = dataset.time_steps
    theseus_inputs["sdf_data"] = (dataset.sdf_data_tensor).to(device)
    theseus_inputs["sdf_cell_size"] = dataset.sdf_cell_size.to(device)
    theseus_inputs["sdf_origin"] = dataset.sdf_origin.to(device)

    theseus_inputs.update(
        get_tactile_nn_measurements_inputs(
            batch=batch,
            device=device,
            class_label=cfg.class_label,
            num_classes=cfg.num_classes,
            min_win_mf=cfg.tactile_cost.min_win_mf,
            max_win_mf=cfg.tactile_cost.max_win_mf,
            step_win_mf=cfg.tactile_cost.step_win_mf,
            time_steps=time_steps,
            model=measurements_model,
        )
    )
    theseus_inputs.update(get_tactile_motion_capture_inputs(batch, device, time_steps))
    theseus_inputs.update(get_tactile_cost_weight_inputs(qsp_model, mf_between_model))
    theseus_inputs.update(get_tactile_initial_optim_vars(batch, device, time_steps))


def get_tactile_poses_from_values(
    values: Dict[str, torch.Tensor], time_steps
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = values["obj_pose_0"].shape[0]
    device = values["obj_pose_0"].device

    obj_poses = torch.empty(batch_size, time_steps, 3, device=device)
    eff_poses = torch.empty(batch_size, time_steps, 3, device=device)

    for key in ["obj", "eff"]:
        ret_tensor = obj_poses if key == "obj" else eff_poses
        for t_ in range(time_steps):
            ret_tensor[:, t_, :2] = values[f"{key}_pose_{t_}"][:, 0:2]
            ret_tensor[:, t_, 2] = torch.atan2(
                values[f"{key}_pose_{t_}"][:, 3], values[f"{key}_pose_{t_}"][:, 2]
            )
    return obj_poses, eff_poses
