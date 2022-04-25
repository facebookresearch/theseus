# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Dict, List

import numpy as np
import omegaconf
import torch

import theseus as th
import theseus.utils.examples as theg
from theseus.optimizer.gbp import GaussianBeliefPropagation, synchronous_schedule

# Smaller values} result in error
th.SO3.SO3_EPS = 1e-6


def print_histogram(
    ba: theg.BundleAdjustmentDataset, var_dict: Dict[str, torch.Tensor], msg: str
):
    print(msg)
    histogram = theg.ba_histogram(
        cameras=[
            theg.Camera(
                th.SE3(data=var_dict[c.pose.name]),
                c.focal_length,
                c.calib_k1,
                c.calib_k2,
            )
            for c in ba.cameras
        ],
        points=[th.Point3(data=var_dict[pt.name]) for pt in ba.points],
        observations=ba.observations,
    )
    for line in histogram.split("\n"):
        print(line)


def camera_loss(
    ba: theg.BundleAdjustmentDataset, camera_pose_vars: List[th.LieGroup]
) -> torch.Tensor:
    loss: torch.Tensor = 0  # type:ignore
    for i in range(len(ba.cameras)):
        camera_loss = th.local(camera_pose_vars[i], ba.gt_cameras[i].pose).norm(dim=1)
        loss += camera_loss
    return loss


def run(cfg: omegaconf.OmegaConf):
    # create (or load) dataset
    ba = theg.BundleAdjustmentDataset.generate_synthetic(
        num_cameras=cfg["num_cameras"],
        num_points=cfg["num_points"],
        average_track_length=cfg["average_track_length"],
        track_locality=cfg["track_locality"],
        feat_random=1.5,
        outlier_feat_random=70,
    )
    # ba.save_to_file(results_path / "ba.txt", gt_path=results_path / "ba_gt.txt")

    # param that control transition from squared loss to huber
    radius_tensor = torch.tensor([1.0], dtype=torch.float64)
    log_loss_radius = th.Vector(data=radius_tensor, name="log_loss_radius")

    # Set up objective
    objective = th.Objective(dtype=torch.float64)

    for obs in ba.observations:
        cam = ba.cameras[obs.camera_index]
        cost_function = theg.Reprojection(
            camera_pose=cam.pose,
            world_point=ba.points[obs.point_index],
            focal_length=cam.focal_length,
            calib_k1=cam.calib_k1,
            calib_k2=cam.calib_k2,
            log_loss_radius=log_loss_radius,
            image_feature_point=obs.image_feature_point,
        )
        objective.add(cost_function)
    dtype = objective.dtype

    # Add regularization
    if cfg["inner_optim"]["regularize"]:
        zero_point3 = th.Point3(dtype=dtype, name="zero_point")
        identity_se3 = th.SE3(dtype=dtype, name="zero_se3")
        w = np.sqrt(cfg["inner_optim"]["reg_w"])
        damping_weight = th.ScaleCostWeight(w * torch.ones(1, dtype=dtype))
        for name, var in objective.optim_vars.items():
            target: th.Manifold
            if isinstance(var, th.SE3):
                target = identity_se3
            elif isinstance(var, th.Point3):
                target = zero_point3
            else:
                assert False
            objective.add(
                th.eb.VariableDifference(
                    var, damping_weight, target, name=f"reg_{name}"
                )
            )

    camera_pose_vars: List[th.LieGroup] = [
        objective.optim_vars[c.pose.name] for c in ba.cameras  # type: ignore
    ]
    if cfg["inner_optim"]["ratio_known_cameras"] > 0.0:
        w = 100.0
        camera_weight = th.ScaleCostWeight(100 * torch.ones(1, dtype=dtype))
        for i in range(len(ba.cameras)):
            if np.random.rand() > cfg["inner_optim"]["ratio_known_cameras"]:
                continue
            objective.add(
                th.eb.VariableDifference(
                    camera_pose_vars[i],
                    camera_weight,
                    ba.gt_cameras[i].pose,
                    name=f"camera_diff_{i}",
                )
            )

    # Create optimizer and theseus layer
    # optimizer = th.GaussNewton(
    #     objective,
    #     max_iterations=cfg["inner_optim"]["max_iters"],
    #     step_size=0.1,
    # )
    optimizer = GaussianBeliefPropagation(
        objective,
        max_iterations=cfg["inner_optim"]["max_iters"],
    )
    theseus_optim = th.TheseusLayer(optimizer)

    optim_arg = {
        "track_best_solution": True,
        "track_err_history": True,
        "verbose": True,
        "backward_mode": th.BackwardMode.FULL,
        "relin_threshold": 0.1,
        "damping": 0.9,
        "dropout": 0.0,
        "schedule": synchronous_schedule(
            cfg["inner_optim"]["max_iters"], optimizer.n_edges
        ),
    }

    theseus_inputs = {}
    for cam in ba.cameras:
        theseus_inputs[cam.pose.name] = cam.pose.data.clone()
    for pt in ba.points:
        theseus_inputs[pt.name] = pt.data.clone()
    theseus_inputs["log_loss_radius"] = log_loss_radius.data.clone()

    with torch.no_grad():
        camera_loss_ref = camera_loss(ba, camera_pose_vars).item()
    print(f"CAMERA LOSS:  {camera_loss_ref: .3f}")
    # print_histogram(ba, theseus_inputs, "Input histogram:")

    objective.update(theseus_inputs)
    print("squred err:", objective.error_squared_norm().item())

    theseus_outputs, info = theseus_optim.forward(
        input_data=theseus_inputs,
        optimizer_kwargs=optim_arg,
    )

    loss = camera_loss(ba, camera_pose_vars).item()
    print(f"CAMERA LOSS: (loss, ref loss) {loss:.3f} {camera_loss_ref: .3f}")


if __name__ == "__main__":

    cfg = {
        "seed": 1,
        "num_cameras": 2,  # 10
        "num_points": 20,  # 200
        "average_track_length": 8,
        "track_locality": 0.2,
        "inner_optim": {
            "max_iters": 10,
            "verbose": True,
            "track_err_history": True,
            "keep_step_size": True,
            "regularize": True,
            "ratio_known_cameras": 0.1,
            "reg_w": 1e-3,
        },
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    run(cfg)
