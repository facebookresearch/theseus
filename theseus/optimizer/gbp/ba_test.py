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
from theseus.optimizer.gbp import (
    BAViewer,
    GaussianBeliefPropagation,
    synchronous_schedule,
)

# Smaller values result in error
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


def average_repojection_error(objective) -> float:

    reproj_norms = []
    for k in objective.cost_functions.keys():
        if "Reprojection" in k:
            err = objective.cost_functions[k].error().norm(dim=1)
            reproj_norms.append(err)

    are = torch.tensor(reproj_norms).mean().item()
    return are


def run(cfg: omegaconf.OmegaConf):
    # create (or load) dataset
    ba = theg.BundleAdjustmentDataset.generate_synthetic(
        num_cameras=cfg["num_cameras"],
        num_points=cfg["num_points"],
        average_track_length=cfg["average_track_length"],
        track_locality=cfg["track_locality"],
        feat_random=0.0,
        prob_feat_is_outlier=0.0,
        outlier_feat_random=70,
        cam_pos_rand=0.5,
        cam_rot_rand=0.1,
        point_rand=5.0,
    )

    # cams, points, obs = theg.BundleAdjustmentDataset.load_bal_dataset(
    #     "/media/joe/3.0TB Hard Disk/bal_data/problem-21-11315-pre.txt")
    # ba = theg.BundleAdjustmentDataset(cams, points, obs)
    # ba.save_to_file(results_path / "ba.txt", gt_path=results_path / "ba_gt.txt")

    # param that control transition from squared loss to huber
    radius_tensor = torch.tensor([1.0], dtype=torch.float64)
    log_loss_radius = th.Vector(data=radius_tensor, name="log_loss_radius")

    # Set up objective
    print("Setting up objective")
    objective = th.Objective(dtype=torch.float64)

    for i, obs in enumerate(ba.observations):
        # print(i, len(ba.observations))
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
        # identity_se3 = th.SE3(dtype=dtype, name="zero_se3")
        w = np.sqrt(cfg["inner_optim"]["reg_w"])
        damping_weight = th.ScaleCostWeight(w * torch.ones(1, dtype=dtype))
        for name, var in objective.optim_vars.items():
            target: th.Manifold
            if isinstance(var, th.SE3):
                target = var.copy(new_name="target_" + var.name)
                # target = identity_se3
            elif isinstance(var, th.Point3):
                # target = var.copy(new_name="target_" + var.name)
                target = zero_point3
            else:
                assert False
            objective.add(
                th.Difference(var, damping_weight, target, name=f"reg_{name}")
            )

    camera_pose_vars: List[th.LieGroup] = [
        objective.optim_vars[c.pose.name] for c in ba.cameras  # type: ignore
    ]
    if cfg["inner_optim"]["ratio_known_cameras"] > 0.0:
        w = 1000.0
        camera_weight = th.ScaleCostWeight(w * torch.ones(1, dtype=dtype))
        for i in range(len(ba.cameras)):
            if np.random.rand() > cfg["inner_optim"]["ratio_known_cameras"]:
                continue
            print("fixing cam", i)
            objective.add(
                th.Difference(
                    camera_pose_vars[i],
                    camera_weight,
                    ba.gt_cameras[i].pose,
                    name=f"camera_diff_{i}",
                )
            )

    # print("Factors:\n", objective.cost_functions.keys(), "\n")

    # Create optimizer and theseus layer
    optimizer = cfg["optimizer_cls"](
        objective,
        max_iterations=cfg["inner_optim"]["max_iters"],
    )
    theseus_optim = th.TheseusLayer(optimizer)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # theseus_optim.to(device)

    optim_arg = {
        "track_best_solution": True,
        "track_err_history": True,
        "verbose": True,
        "backward_mode": th.BackwardMode.FULL,
    }
    if cfg["optimizer_cls"] == GaussianBeliefPropagation:
        gbp_optim_arg = {
            "relin_threshold": 0.0000000001,
            "damping": 0.0,
            "dropout": 0.0,
            "schedule": synchronous_schedule(
                cfg["inner_optim"]["max_iters"], optimizer.n_edges
            ),
            "lin_system_damping": 1e-5,
        }
        optim_arg = {**optim_arg, **gbp_optim_arg}

    theseus_inputs = {}
    for cam in ba.cameras:
        theseus_inputs[cam.pose.name] = cam.pose.data.clone()
    for pt in ba.points:
        theseus_inputs[pt.name] = pt.data.clone()
    theseus_inputs["log_loss_radius"] = log_loss_radius.data.clone()

    with torch.no_grad():
        camera_loss_ref = camera_loss(ba, camera_pose_vars).item()
    print(f"CAMERA LOSS:  {camera_loss_ref: .3f}")
    print_histogram(ba, theseus_inputs, "Input histogram:")

    objective.update(theseus_inputs)
    print("squred err:", objective.error_squared_norm().item())

    theseus_outputs, info = theseus_optim.forward(
        input_data=theseus_inputs,
        optimizer_kwargs=optim_arg,
    )

    loss = camera_loss(ba, camera_pose_vars).item()
    print(f"CAMERA LOSS: (loss, ref loss) {loss:.3f} {camera_loss_ref: .3f}")

    are = average_repojection_error(objective)
    print("Average reprojection error (pixels): ", are)

    with torch.no_grad():
        camera_loss_ref = camera_loss(ba, camera_pose_vars).item()
    print(f"CAMERA LOSS:  {camera_loss_ref: .3f}")
    print_histogram(ba, theseus_inputs, "Final histogram:")

    BAViewer(
        optimizer.belief_history, gt_cameras=ba.gt_cameras, gt_points=ba.gt_points
    )  # , msg_history=optimizer.ftov_msgs_history)


if __name__ == "__main__":

    cfg = {
        "seed": 1,
        "num_cameras": 10,
        "num_points": 100,
        "average_track_length": 8,
        "track_locality": 0.2,
        "optimizer_cls": GaussianBeliefPropagation,
        # "optimizer_cls": th.GaussNewton,
        "inner_optim": {
            "max_iters": 10,
            "verbose": True,
            "track_err_history": True,
            "keep_step_size": True,
            "regularize": True,
            "ratio_known_cameras": 0.3,
            "reg_w": 1e-7,
        },
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    run(cfg)
