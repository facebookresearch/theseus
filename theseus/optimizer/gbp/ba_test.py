# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Dict, List

import numpy as np
import omegaconf
import time
import torch

# import os

import theseus as th
import theseus.utils.examples as theg
from theseus.optimizer.gbp import GaussianBeliefPropagation, GBPSchedule

# from theseus.optimizer.gbp import BAViewer


OPTIMIZER_CLASS = {
    "gbp": GaussianBeliefPropagation,
    "gauss_newton": th.GaussNewton,
    "levenberg_marquardt": th.LevenbergMarquardt,
}


def print_histogram(
    ba: theg.BundleAdjustmentDataset, var_dict: Dict[str, torch.Tensor], msg: str
):
    print(msg)
    histogram = theg.ba_histogram(
        cameras=[
            theg.Camera(
                th.SE3(tensor=var_dict[c.pose.name]),
                c.focal_length,
                c.calib_k1,
                c.calib_k2,
            )
            for c in ba.cameras
        ],
        points=[th.Point3(tensor=var_dict[pt.name]) for pt in ba.points],
        observations=ba.observations,
    )
    for line in histogram.split("\n"):
        print(line)


def camera_loss(
    ba: theg.BundleAdjustmentDataset, camera_pose_vars: List[th.LieGroup]
) -> torch.Tensor:
    loss: torch.Tensor = 0  # type:ignore
    for i in range(len(ba.cameras)):
        cam_pose = camera_pose_vars[i].copy()
        cam_pose.to(ba.gt_cameras[i].pose.device)
        camera_loss = th.local(cam_pose, ba.gt_cameras[i].pose).norm(dim=1).cpu()
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
    if cfg["bal_file"] is None:
        ba = theg.BundleAdjustmentDataset.generate_synthetic(
            num_cameras=cfg["synthetic"]["num_cameras"],
            num_points=cfg["synthetic"]["num_points"],
            average_track_length=cfg["synthetic"]["average_track_length"],
            track_locality=cfg["synthetic"]["track_locality"],
            feat_random=1.5,
            prob_feat_is_outlier=0.02,
            outlier_feat_random=70,
            cam_pos_rand=5.0,
            cam_rot_rand=0.9,
            point_rand=10.0,
        )
    else:
        cams, points, obs = theg.BundleAdjustmentDataset.load_bal_dataset(
            cfg["bal_file"], drop_obs=0.0
        )
        ba = theg.BundleAdjustmentDataset(cams, points, obs)

    print("Cameras:", len(ba.cameras))
    print("Points:", len(ba.points))
    print("Observations:", len(ba.observations), "\n")

    # param that control transition from squared loss to huber
    radius_tensor = torch.tensor([1.0], dtype=torch.float64)
    log_loss_radius = th.Vector(tensor=radius_tensor, name="log_loss_radius")

    # Set up objective
    print("Setting up objective")
    t0 = time.time()
    objective = th.Objective(dtype=torch.float64)

    weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=ba.cameras[0].pose.dtype))
    for i, obs in enumerate(ba.observations):
        # print(i, len(ba.observations))
        cam = ba.cameras[obs.camera_index]
        cost_function = th.eb.Reprojection(
            camera_pose=cam.pose,
            world_point=ba.points[obs.point_index],
            focal_length=cam.focal_length,
            calib_k1=cam.calib_k1,
            calib_k2=cam.calib_k2,
            image_feature_point=obs.image_feature_point,
            weight=weight,
        )
        robust_cost_function = th.RobustCostFunction(
            cost_function,
            th.HuberLoss,
            log_loss_radius,
            name=f"robust_{cost_function.name}",
        )
        objective.add(robust_cost_function)
    dtype = objective.dtype

    # Add regularization
    if cfg["optim"]["regularize"]:
        # zero_point3 = th.Point3(dtype=dtype, name="zero_point")
        # identity_se3 = th.SE3(dtype=dtype, name="zero_se3")
        w = np.sqrt(cfg["optim"]["reg_w"])
        damping_weight = th.ScaleCostWeight(w * torch.ones(1, dtype=dtype))
        for name, var in objective.optim_vars.items():
            target: th.Manifold
            if isinstance(var, th.SE3):
                target = var.copy(new_name="target_" + var.name)
                # target = identity_se3
                objective.add(
                    th.Difference(var, target, damping_weight, name=f"reg_{name}")
                )
            # elif isinstance(var, th.Point3):
            #     target = var.copy(new_name="target_" + var.name)
            #     # target = zero_point3
            # else:
            #     assert False
            # objective.add(
            #     th.Difference(var, target, damping_weight, name=f"reg_{name}")
            # )

    camera_pose_vars: List[th.LieGroup] = [
        objective.optim_vars[c.pose.name] for c in ba.cameras  # type: ignore
    ]
    if cfg["optim"]["ratio_known_cameras"] > 0.0 and ba.gt_cameras is not None:
        w = 1000.0
        camera_weight = th.ScaleCostWeight(w * torch.ones(1, dtype=dtype))
        for i in range(len(ba.cameras)):
            if np.random.rand() > cfg["optim"]["ratio_known_cameras"]:
                continue
            print("fixing cam", i)
            objective.add(
                th.Difference(
                    camera_pose_vars[i],
                    ba.gt_cameras[i].pose,
                    camera_weight,
                    name=f"camera_diff_{i}",
                )
            )
    print("done in:", time.time() - t0)

    # Create optimizer and theseus layer
    vectorize = True
    optimizer = OPTIMIZER_CLASS[cfg["optim"]["optimizer_cls"]](
        objective,
        max_iterations=cfg["optim"]["max_iters"],
        vectorize=vectorize,
        # linearization_cls=th.SparseLinearization,
        # linear_solver_cls=th.LUCudaSparseSolver,
    )
    theseus_optim = th.TheseusLayer(optimizer, vectorize=vectorize)

    if cfg["device"] == "cuda":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    theseus_optim.to(cfg["device"])
    print("Device:", cfg["device"])

    optim_arg = {
        "track_best_solution": False,
        "track_err_history": True,
        "track_state_history": cfg["optim"]["track_state_history"],
        "verbose": True,
        "backward_mode": th.BackwardMode.FULL,
    }
    if isinstance(optimizer, GaussianBeliefPropagation):
        extra_args = {
            "relin_threshold": 0.0000000001,
            "damping": 0.0,
            "dropout": 0.0,
            "schedule": GBPSchedule.SYNCHRONOUS,
            "lin_system_damping": 1.0e-0,
        }
        optim_arg = {**optim_arg, **extra_args}

    theseus_inputs = {}
    for cam in ba.cameras:
        theseus_inputs[cam.pose.name] = cam.pose.tensor.clone()
    for pt in ba.points:
        theseus_inputs[pt.name] = pt.tensor.clone()

    if ba.gt_cameras is not None:
        with torch.no_grad():
            camera_loss_ref = camera_loss(ba, camera_pose_vars).item()
        print(f"CAMERA LOSS:  {camera_loss_ref: .3f}")
    are = average_repojection_error(objective)
    print("Average reprojection error (pixels): ", are)
    print_histogram(ba, theseus_inputs, "Input histogram:")

    objective.update(theseus_inputs)
    print("squred err:", objective.error_squared_norm().item())

    with torch.no_grad():
        theseus_outputs, info = theseus_optim.forward(
            input_tensors=theseus_inputs,
            optimizer_kwargs=optim_arg,
        )

    if ba.gt_cameras is not None:
        loss = camera_loss(ba, camera_pose_vars).item()
        print(f"CAMERA LOSS: (loss, ref loss) {loss:.3f} {camera_loss_ref: .3f}")

    are = average_repojection_error(objective)
    print("Average reprojection error (pixels): ", are)
    print_histogram(ba, theseus_outputs, "Final histogram:")

    # if cfg["optim"]["track_state_history"]:
    #     BAViewer(
    #         info.state_history, gt_cameras=ba.gt_cameras, gt_points=ba.gt_points
    #     )  # , msg_history=optimizer.ftov_msgs_history)

    # if cfg["bal_file"] is not None:
    #     save_dir = os.path.join(os.getcwd(), "outputs")
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     err_history = info.err_history[0].cpu().numpy()
    #     save_file = os.path.join(
    #         save_dir,
    #         f"{cfg['optim']['optimizer_cls']}_{cfg['bal_file'].split('/')[-1]}",
    #     )
    #     np.savetxt(save_file, err_history)


if __name__ == "__main__":

    cfg = {
        "seed": 1,
        "device": "cpu",
        # "bal_file": None,
        "bal_file": "/media/joe/data/bal/trafalgar/problem-21-11315-pre.txt",
        "synthetic": {
            "num_cameras": 10,
            "num_points": 100,
            "average_track_length": 8,
            "track_locality": 0.2,
        },
        "optim": {
            "max_iters": 500,
            "optimizer_cls": "gbp",
            # "optimizer_cls": "gauss_newton",
            # "optimizer_cls": "levenberg_marquardt",
            "track_state_history": False,
            "regularize": True,
            "ratio_known_cameras": 0.1,
            "reg_w": 1e-7,
        },
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    run(cfg)
