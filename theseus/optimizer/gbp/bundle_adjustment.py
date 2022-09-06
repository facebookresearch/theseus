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

import os
import json
from datetime import datetime

import theseus as th
from theseus.core import Vectorize
import theseus.utils.examples as theg
from theseus.optimizer.gbp import GaussianBeliefPropagation, GBPSchedule

# from theseus.optimizer.gbp import BAViewer


OPTIMIZER_CLASS = {
    "gbp": GaussianBeliefPropagation,
    "gauss_newton": th.GaussNewton,
    "levenberg_marquardt": th.LevenbergMarquardt,
}

OUTER_OPTIMIZER_CLASS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

GBP_SCHEDULE = {
    "synchronous": GBPSchedule.SYNCHRONOUS,
}


def save_res_loss_rad(save_dir, cfg, sweep_radii, sweep_losses, radius_vals, losses):
    with open(f"{save_dir}/config.txt", "w") as f:
        json.dump(cfg, f, indent=4)

    # sweep values
    np.savetxt(f"{save_dir}/sweep_radius.txt", sweep_radii)
    np.savetxt(f"{save_dir}/sweep_loss.txt", sweep_losses)

    # optim trajectory
    np.savetxt(f"{save_dir}/optim_radius.txt", radius_vals)
    np.savetxt(f"{save_dir}/optim_loss.txt", losses)


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


# Assumes the weight of the cost functions are 1
def average_repojection_error(objective, values_dict=None) -> float:
    if values_dict is not None:
        objective.update(values_dict)
    if objective._vectorized is False:
        Vectorize(objective)
    reproj_norms = []
    for cost_function in objective._get_iterator():
        if "Reprojection" in cost_function.name:
            # should equal error as weight is 1
            # need to call weighted_error as error is not cached
            err = cost_function.weighted_error().norm(dim=1)
            reproj_norms.append(err)

    are = torch.tensor(reproj_norms).mean().item()
    return are


def load_problem(cfg: omegaconf.OmegaConf):
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

    return ba


def setup_layer(cfg: omegaconf.OmegaConf):
    ba = load_problem(cfg)

    print("Optimizer:", cfg["optim"]["optimizer_cls"], "\n")

    # param that control transition from squared loss to huber
    radius_tensor = torch.tensor([1.0], dtype=torch.float64)
    log_loss_radius = th.Vector(tensor=radius_tensor, name="log_loss_radius")

    # Set up objective
    print("Setting up objective")
    t0 = time.time()
    dtype = torch.float64
    objective = th.Objective(dtype=dtype)
    dummy_objective = th.Objective(dtype=dtype)  # for computing are

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
        dummy_objective.add(cost_function)

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
    vectorize = cfg["optim"]["vectorize"]
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
    dummy_objective.to(cfg["device"])
    print("Device:", cfg["device"])

    # create damping parameter
    lin_system_damping = torch.nn.Parameter(
        torch.tensor(
            [cfg["optim"]["gbp_settings"]["lin_system_damping"]], dtype=torch.float64
        )
    )
    lin_system_damping.to(device=cfg["device"])

    optim_arg = {
        "track_best_solution": False,
        "track_err_history": True,
        "track_state_history": cfg["optim"]["track_state_history"],
        "verbose": True,
        "backward_mode": th.BackwardMode.FULL,
    }
    if isinstance(optimizer, GaussianBeliefPropagation):
        gbp_args = cfg["optim"]["gbp_settings"].copy()
        gbp_args["lin_system_damping"] = lin_system_damping
        gbp_args["schedule"] = GBP_SCHEDULE[gbp_args["schedule"]]
        optim_arg = {**optim_arg, **gbp_args}

    theseus_inputs = {}
    for cam in ba.cameras:
        theseus_inputs[cam.pose.name] = cam.pose.tensor.clone()
    for pt in ba.points:
        theseus_inputs[pt.name] = pt.tensor.clone()

    return (
        theseus_optim,
        theseus_inputs,
        optim_arg,
        ba,
        dummy_objective,
        camera_pose_vars,
        lin_system_damping,
    )


def run_inner(
    theseus_optim,
    theseus_inputs,
    optim_arg,
    ba,
    dummy_objective,
    camera_pose_vars,
    lin_system_damping,
):
    if ba.gt_cameras is not None:
        with torch.no_grad():
            camera_loss_ref = camera_loss(ba, camera_pose_vars).item()
        print(f"CAMERA LOSS:  {camera_loss_ref: .3f}")
    are = average_repojection_error(dummy_objective, values_dict=theseus_inputs)
    print("Average reprojection error (pixels): ", are)
    print_histogram(ba, theseus_inputs, "Input histogram:")

    with torch.no_grad():
        theseus_outputs, info = theseus_optim.forward(
            input_tensors=theseus_inputs,
            optimizer_kwargs=optim_arg,
        )

    if ba.gt_cameras is not None:
        loss = camera_loss(ba, camera_pose_vars).item()
        print(f"CAMERA LOSS: (loss, ref loss) {loss:.3f} {camera_loss_ref: .3f}")

    are = average_repojection_error(dummy_objective, values_dict=theseus_outputs)
    print("Average reprojection error (pixels): ", are)
    print_histogram(ba, theseus_outputs, "Final histogram:")

    # if info.state_history is not None:
    #     BAViewer(
    #         info.state_history, gt_cameras=ba.gt_cameras, gt_points=ba.gt_points
    #     )  # , msg_history=optimizer.ftov_msgs_history)

    """
    Save for nesterov experiments
    """
    save_dir = os.getcwd() + "/outputs/nesterov/bal/"
    if cfg["optim"]["gbp_settings"]["nesterov"]:
        save_dir += "1/"
    else:
        save_dir += "0/"
    os.mkdir(save_dir)
    with open(f"{save_dir}/config.txt", "w") as f:
        json.dump(cfg, f, indent=4)
    np.savetxt(save_dir + "/error_history.txt", info.err_history[0].cpu().numpy())

    """
    Save for bal sequences
    """

    # if cfg["bal_file"] is not None:
    #     save_dir = os.path.join(os.getcwd(), "outputs")
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     err_history = info.err_history[0].cpu().numpy()
    #     save_file = os.path.join(
    #         save_dir,
    #         f"{cfg['optim']['optimizer_cls']}_err_{cfg['bal_file'].split('/')[-1]}",
    #     )
    #     np.savetxt(save_file, err_history)

    # # get average reprojection error for each iteration
    # if info.state_history is not None:
    #     ares = []
    #     iters = (
    #         info.converged_iter
    #         if info.converged_iter != -1
    #         else cfg["optim"]["max_iters"]
    #     )
    #     for i in range(iters):
    #         t0 = time.time()
    #         values_dict = {}
    #         for name, state in info.state_history.items():
    #             values_dict[name] = (
    #                 state[..., i].to(dtype=torch.float64).to(dummy_objective.device)
    #             )
    #         are = average_repojection_error(dummy_objective, values_dict=values_dict)
    #         ares.append(are)
    #         print(i, "-- ARE:", are, " -- time", time.time() - t0)
    #     are = average_repojection_error(dummy_objective, values_dict=theseus_outputs)
    #     ares.append(are)

    #     if cfg["bal_file"] is not None:
    #         save_dir = os.path.join(os.getcwd(), "outputs")
    #         if not os.path.exists(save_dir):
    #             os.mkdir(save_dir)
    #         save_file = os.path.join(
    #             save_dir,
    #             f"{cfg['optim']['optimizer_cls']}_are_{cfg['bal_file'].split('/')[-1]}",
    #         )
    #         np.savetxt(save_file, np.array(ares))


def run_outer(cfg: omegaconf.OmegaConf):

    (
        theseus_optim,
        theseus_inputs,
        optim_arg,
        ba,
        dummy_objective,
        camera_pose_vars,
        lin_system_damping,
    ) = setup_layer(cfg)

    loss_radius_tensor = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    model_optimizer = OUTER_OPTIMIZER_CLASS[cfg["outer"]["optimizer"]](
        [loss_radius_tensor], lr=cfg["outer"]["lr"]
    )
    # model_optimizer = torch.optim.Adam([lin_system_damping], lr=cfg["outer"]["lr"])

    theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

    with torch.no_grad():
        camera_loss_ref = camera_loss(ba, camera_pose_vars).item()
    print(f"CAMERA LOSS (no learning):  {camera_loss_ref: .3f}")
    print_histogram(ba, theseus_inputs, "Input histogram:")

    import matplotlib.pylab as plt

    sweep_radii = torch.linspace(0.01, 5.0, 20)
    sweep_losses = []
    with torch.set_grad_enabled(False):
        for r in sweep_radii:
            theseus_inputs["log_loss_radius"][0] = r

            print(theseus_inputs["log_loss_radius"])

            theseus_outputs, info = theseus_optim.forward(
                input_tensors=theseus_inputs,
                optimizer_kwargs=optim_arg,
            )
            cam_loss = camera_loss(ba, camera_pose_vars)
            loss = (cam_loss - camera_loss_ref) / camera_loss_ref
            sweep_losses.append(torch.sum(loss.detach()).item())

    plt.plot(sweep_radii, sweep_losses)
    plt.xlabel("Log loss radius")
    plt.ylabel("(Camera loss - reference loss) / reference loss")

    losses = []
    radius_vals = []
    theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

    for epoch in range(cfg["outer"]["num_epochs"]):
        print(f" ******************* EPOCH {epoch} ******************* ")
        start_time = time.time_ns()
        model_optimizer.zero_grad()
        theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

        theseus_outputs, info = theseus_optim.forward(
            input_tensors=theseus_inputs,
            optimizer_kwargs=optim_arg,
        )

        cam_loss = camera_loss(ba, camera_pose_vars)
        loss = (cam_loss - camera_loss_ref) / camera_loss_ref
        loss.backward()
        radius_vals.append(loss_radius_tensor.data.item())
        print(loss_radius_tensor.grad)
        model_optimizer.step()
        loss_value = torch.sum(loss.detach()).item()
        losses.append(loss_value)
        end_time = time.time_ns()

        # print_histogram(ba, theseus_outputs, "Output histogram:")
        print(f"camera loss {cam_loss} and ref loss {camera_loss_ref}")
        print(
            f"Epoch: {epoch} Loss: {loss_value} "
            # f"Lin system damping {lin_system_damping}"
            f"Kernel Radius: exp({loss_radius_tensor.data.item()})="
            f"{torch.exp(loss_radius_tensor.data).item()}"
        )
        print(f"Epoch took {(end_time - start_time) / 1e9: .3f} seconds")

    print("Loss values:", losses)

    now = datetime.now()
    time_str = now.strftime("%m-%d-%y_%H-%M-%S")
    save_dir = os.getcwd() + "/outputs/loss_radius_exp/" + time_str
    os.mkdir(save_dir)

    save_res_loss_rad(save_dir, cfg, sweep_radii, sweep_losses, radius_vals, losses)

    plt.scatter(radius_vals, losses, c=range(len(losses)), cmap=plt.get_cmap("viridis"))
    plt.title(cfg["optim"]["optimizer_cls"] + " - " + time_str)
    plt.show()


if __name__ == "__main__":

    cfg = {
        "seed": 1,
        "device": "cpu",
        # "bal_file": None,
        "bal_file": "/mnt/sda/bal/problem-21-11315-pre.txt",
        "synthetic": {
            "num_cameras": 10,
            "num_points": 100,
            "average_track_length": 8,
            "track_locality": 0.2,
        },
        "optim": {
            "max_iters": 300,
            "vectorize": True,
            "optimizer_cls": "gbp",
            # "optimizer_cls": "gauss_newton",
            # "optimizer_cls": "levenberg_marquardt",
            "track_state_history": True,
            "regularize": True,
            "ratio_known_cameras": 0.1,
            "reg_w": 1e-7,
            "gbp_settings": {
                "relin_threshold": 1e-8,
                "ftov_msg_damping": 0.0,
                "dropout": 0.0,
                "schedule": "synchronous",
                "lin_system_damping": 1.0e-2,
                "nesterov": True,
            },
        },
        "outer": {
            "num_epochs": 15,
            "lr": 1e2,  # 5.0e-1,
            "optimizer": "sgd",
        },
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    args = setup_layer(cfg)
    run_inner(*args)

    # run_outer(cfg)
