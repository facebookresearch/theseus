from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch

import theseus as th
from theseus.optimizer.nonlinear.levenberg_marquardt import LevenbergMarquardt

from .misc import TactilePushingDataset


class TactilePoseEstimator:
    def __init__(
        self,
        dataset: TactilePushingDataset,
        min_window_moving_frame: int,
        max_window_moving_frame: int,
        step_window_moving_frame: int,
        rectangle_shape: Tuple[float, float],
        device: th.DeviceType,
        optimizer_cls: Optional[Type[th.NonlinearLeastSquares]] = LevenbergMarquardt,
        max_iterations: int = 3,
        step_size: float = 1.0,
        regularization_w: float = 0.0,
        force_max_iters: bool = False,
    ):
        time_steps = dataset.time_steps

        # -------------------------------------------------------------------- #
        # Creating optimization variables
        # -------------------------------------------------------------------- #
        # The optimization variables for this problem are SE2 object and end effector
        # poses over time.
        obj_poses, eff_poses = [], []
        for i in range(time_steps):
            obj_poses.append(th.SE2(name=f"obj_pose_{i}", dtype=torch.double))
            eff_poses.append(th.SE2(name=f"eff_pose_{i}", dtype=torch.double))

        # -------------------------------------------------------------------- #
        # Creating auxiliary variables
        # -------------------------------------------------------------------- #
        #  - obj_start_pose: target for boundary cost functions
        #  - motion_captures: priors on the end-effector poses
        #  - nn_measurements: tactile measurement prediction from image features
        #  - sdf_data, sdf_cell_size, sdf_origin: signed distance field data,
        #    cell_size and origin
        obj_start_pose = th.SE2(name="obj_start_pose")
        self.obj_start_pose = obj_start_pose

        motion_captures: List[th.SE2] = []
        for i in range(time_steps):
            motion_captures.append(th.SE2(name=f"motion_capture_{i}"))
        self.motion_captures = motion_captures

        nn_measurements = []
        for i in range(min_window_moving_frame, time_steps):
            for offset in range(
                min_window_moving_frame,
                np.minimum(i, max_window_moving_frame),
                step_window_moving_frame,
            ):
                nn_measurements.append(th.SE2(name=f"nn_measurement_{i-offset}_{i}"))

        sdf_data = th.Variable(dataset.sdf_data_tensor, name="sdf_data")
        sdf_cell_size = th.Variable(dataset.sdf_cell_size, name="sdf_cell_size")
        sdf_origin = th.Point2(dataset.sdf_origin, name="sdf_origin")
        eff_radius = th.Variable(torch.zeros(1, 1), name="eff_radius")

        # -------------------------------------------------------------------- #
        # Creating cost weights
        # -------------------------------------------------------------------- #
        #  - qsp_weight: diagonal cost weight shared across all quasi-static cost
        #    functions.
        #  - mf_between_weight: diagonal cost weight shared across all moving factor
        #    cost functions.
        #  - intersect_weight: scalar cost weight shared across all object-effector
        #    intersection cost functions.
        #  - motion_capture_weight: diagonal cost weight shared across all end-effector
        #    priors cost functions.
        qsp_weight = th.DiagonalCostWeight(
            th.Variable(torch.ones(1, 3), name="qsp_weight")
        )
        mf_between_weight = th.DiagonalCostWeight(
            th.Variable(torch.ones(1, 3), name="mf_between_weight")
        )
        intersect_weight = th.ScaleCostWeight(
            th.Variable(torch.ones(1, 1), name="intersect_weight")
        )
        motion_capture_weight = th.DiagonalCostWeight(
            th.Variable(torch.ones(1, 3), name="mc_weight")
        )

        # -------------------------------------------------------------------- #
        # Creating cost functions
        # -------------------------------------------------------------------- #
        #  - Difference: Penalizes deviation between first object pose from
        #    a global pose prior.
        #  - QuasiStaticPushingPlanar: Penalizes deviation from velocity-only
        #    quasi-static dynamics model QuasiStaticPushingPlanar
        #  - MovingFrameBetween: Penalizes deviation between relative end effector poses
        #    in object frame against a measurement target. Measurement target
        #    `nn_measurements` is obtained from a network prediction.
        #  - EffectorObjectContactPlanar: Penalizes intersections between object and end
        #    effector based on the object sdf.
        #  - Difference: Penalizes deviations of end-effector poses from motion
        #    capture readings

        # Loop over and add all cost functions,
        # cost weights, and their auxiliary variables
        objective = th.Objective()
        nn_meas_idx = 0
        c_square = (np.sqrt(rectangle_shape[0] ** 2 + rectangle_shape[1] ** 2)) ** 2
        for i in range(time_steps):
            if i == 0:
                objective.add(
                    th.Difference(
                        obj_poses[i],
                        obj_start_pose,
                        motion_capture_weight,
                        name=f"obj_priors_{i}",
                    )
                )

            if i < time_steps - 1:
                objective.add(
                    th.eb.QuasiStaticPushingPlanar(
                        obj_poses[i],
                        obj_poses[i + 1],
                        eff_poses[i],
                        eff_poses[i + 1],
                        c_square,
                        qsp_weight,
                        name=f"qsp_{i}",
                    )
                )
            if i >= min_window_moving_frame:
                for offset in range(
                    min_window_moving_frame,
                    np.minimum(i, max_window_moving_frame),
                    step_window_moving_frame,
                ):
                    objective.add(
                        th.eb.MovingFrameBetween(
                            obj_poses[i - offset],
                            obj_poses[i],
                            eff_poses[i - offset],
                            eff_poses[i],
                            nn_measurements[nn_meas_idx],
                            mf_between_weight,
                            name=f"mf_between_{i - offset}_{i}",
                        )
                    )
                    nn_meas_idx = nn_meas_idx + 1

            objective.add(
                th.eb.EffectorObjectContactPlanar(
                    obj_poses[i],
                    eff_poses[i],
                    sdf_origin,
                    sdf_data,
                    sdf_cell_size,
                    eff_radius,
                    intersect_weight,
                    name=f"intersect_{i}",
                )
            )

            objective.add(
                th.Difference(
                    eff_poses[i],
                    motion_captures[i],
                    motion_capture_weight,
                    name=f"eff_priors_{i}",
                )
            )

        if regularization_w > 0.0:
            reg_w = th.ScaleCostWeight(np.sqrt(regularization_w))
            reg_w.to(dtype=torch.double)
            identity_se2 = th.SE2(name="identity")
            for pose_list in [obj_poses, eff_poses]:
                for pose in pose_list:
                    objective.add(
                        th.Difference(
                            pose, identity_se2, reg_w, name=f"reg_{pose.name}"
                        )
                    )

        # -------------------------------------------------------------------- #
        # Creating TheseusLayer
        # -------------------------------------------------------------------- #
        # Wrap the objective and inner-loop optimizer into a `TheseusLayer`.
        # Inner-loop optimizer here is the Levenberg-Marquardt nonlinear optimizer
        # coupled with a dense linear solver based on Cholesky decomposition.
        nl_optimizer = optimizer_cls(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=max_iterations,
            step_size=step_size,
            abs_err_tolerance=0 if force_max_iters else 1e-10,
            rel_err_tolerance=0 if force_max_iters else 1e-8,
        )
        self.theseus_layer = th.TheseusLayer(nl_optimizer)
        self.theseus_layer.to(device=device, dtype=torch.double)

        self.forward = self.theseus_layer.forward

    # Gets a dictionary mapping variable names to tensors, with the batch data needed
    # to update start pose and motion capture data (which is in xytheta format)
    def get_start_pose_and_motion_capture_dict(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        var_names = [self.obj_start_pose.name] + [v.name for v in self.motion_captures]
        for name in var_names:
            tensor_dict[name] = th.SE2(x_y_theta=batch[name]).tensor
        return tensor_dict
