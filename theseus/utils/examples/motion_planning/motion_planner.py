# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

import theseus as th
from theseus.embodied import HingeCost, Nonholonomic


class _XYDifference(th.CostFunction):
    def __init__(
        self,
        var: th.SE2,
        target: th.Point2,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        if not isinstance(var, th.SE2) and not isinstance(target, th.Point2):
            raise ValueError(
                "XYDifference expects var of type SE2 and target of type Point2."
            )
        self.var = var
        self.target = target
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

    def _jacobians_and_error_impl(
        self, compute_jacobians: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        Jlocal: List[torch.Tensor] = [] if compute_jacobians else None
        Jxy: List[torch.Tensor] = [] if compute_jacobians else None
        error = self.target.local(self.var.xy(jacobians=Jxy), jacobians=Jlocal)
        jac = [Jlocal[1].matmul(Jxy[0])] if compute_jacobians else None
        return jac, error

    def error(self) -> torch.Tensor:
        return self._jacobians_and_error_impl(compute_jacobians=False)[1]

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._jacobians_and_error_impl(compute_jacobians=True)

    def dim(self) -> int:
        return 2

    def _copy_impl(self, new_name: Optional[str] = None) -> "_XYDifference":
        return _XYDifference(  # type: ignore
            self.var.copy(), self.target.copy(), self.weight.copy(), name=new_name
        )


class MotionPlannerObjective(th.Objective):
    def __init__(
        self,
        map_size: int,
        epsilon_dist: float,
        total_time: float,
        collision_weight: float,
        Qc_inv: Union[List[List[float]], torch.Tensor],
        num_time_steps: int,
        use_single_collision_weight: bool = True,
        pose_type: Union[Type[th.Point2], Type[th.SE2]] = th.Point2,
        dtype: torch.dtype = torch.double,
        nonholonomic_w: float = 0.0,
        positive_vel_w: float = 0.0,
    ):
        for v in [
            map_size,
            epsilon_dist,
            total_time,
            collision_weight,
            Qc_inv,
            num_time_steps,
        ]:
            assert v is not None

        super().__init__(dtype=dtype)
        self.map_size = map_size
        self.epsilon_dist = epsilon_dist
        self.total_time = total_time
        self.collision_weight = collision_weight
        self.Qc_inv = copy.deepcopy(Qc_inv)
        self.num_time_steps = num_time_steps
        self.use_single_collision_weight = use_single_collision_weight
        self.pose_type = pose_type

        self.trajectory_len = num_time_steps + 1

        # --------------------------------------------------------------------------- #
        # ---------------------------- Auxiliary variables -------------------------- #
        # --------------------------------------------------------------------------- #
        # First we create auxiliary variables for all the cost functions we are going
        # to need. Auxiliary variables are variables whose value won't be changed by
        # the motion planner's optimizer, but that are required to compute cost
        # functions.
        # By giving them names, we can update for each batch (if needed),
        # via the motion planner's forward method.
        sdf_origin = th.Point2(name="sdf_origin", dtype=dtype)
        start_pose = pose_type(name="start", dtype=dtype)
        goal_point = th.Point2(name="goal", dtype=dtype)
        cell_size = th.Variable(torch.empty(1, 1, dtype=dtype), name="cell_size")
        sdf_data = th.Variable(
            torch.empty(1, map_size, map_size, dtype=dtype), name="sdf_data"
        )
        cost_eps = th.Variable(
            torch.tensor(epsilon_dist, dtype=dtype).view(1, 1), name="cost_eps"
        )
        dt = th.Variable(
            torch.tensor(total_time / num_time_steps, dtype=dtype).view(1, 1), name="dt"
        )

        # --------------------------------------------------------------------------- #
        # ------------------------------- Cost weights ------------------------------ #
        # --------------------------------------------------------------------------- #
        # For the GP cost functions, we create a single GPCost weight
        qc_inv_tensor = torch.as_tensor(Qc_inv, dtype=dtype)
        gp_cost_weight = th.eb.GPCostWeight(qc_inv_tensor.to(dtype=dtype), dt)

        # Now we create cost weights for the collision-avoidance cost functions
        # Each of this is a scalar cost weight with a named auxiliary variable.
        # Before running the motion planner, each cost weight value can be updated
        # by passing {name: new_cost_weight_tensor} to the forward method
        collision_cost_weights = []
        if use_single_collision_weight:
            collision_cost_weights.append(
                th.ScaleCostWeight(
                    th.Variable(
                        torch.tensor(collision_weight, dtype=dtype), name="collision_w"
                    )
                )
            )
        else:
            for i in range(1, self.trajectory_len):
                collision_cost_weights.append(
                    th.ScaleCostWeight(
                        th.Variable(
                            torch.tensor(collision_weight, dtype=dtype),
                            name=f"collision_w_{i}",
                        )
                    )
                )

        # For hard-constraints (end points pos/vel) we use a single scalar weight
        # with high value
        boundary_cost_weight = th.ScaleCostWeight(torch.tensor(100.0, dtype=dtype))

        # --------------------------------------------------------------------------- #
        # -------------------------- Optimization variables ------------------------- #
        # --------------------------------------------------------------------------- #
        # The optimization variables for the motion planer are poses (Point2/SE2) and
        # velocities (Vector) for each of the discrete time steps
        poses: List[Union[th.Point2, th.SE2]] = []
        velocities: List[th.Vector] = []
        for i in range(self.trajectory_len):
            poses.append(pose_type(name=f"pose_{i}", dtype=dtype))
            velocities.append(th.Vector(poses[-1].dof(), name=f"vel_{i}", dtype=dtype))

        # --------------------------------------------------------------------------- #
        # ------------------------------ Cost functions ----------------------------- #
        # --------------------------------------------------------------------------- #
        # First create the cost functions for the end point positions and velocities
        # which are hard constraints, and can be implemented via Difference cost
        # functions.
        self.add(
            th.Difference(poses[0], start_pose, boundary_cost_weight, name="pose_0")
        )
        self.add(
            th.Difference(
                velocities[0],
                th.Vector(tensor=torch.zeros(1, velocities[0].dof(), dtype=dtype)),
                boundary_cost_weight,
                name="vel_0",
            )
        )
        assert pose_type in [th.Point2, th.SE2]
        goal_cost_cls = th.Difference if pose_type == th.Point2 else _XYDifference
        self.add(
            goal_cost_cls(
                poses[-1],  # type: ignore
                goal_point,
                boundary_cost_weight,
                name="pose_N",
            )
        )
        self.add(
            th.Difference(
                velocities[-1],
                th.Vector(tensor=torch.zeros(1, velocities[-1].dof(), dtype=dtype)),
                boundary_cost_weight,
                name="vel_N",
            )
        )

        if nonholonomic_w > 0.0:
            assert pose_type == th.SE2
            nhw = th.ScaleCostWeight(nonholonomic_w, name="nonholonomic_w")

        if positive_vel_w > 0.0:
            assert pose_type == th.SE2
            pvw = th.ScaleCostWeight(positive_vel_w, name="positive_vel_w")

        # Next we add 2-D collisions and GP cost functions, and associate them with the
        # cost weights created above. We need a separate cost function for each time
        # step
        for i in range(1, self.trajectory_len):
            self.add(
                th.eb.Collision2D(
                    poses[i],
                    sdf_origin,
                    sdf_data,
                    cell_size,
                    cost_eps,
                    collision_cost_weights[0]
                    if use_single_collision_weight
                    else collision_cost_weights[i - 1],
                    name=f"collision_{i}",
                )
            )
            self.add(
                (
                    th.eb.GPMotionModel(
                        poses[i - 1],
                        velocities[i - 1],
                        poses[i],
                        velocities[i],
                        dt,
                        gp_cost_weight,
                        name=f"gp_{i}",
                    )
                )
            )
            if nonholonomic_w > 0.0:
                self.add(
                    Nonholonomic(
                        poses[i],  # type: ignore
                        velocities[i],
                        nhw,
                        name=f"nonholonomic_{i}",
                    )
                )
            if positive_vel_w:
                self.add(
                    HingeCost(
                        velocities[i - 1],
                        torch.tensor([0.0, -torch.inf, -torch.inf]).view(1, 3),
                        torch.tensor([torch.inf, torch.inf, torch.inf]).view(1, 3),
                        1.0,
                        pvw,
                        name=f"positive_vel_{i}",
                    ),
                )


class MotionPlanner:
    # If objective is given, this overrides problem arguments
    def __init__(
        self,
        optimizer_config: Tuple[str, Dict[str, Any]],
        objective: Optional[MotionPlannerObjective] = None,
        device: th.DeviceType = "cpu",
        dtype: torch.dtype = torch.double,
        # The following are only used if objective is None
        map_size: Optional[int] = None,
        epsilon_dist: Optional[float] = None,
        total_time: Optional[float] = None,
        collision_weight: Optional[float] = None,
        Qc_inv: Optional[Union[List[List[float]], torch.Tensor]] = None,
        num_time_steps: Optional[int] = None,
        use_single_collision_weight: bool = True,
        pose_type: Union[Type[th.Point2], Type[th.SE2]] = th.Point2,
        nonholonomic_w: float = 0.0,
        positive_vel_w: float = 0.0,
    ):
        if objective is None:
            self.objective = MotionPlannerObjective(
                map_size,
                epsilon_dist,
                total_time,
                collision_weight,
                Qc_inv,
                num_time_steps,
                use_single_collision_weight=use_single_collision_weight,
                pose_type=pose_type,
                dtype=dtype,
                nonholonomic_w=nonholonomic_w,
                positive_vel_w=positive_vel_w,
            )
        else:
            self.objective = objective

        self.optimizer_config = optimizer_config
        self.device = device
        self.dtype = dtype

        # Finally, create the Nonlinear Least Squares optimizer for this objective
        # and wrap both into a TheseusLayer
        optimizer: th.NonlinearLeastSquares
        optimizer_cls = getattr(th, optimizer_config[0])
        assert issubclass(optimizer_cls, th.NonlinearLeastSquares)
        optimizer = optimizer_cls(self.objective, **optimizer_config[1])
        self.layer = th.TheseusLayer(optimizer)
        self.layer.to(device=device, dtype=dtype)

        # A call to motion_planner.layer.forward(input_dict) will run the NLLS optimizer
        # to solve for a trajectory given the input data. The input dictionary, supports
        # the keys and values illustrated below, where ':' represents a batch dimension.
        #
        # input_dict = {
        #   "cell_size": tensor(:, 1),
        #   "cost_eps": tensor(:, 1),
        #   "dt": tensor(:, 1),
        #   "sdf_origin": tensor (:, 2),
        #   "start": tensor (:, 2),
        #   "goal": tensor(:, 2),
        #   "sdf_data": tensor(:, map_size, map_size),
        #   "collision_w_0": tensor(:, 1),
        #   "collision_w_1": tensor(:, 1),
        #   ...
        #   "pose_0": tensor(:, 2),
        #   "pose_1": tensor(:, 2),
        #   ...
        #   "pose_N": tensor(:, 2),
        #   "vel_0": tensor(:, 2),
        #   "vel_1": tensor(:, 2),
        #   ...
        #   "vel_N": tensor(:, 2),
        # }
        #
        # TheseusLayer will match the names to their corresponding variables and update
        # their data before calling the optimizer.
        #
        # **Important**, all of these keys are
        # optional, so that only variables that need to change from call to call need to
        # have their names passed. Those not updated explictly here will retain their
        # previous internal data.
        #
        # When running
        #   output, info = motion_planner.forward(input_dict)
        #
        # The motion planner will only modify optimization variables, and all other
        # variables are not modified. For convenience, the output is a dictionary of
        # (str, tensor) mapping variable names to optimized variable data tensors.

    def forward(
        self,
        input_tensors: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], th.OptimizerInfo]:
        return self.layer.forward(
            input_tensors=input_tensors, optimizer_kwargs=optimizer_kwargs
        )

    def get_variable_values_from_straight_line(
        self, start: torch.Tensor, goal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Returns a dictionary of variable names to values that represent a straight
        # line trajectory from start to goal.
        # For SE2 variables, the start's angle is used for the full trajectory
        start_goal_dist = goal[:, :2] - start[:, :2]
        avg_vel = start_goal_dist / self.objective.total_time
        unit_trajectory_len = start_goal_dist / (self.objective.trajectory_len - 1)
        input_dict: Dict[str, torch.Tensor] = {}
        for i in range(self.objective.trajectory_len):
            if self.objective.pose_type == th.SE2:
                cur_pos = start[:, :2] + unit_trajectory_len * i
                input_dict[f"pose_{i}"] = torch.cat([cur_pos, start[:, 2:]], dim=1)
                input_dict[f"vel_{i}"] = torch.cat(
                    [avg_vel, torch.zeros_like(avg_vel[:, :1])], dim=1
                )
            else:
                input_dict[f"pose_{i}"] = start + unit_trajectory_len * i
                input_dict[f"vel_{i}"] = avg_vel
        return input_dict

    def get_randn_trajectory_like(
        self,
        start: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Returns a dictionary of variable names with random initial poses.
        # The batch size, device and dtype are obtained from given start
        pose_numel = self.objective.optim_vars["pose_0"].numel()
        vel_numel = self.objective.optim_vars["vel_0"].numel()
        input_dict: Dict[str, torch.Tensor] = {}
        assert start.shape[1] == pose_numel
        for i in range(self.objective.trajectory_len):
            input_dict[f"pose_{i}"] = torch.randn_like(start)
            input_dict[f"vel_{i}"] = torch.randn_like(start[:, :vel_numel])
        return input_dict

    def get_variable_values_from_trajectory(
        self, trajectory: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Returns a dictionary of variable names to values, so that values
        # are assigned with the data from the given trajectory.
        # For Point2 trajectories, trajectory should be a
        # tensor of shape (batch_size, 4, planner.trajectory_len).
        # For SE2 trajectories, it should be a tensor of shape
        # tensor of shape (batch_size, 7, planner.trajectory_len).
        pose_numel = self.objective.optim_vars["pose_0"].numel()
        vel_numel = self.objective.optim_vars["vel_0"].numel()
        assert trajectory.shape[1:] == (
            pose_numel + vel_numel,
            self.objective.trajectory_len,
        )
        input_dict: Dict[str, torch.Tensor] = {}
        for i in range(self.objective.trajectory_len):
            input_dict[f"pose_{i}"] = trajectory[:, :pose_numel, i]
            input_dict[f"vel_{i}"] = trajectory[:, pose_numel:, i]
        return input_dict

    def error(self) -> float:
        # Returns the current MSE of the optimization problem
        with torch.no_grad():
            return self.objective.error_metric().mean().item()

    def get_trajectory(
        self,
        values_dict: Optional[Dict[str, torch.Tensor]] = None,
        detach: bool = False,
    ) -> torch.Tensor:
        # Returns the a tensor with the trajectory that the given variable
        # values represent. If no dictionary is passed, it will used the latest
        # values stored in the objective's variables.
        pose_numel = 2 if self.objective.pose_type == th.Point2 else 4
        vel_numel = 2 if self.objective.pose_type == th.Point2 else 3
        trajectory = torch.zeros(
            self.objective.batch_size,
            pose_numel + vel_numel,
            self.objective.trajectory_len,
            device=self.objective.device,
        )
        if values_dict is None:
            values_dict = {
                k: t.tensor.clone() for (k, t) in self.objective.optim_vars.items()
            }
        for i in range(self.objective.trajectory_len):
            trajectory[:, :pose_numel, i] = values_dict[f"pose_{i}"]
            if f"vel_{i}" in values_dict:
                trajectory[:, pose_numel:, i] = values_dict[f"vel_{i}"]
        return trajectory.detach() if detach else trajectory

    def get_total_squared_errors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        gp_error: torch.Tensor = 0  # type: ignore
        collision_error: torch.Tensor = 0  # type: ignore
        for name, cf in self.objective.cost_functions.items():
            if "gp" in name:
                gp_error += cf.error().square().mean()
            if "collision" in name:
                collision_error += cf.error().square().mean()
        return gp_error, collision_error

    def copy(self, collision_weight: Optional[float] = None) -> "MotionPlanner":
        return MotionPlanner(
            self.optimizer_config,
            map_size=self.objective.map_size,
            epsilon_dist=self.objective.epsilon_dist,
            total_time=self.objective.total_time,
            collision_weight=collision_weight or self.objective.collision_weight,
            Qc_inv=self.objective.Qc_inv,
            num_time_steps=self.objective.num_time_steps,
            use_single_collision_weight=self.objective.use_single_collision_weight,
            device=self.device,
            pose_type=self.objective.pose_type,
            dtype=self.dtype,
        )
