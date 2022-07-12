# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Dict, List, Optional, Tuple

import torch

import theseus as th


class MotionPlanner:
    def __init__(
        self,
        map_size: int,
        epsilon_dist: float,
        total_time: float,
        collision_weight: float,
        Qc_inv: List[List[int]],
        num_time_steps: int,
        optim_method: str,
        max_optim_iters: int,
        step_size: float = 1.0,
        use_single_collision_weight: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.double,
    ):
        self.map_size = map_size
        self.epsilon_dist = epsilon_dist
        self.total_time = total_time
        self.collision_weight = collision_weight
        self.Qc_inv = copy.deepcopy(Qc_inv)
        self.num_time_steps = num_time_steps
        self.optim_method = optim_method
        self.max_optim_iters = max_optim_iters
        self.step_size = step_size
        self.use_single_collision_weight = use_single_collision_weight
        self.device = device
        self.dtype = dtype

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
        sdf_origin = th.Point2(name="sdf_origin")
        start_point = th.Point2(name="start")
        goal_point = th.Point2(name="goal")
        cell_size_tensor = th.Variable(torch.empty(1, 1), name="cell_size")
        sdf_data_tensor = th.Variable(
            torch.empty(1, map_size, map_size), name="sdf_data"
        )
        cost_eps = th.Variable(torch.tensor(epsilon_dist).view(1, 1), name="cost_eps")
        dt = th.Variable(
            torch.tensor(total_time / num_time_steps).view(1, 1), name="dt"
        )

        # --------------------------------------------------------------------------- #
        # ------------------------------- Cost weights ------------------------------ #
        # --------------------------------------------------------------------------- #
        # For the GP cost functions, we create a single GPCost weight
        gp_cost_weight = th.eb.GPCostWeight(torch.tensor(Qc_inv), dt)

        # Now we create cost weights for the collision-avoidance cost functions
        # Each of this is a scalar cost weight with a named auxiliary variable.
        # Before running the motion planner, each cost weight value can be updated
        # by passing {name: new_cost_weight_tensor} to the forward method
        collision_cost_weights = []
        if use_single_collision_weight:
            collision_cost_weights.append(
                th.ScaleCostWeight(
                    th.Variable(torch.tensor(collision_weight), name="collision_w")
                )
            )
        else:
            for i in range(1, self.trajectory_len):
                collision_cost_weights.append(
                    th.ScaleCostWeight(
                        th.Variable(
                            torch.tensor(collision_weight), name=f"collision_w_{i}"
                        )
                    )
                )

        # For hard-constraints (end points pos/vel) we use a single scalar weight
        # with high value
        boundary_cost_weight = th.ScaleCostWeight(torch.tensor(100.0))

        # --------------------------------------------------------------------------- #
        # -------------------------- Optimization variables ------------------------- #
        # --------------------------------------------------------------------------- #
        # The optimization variables for the motion planer are 2-D positions and
        # velocities for each of the discrete time steps
        poses = []
        velocities = []
        for i in range(self.trajectory_len):
            poses.append(th.Point2(name=f"pose_{i}", dtype=torch.double))
            velocities.append(th.Point2(name=f"vel_{i}", dtype=torch.double))

        # --------------------------------------------------------------------------- #
        # ------------------------------ Cost functions ----------------------------- #
        # --------------------------------------------------------------------------- #
        # Create a Theseus objective for adding the cost functions
        objective = th.Objective(dtype=torch.double)

        # First create the cost functions for the end point positions and velocities
        # which are hard constraints, and can be implemented via Difference cost
        # functions.
        objective.add(
            th.Difference(poses[0], start_point, boundary_cost_weight, name="pose_0")
        )
        objective.add(
            th.Difference(
                velocities[0],
                th.Point2(tensor=torch.zeros(1, 2)),
                boundary_cost_weight,
                name="vel_0",
            )
        )
        objective.add(
            th.Difference(poses[-1], goal_point, boundary_cost_weight, name="pose_N")
        )
        objective.add(
            th.Difference(
                velocities[-1],
                th.Point2(tensor=torch.zeros(1, 2)),
                boundary_cost_weight,
                name="vel_N",
            )
        )

        # Next we add 2-D collisions and GP cost functions, and associate them with the
        # cost weights created above. We need a separate cost function for each time
        # step
        for i in range(1, self.trajectory_len):
            objective.add(
                th.eb.Collision2D(
                    poses[i],
                    sdf_origin,
                    sdf_data_tensor,
                    cell_size_tensor,
                    cost_eps,
                    collision_cost_weights[0]
                    if use_single_collision_weight
                    else collision_cost_weights[i - 1],
                    name=f"collision_{i}",
                )
            )
            objective.add(
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

        # Finally, create the Nonlinear Least Squares optimizer for this objective
        # and wrap both into a TheseusLayer
        optimizer: th.NonlinearLeastSquares
        if optim_method == "gauss_newton":
            optimizer = th.GaussNewton(
                objective,
                th.CholeskyDenseSolver,
                max_iterations=max_optim_iters,
                step_size=step_size,
            )
        elif optim_method == "levenberg_marquardt":
            optimizer = th.LevenbergMarquardt(
                objective,
                th.CholeskyDenseSolver,
                max_iterations=max_optim_iters,
                step_size=step_size,
            )

        self.objective = objective
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

    def get_variable_values_from_straight_line(
        self, start: torch.Tensor, goal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Returns a dictionary of variable names to values that represent a straight
        # line trajectory from start to goal.
        start_goal_dist = goal - start
        avg_vel = start_goal_dist / self.total_time
        unit_trajectory_len = start_goal_dist / (self.trajectory_len - 1)
        input_dict: Dict[str, torch.Tensor] = {}
        for i in range(self.trajectory_len):
            input_dict[f"pose_{i}"] = start + unit_trajectory_len * i
            input_dict[f"vel_{i}"] = avg_vel
        return input_dict

    def get_random_variable_values(
        self, start: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Returns a dictionary of variable names with random initial poses.
        input_dict: Dict[str, torch.Tensor] = {}
        for i in range(self.trajectory_len):
            input_dict[f"pose_{i}"] = torch.randn_like(start)
            input_dict[f"vel_{i}"] = torch.randn_like(start)
        return input_dict

    def get_variable_values_from_trajectory(
        self, trajectory: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Returns a dictionary of variable names to values, so that values
        # are assigned with the data from the given trajectory. Trajectory should be a
        # tensor of shape (batch_size, 4, planner.trajectory_len).
        assert trajectory.shape[1:] == (4, self.trajectory_len)
        input_dict: Dict[str, torch.Tensor] = {}
        for i in range(self.trajectory_len):
            input_dict[f"pose_{i}"] = trajectory[:, :2, i]
            input_dict[f"vel_{i}"] = trajectory[:, :2, i]
        return input_dict

    def error(self) -> float:
        # Returns the current MSE of the optimization problem
        with torch.no_grad():
            return self.objective.error_squared_norm().mean().item()

    def get_trajectory(
        self,
        values_dict: Optional[Dict[str, torch.Tensor]] = None,
        detach: bool = False,
    ) -> torch.Tensor:
        # Returns the a tensor with the trajectory that the given variable
        # values represent. If no dictionary is passed, it will used the latest
        # values stored in the objective's variables.
        trajectory = torch.empty(
            self.objective.batch_size,
            4,
            self.trajectory_len,
            device=self.objective.device,
        )
        variables = self.objective.optim_vars
        for i in range(self.trajectory_len):
            if values_dict is None:
                trajectory[:, :2, i] = variables[f"pose_{i}"].tensor.clone()
                trajectory[:, 2:, i] = variables[f"vel_{i}"].tensor.clone()
            else:
                trajectory[:, :2, i] = values_dict[f"pose_{i}"]
                trajectory[:, 2:, i] = values_dict[f"vel_{i}"]
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
            self.map_size,
            self.epsilon_dist,
            self.total_time,
            collision_weight or self.collision_weight,
            self.Qc_inv,
            self.num_time_steps,
            self.optim_method,
            self.max_optim_iters,
            step_size=self.step_size,
            use_single_collision_weight=self.use_single_collision_weight,
            device=self.device,
            dtype=self.dtype,
        )
