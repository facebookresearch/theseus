# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

from .bundle_adjustment import BundleAdjustmentDataset, Camera, ba_histogram

try:
    from .motion_planning import (
        InitialTrajectoryModel,
        MotionPlanner,
        MotionPlannerObjective,
        ScalarCollisionWeightAndCostEpstModel,
        ScalarCollisionWeightModel,
        TrajectoryDataset,
        generate_trajectory_figs,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Unable to import Motion Planning utilities. "
        "Please make sure you have matplotlib installed."
    )

try:
    from .tactile_pose_estimation import (
        TactileMeasModel,
        TactilePoseEstimator,
        TactilePushingDataset,
        TactilePushingTrainer,
        TactileWeightModel,
        create_tactile_models,
        get_tactile_cost_weight_inputs,
        get_tactile_initial_optim_vars,
        get_tactile_motion_capture_inputs,
        get_tactile_nn_measurements_inputs,
        get_tactile_poses_from_values,
        init_tactile_model_from_file,
        update_tactile_pushing_inputs,
        visualize_tactile_push2d,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Unable to import Tactile Pose Estimation utilities. "
        "Please make sure you have matplotlib and omegaconf installed."
    )


from .pose_graph import PoseGraphDataset, PoseGraphEdge, pg_histogram
