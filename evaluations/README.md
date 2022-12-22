This folder contains scripts for evaluating `theseus`'s components.
These scripts should be run from the root folder of the repository. The following
scripts are available, with reference to corresponding figure in our white paper:

 - `vectorization_ablation.sh`: Runs pose graph optimization with synthetic data with or without cost function vectorization (Fig. 1).
 - `pose_graph_synthetic.sh`: Same as above, but can change linear solver and problem size (Fig. 2).
 - `pose_graph_cube.sh`: Same as above, but using the cube data for Ceres comparison (Fig. 3). 
 - `backward_modes_tactile.sh`: Runs tactile state estimation with different backward modes (Fig. 4).
 - `autodiff_cost_function_ablation.sh`: Runs homography estimation with different autograd modes.

Some other relevant files to look at:

* Pose Graph Optimization:
    - `examples/pose_graph/pose_graph_{cube/synthetic}.py`: Puts together optimization layer and implements outer loop. 

* Tactile State Estimation:
    - `theseus/utils/examples/tactile_pose_estimation/trainer.py`: Main outer learning loop.
    - `theseus/utils/examples/tactile_pose_estimation/pose_estimator.py`: Puts together the optimization layer.

* Bundle Adjustment:
    - `examples/bundle_adjustment.py`: Puts together optimization layer and implements outer loop. 

* Motion Planning:
    - `theseus/utils/examples/motion_planning/motion_planner.py`: Puts together optimization layer.
    - `examples/motion_planning_2d.py`: Implements outer loop.

* Homography Estimation:
    - `examples/homography_estimation.py`: Puts together optimization layer and implements outer loop.
