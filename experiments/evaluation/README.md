This folder contains scripts for evaluating `theseus`'s components.
These scripts should be run from the root folder of the repository.

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