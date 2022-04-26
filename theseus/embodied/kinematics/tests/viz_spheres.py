import time

# import mujoco_py
import pybullet as p
import torch

import theseus as th

urdf_path = "data/panda_no_gripper.urdf"
xml_path = "data/panda_no_gripper.xml"
joint_pos_home = torch.Tensor(
    [-0.1394, -0.0205, -0.0520, -2.0691, 0.0506, 2.0029, -0.9168]
)
sample_range_rad = 0.5

# Initialize simulation
sim_id = p.connect(p.GUI)
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setTimeStep(1.0 / 120.0)

robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0],
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE,
)

# Initialize kinematics model
robot_model = th.eb.UrdfRobotModel(urdf_path)

# Sample pose & update sim
joint_pos = joint_pos_home + sample_range_rad * torch.randn((7,))
for i in range(7):
    j_idx = robot_model.drm_model._controlled_joints[i] - 1
    p.resetJointState(
        bodyUniqueId=robot_id,
        jointIndex=j_idx,
        targetValue=joint_pos[i],
    )

# Query kinematics model
link_poses = robot_model.forward_kinematics(joint_pos)
spheres_data = robot_model.get_collision_spheres(link_poses)

# Update sim with spheres
for link_name, spheres in spheres_data.items():
    for sphere in spheres:
        v_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=sphere.radius,
            rgbaColor=[1, 0, 0, 0.2],
        )

        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=v_id,
            basePosition=sphere.position.data.squeeze(),
            baseInertialFramePosition=[0, 0, 0],
            useMaximalCoordinates=False,
        )

try:
    while True:
        keys = p.getKeyboardEvents()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

except KeyboardInterrupt:
    pass
