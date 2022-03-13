import torch
import theseus as th

import theseus.utils.examples as theg
from theseus.utils.examples.bundle_adjustment.data import *

torch.manual_seed(1)

# Smaller values result in error
th.SO3.SO3_EPS = 1e-6

# small test to check load/save/synthgen work
if False:
    ba = BundleAdjustmentDataset.generate_synthetic(30, 1000)
    print("\nBA:")
    ba.histogram()

    path = "/tmp/test.txt"
    ba.save_to_file(path)

    ba2 = BundleAdjustmentDataset.load_from_file(path)
    print("\nBA2:")
    ba2.histogram()

    ba3 = BundleAdjustmentDataset.load_from_file("/home/maurimo/BAL/problem-49-7776-pre.txt")
    print("\nBA3:")
    ba3.histogram()


# create (or load) dataset
ba = BundleAdjustmentDataset.generate_synthetic(num_cameras=10,
                                                num_points=200,
                                                average_track_length=8,
                                                track_locality=0.2)

# hyper parameters (ie outer loop's parameters)
log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=torch.float64)

# Set up objective
objective = th.Objective(dtype=torch.float64)

for obs in ba.observations:
    cam = ba.cameras[obs.camera_index]
    cost_function = theg.ReprojectionError(
        camera_pose=cam.pose,
        focal_length=cam.focal_length,
        calib_k1=cam.calib_k1,
        calib_k2=cam.calib_k2,
        log_loss_radius=log_loss_radius,
        world_point=ba.points[obs.point_index],
        image_feature_point=obs.image_feature_point,
    )
    objective.add(cost_function)

# Create optimizer
optimizer = th.LevenbergMarquardt(  # GaussNewton(
    objective,
    max_iterations=3,
    step_size=0.3,
)

# Set up Theseus layer
theseus_optim = th.TheseusLayer(optimizer)

# copy the poses/pts to feed them to each outer iteration
ba_orig_poses = {cam.pose.name: cam.pose.data.clone() for cam in ba.cameras}
ba_orig_pts = {pt.name: pt.data.clone() for pt in ba.points}

# loads (the only) batch
def get_batch(i):
    retv = {}
    for cam in ba.cameras:
        retv[cam.pose.name] = ba_orig_poses[cam.pose.name].clone()
    for pt in ba.points:
        retv[pt.name] = ba_orig_pts[pt.name].clone()
    return retv

num_batches = 1


# Outer optimization loop
loss_radius_tensor = torch.nn.Parameter(torch.tensor([-1], dtype=torch.float64))
model_optimizer = torch.optim.Adam([loss_radius_tensor], lr=0.1)


num_epochs = 20
camera_pose_vars = [theseus_optim.objective.optim_vars[c.pose.name] for c in ba.cameras]
for epoch in range(num_epochs):
    print(f" ******************* EPOCH {epoch} ******************* ")
    epoch_loss = 0.0
    for i in range(num_batches):
        model_optimizer.zero_grad()
        theseus_inputs = get_batch(i)
        batch_size = 1
        theseus_inputs["log_loss_radius"] = loss_radius_tensor.repeat(
            batch_size
        ).unsqueeze(1).clone()

        # histogram of optimization result, create datasets from input values
        print("Input histograms:")
        ba_input = BundleAdjustmentDataset(cameras=[Camera(th.SE3(data=theseus_inputs[c.pose.name]),
                                                    c.focal_length, c.calib_k1, c.calib_k2) for c in ba.cameras],
                                            points=[theseus_inputs[pt.name] for pt in ba.points],
                                            observations=ba.observations)
        ba_input.histogram()

        theseus_outputs, info = theseus_optim.forward(
            input_data=theseus_inputs, optimizer_kwargs={"verbose": True}
        )

        # histogram of optimization result, create datasets from optimized values
        print("Output histograms:")
        ba_result = BundleAdjustmentDataset(cameras=[Camera(theseus_optim.objective.optim_vars[c.pose.name],
                                                    c.focal_length, c.calib_k1, c.calib_k2) for c in ba.cameras],
                                            points=[theseus_optim.objective.optim_vars[pt.name] for pt in ba.points],
                                            observations=ba.observations)
        ba_result.histogram()

        loss = sum(th.local(camera_pose_vars[i], ba.gt_cameras[i].pose).norm(dim=1)
            for i in range(len(ba.cameras))
        )
        loss.backward()
        model_optimizer.step()
        loss_value = torch.sum(loss.detach()).item()
        epoch_loss += loss_value

    print(
        f"Epoch: {epoch} Loss: {epoch_loss} "
        f"Kernel Radius: exp({loss_radius_tensor.data.item()})="
        f"{torch.exp(loss_radius_tensor.data).item()}"
    )
