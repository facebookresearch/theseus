import torch
import theseus as th

import theseus.utils.examples as theg


torch.manual_seed(1)

# Smaller values result in error
th.SO3.SO3_EPS = 1e-6

# Experiment config
num_samples = 16
num_points = 60
batch_size = 4
num_epochs = 20

# create optimization problem
camera_rotation = th.SO3(name="camera_rotation", dtype=torch.float64)
camera_translation = th.Point3(name="camera_translation", dtype=torch.float64)
loss_radius = th.Vector(1, name="loss_radius", dtype=torch.float64)
focal_length = th.Vector(1, name="focal_length", dtype=torch.float64)

# Set up objective
objective = th.Objective(dtype=torch.float64)

for i in range(num_points):
    world_point = th.Point3(name=f"world_point_{i}", dtype=torch.float64)
    image_feature_point = th.Point2(
        name=f"image_feature_point_{i}", dtype=torch.float64
    )
    cost_function = theg.ReprojectionError(
        camera_rotation,
        camera_translation,
        focal_length,
        loss_radius,
        world_point,
        image_feature_point,
    )
    objective.add(cost_function)


# Create optimizer
optimizer = th.LevenbergMarquardt(  # GaussNewton(
    objective,
    max_iterations=10,
    step_size=0.3,
)

# Set up Theseus layer
theseus_optim = th.TheseusLayer(optimizer)

# Create dataset
localization_dataset = theg.LocalizationDataset(num_samples, num_points, batch_size)

# Outer optimization loop
loss_radius_tensor = torch.nn.Parameter(torch.tensor([-3], dtype=torch.float64))
model_optimizer = torch.optim.Adam([loss_radius_tensor], lr=0.1)
camera_rotation_var = theseus_optim.objective.optim_vars["camera_rotation"]
camera_translation_var = theseus_optim.objective.optim_vars["camera_translation"]
for epoch in range(num_epochs):
    print(" ******************* EPOCH {epoch} ******************* ")
    epoch_loss = 0.0
    for i, batch in enumerate(localization_dataset):
        print(f"BATCH {i}/{len(localization_dataset)}")
        model_optimizer.zero_grad()
        theseus_inputs, gt_camera_rotation, gt_camera_translation = batch
        theseus_inputs["loss_radius"] = loss_radius_tensor.repeat(
            gt_camera_translation.data.shape[0]
        ).unsqueeze(1)

        theseus_outputs, info = theseus_optim.forward(
            theseus_inputs, optimizer_kwargs={"verbose": False}
        )

        cam_rot_loss = th.local(camera_rotation_var, gt_camera_rotation).norm(dim=1)
        cam_tr_loss = th.local(camera_translation_var, gt_camera_translation).norm(
            dim=1, p=1
        )
        loss = 100 * cam_rot_loss + cam_tr_loss
        loss = torch.where(loss < 10e5, loss, 0.0).sum()
        loss.backward()
        model_optimizer.step()

        loss_value = torch.sum(loss.detach()).item()
        epoch_loss += loss_value

    print(
        f"Epoch: {epoch} Loss: {epoch_loss} "
        f"Kernel Radius: exp({loss_radius_tensor.data.item()})="
        f"{torch.exp(loss_radius_tensor.data).item()}"
    )
