import math

import torch
import theseus as th

import theseus.utils.examples as theg

torch.manual_seed(1)


# Smaller values result in error
th.SO3.SO3_EPS = 1e-6


# returns a uniformly random point of the 2-sphere
def random_S2():
    theta = torch.rand(()) * math.tau
    z = torch.rand(()) * 2 - 1
    r = torch.sqrt(1 - z**2)
    return torch.tensor([r * torch.cos(theta), r * torch.sin(theta), z]).double()


# returns a uniformly random point of the 3-sphere
def random_S3():
    u, v, w = torch.rand(3)
    return torch.tensor(
        [
            torch.sqrt(1 - u) * torch.sin(math.tau * v),
            torch.sqrt(1 - u) * torch.cos(math.tau * v),
            torch.sqrt(u) * torch.sin(math.tau * w),
            torch.sqrt(u) * torch.cos(math.tau * w),
        ]
    ).double()


def randomSmallQuaternion(max_degrees, min_degrees=0):
    x, y, z = random_S2()
    theta = (
        (min_degrees + (max_degrees - min_degrees) * torch.rand(())) * math.tau / 360.0
    )
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.tensor([c, s * x, s * y, s * z])


def add_noise_and_outliers(
    proj_points,
    noise_size=1,
    linear_noise=True,
    outliers_proportion=0.05,
    outlier_distance=500,
):

    if linear_noise:
        feat_image_points = proj_points + noise_size * (
            torch.rand(proj_points.shape, dtype=torch.float64) * 2 - 1
        )
    else:  # normal, stdDev = noiseSize
        feat_image_points = proj_points + torch.normal(
            mean=torch.zeros(proj_points.shape), std=noise_size, dtype=torch.float64
        )

    # add real bad outliers
    outliers_mask = torch.rand(feat_image_points.shape[0]) < outliers_proportion
    num_outliers = feat_image_points[outliers_mask].shape[0]
    feat_image_points[outliers_mask] += outlier_distance * (
        torch.rand((num_outliers, proj_points.shape[1]), dtype=proj_points.dtype) * 2
        - 1
    )
    return feat_image_points


class LocalizationSample:
    def __init__(self, num_points=60, focal_length=1000):
        self.focal_length = th.Variable(
            data=torch.tensor([focal_length], dtype=torch.float64), name="focal_length"
        )

        # pts = [+/-10, +/-10, +/-1]
        self.world_points = torch.cat(
            [
                torch.rand(2, num_points, dtype=torch.float64) * 20 - 10,
                torch.rand(1, num_points, dtype=torch.float64) * 2 - 1,
            ]
        ).T

        # gtCamPos = [+/-3, +/-3, 5 +/-1]
        gtCamPos = th.Point3(
            torch.tensor(
                [
                    [
                        torch.rand((), dtype=torch.float64) * 3,
                        torch.rand((), dtype=torch.float64) * 3,
                        5 + torch.rand((), dtype=torch.float64),
                    ]
                ]
            ),
            name="gtCamPos",
        )
        self.gt_cam_rotation = th.SO3(
            randomSmallQuaternion(max_degrees=20), name="gt_cam_rotation"
        )
        self.gt_cam_translation = (-self.gt_cam_rotation.rotate(gtCamPos)).copy(
            new_name="gt_cam_translation"
        )

        camera_points = (
            self.gt_cam_rotation.rotate(self.world_points) + self.gt_cam_translation
        )
        proj_points = (
            camera_points[:, :2] / camera_points[:, 2:3] * self.focal_length.data
        )
        self.image_feature_points = add_noise_and_outliers(proj_points)

        small_rotation = th.SO3(randomSmallQuaternion(max_degrees=0.3))
        small_translation = torch.rand(3, dtype=torch.float64) * 0.1
        self.obs_cam_rotation = small_rotation.compose(self.gt_cam_rotation).copy(
            new_name="obs_cam_rotation"
        )
        self.obs_cam_translation = (
            small_rotation.rotate(self.gt_cam_translation) + small_translation
        ).copy(new_name="obs_cam_translation")


localization_sample = LocalizationSample()


# create optimization problem
camera_rotation = localization_sample.obs_cam_rotation.copy(new_name="camera_rotation")
camera_translation = localization_sample.obs_cam_translation.copy(
    new_name="camera_translation"
)
loss_radius = th.Vector(1, name="loss_radius", dtype=torch.float64)
focal_length = th.Vector(1, name="focal_length", dtype=torch.float64)

# NOTE: if not set explicitly will crash using a weight of wrong type `float32`
weight = th.ScaleCostWeight(
    th.Vector(data=torch.tensor([1.0], dtype=torch.float64), name="weight")
)

# Set up objective
objective = th.Objective(dtype=torch.float64)

for i in range(len(localization_sample.world_points)):
    world_point = th.Point3(
        data=localization_sample.world_points[i], name=f"world_point_{i}"
    )
    image_feature_point = th.Point2(
        data=localization_sample.image_feature_points[i],
        name=f"image_feature_point_{i}",
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
loc_samples = [LocalizationSample() for _ in range(16)]
batch_size = 4
num_batches = (len(loc_samples) + batch_size - 1) // batch_size


def get_batch(b):
    assert b * batch_size < len(loc_samples)
    batch_ls = loc_samples[b * batch_size : (b + 1) * batch_size]
    batch_data = {
        "camera_rotation": th.SO3(
            data=torch.cat([ls.obs_cam_rotation.data for ls in batch_ls])
        ),
        "camera_translation": th.Point3(
            data=torch.cat([ls.obs_cam_translation.data for ls in batch_ls])
        ),
        "focal_length": th.Vector(
            data=torch.cat([ls.focal_length.data.unsqueeze(1) for ls in batch_ls]),
            name="focal_length",
        ),
    }

    # batch of 3d points and 2d feature points
    for i in range(len(batch_ls[0].world_points)):
        batch_data[f"world_point_{i}"] = th.Point3(
            data=torch.cat([ls.world_points[i : i + 1].data for ls in batch_ls]),
            name=f"world_point_{i}",
        )
        batch_data[f"image_feature_point_{i}"] = th.Point2(
            data=torch.cat(
                [ls.image_feature_points[i : i + 1].data for ls in batch_ls]
            ),
            name=f"image_feature_point_{i}",
        )

    gt_cam_rotation = th.SO3(
        data=torch.cat([ls.gt_cam_rotation.data for ls in batch_ls])
    )
    gt_cam_translation = th.Point3(
        data=torch.cat([ls.gt_cam_translation.data for ls in batch_ls])
    )
    return batch_data, gt_cam_rotation, gt_cam_translation


# Outer optimization loop
loss_radius_tensor = torch.nn.Parameter(torch.tensor([-3], dtype=torch.float64))
model_optimizer = torch.optim.Adam([loss_radius_tensor], lr=0.1)


num_epochs = 20
camera_rotation_var = theseus_optim.objective.optim_vars["camera_rotation"]
camera_translation_var = theseus_optim.objective.optim_vars["camera_translation"]
for epoch in range(num_epochs):
    print(" ******************* EPOCH {epoch} ******************* ")
    epoch_loss = 0.0
    for i in range(num_batches):
        print(f"BATCH {i}/{num_batches}")
        model_optimizer.zero_grad()
        theseus_inputs, gt_camera_rotation, gt_camera_translation = get_batch(i)
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
