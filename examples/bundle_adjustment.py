import math

import torch
import theseus as th

import theseus.utils.examples as theg

torch.manual_seed(0)


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


# unit test for Cost term
camera_rotation = th.SO3(
    torch.cat([randomSmallQuaternion(max_degrees=20).unsqueeze(0) for _ in range(4)]),
    name="camera_rotation",
)
camera_translation = th.Point3(
    data=torch.zeros((4, 3), dtype=torch.float64), name="camera_translation"
)
camera_translation.data[:, 2] += 5.0
focal_length = th.Vector(
    data=torch.tensor([1000], dtype=torch.float64).repeat(4).unsqueeze(1),
    name="focal_length",
)
loss_radius = th.Vector(
    data=torch.tensor([0], dtype=torch.float64).repeat(4).unsqueeze(1),
    name="loss_radius",
)
world_point = th.Point3(
    data=torch.rand((4, 3), dtype=torch.float64), name="world_point"
)
point__cam = camera_rotation.rotate(world_point) + camera_translation
image_feature_point = th.Point2(
    data=point__cam[:, :2] / point__cam[:, 2:] + torch.rand((4, 2)) * 50,
    name="image_feature_point",
)
r = theg.ReprojectionError(
    camera_rotation,
    camera_translation,
    focal_length,
    loss_radius,
    world_point,
    image_feature_point,
)


def add_noise_and_outliers(
    projPoints,
    noiseSize=1,
    noiseLinear=True,
    proportionOutliers=0.05,
    outlierDistance=500,
):

    if noiseLinear:
        featImagePoints = projPoints + noiseSize * (
            torch.rand(projPoints.shape, dtype=torch.float64) * 2 - 1
        )
    else:  # normal, stdDev = noiseSize
        featImagePoints = projPoints + torch.normal(
            mean=torch.zeros(projPoints.shape), std=noiseSize, dtype=torch.float64
        )

    # add real bad outliers
    outliersMask = torch.rand(featImagePoints.shape[0]) < proportionOutliers
    numOutliers = featImagePoints[outliersMask].shape[0]
    featImagePoints[outliersMask] += outlierDistance * (
        torch.rand((numOutliers, projPoints.shape[1]), dtype=projPoints.dtype) * 2 - 1
    )
    return featImagePoints


class LocalizationSample:
    def __init__(self, num_points=60, focalLength=1000):
        self.focalLength = th.Variable(
            data=torch.tensor([focalLength], dtype=torch.float64), name="focal_length"
        )

        # pts = [+/-10, +/-10, +/-1]
        self.worldPoints = torch.cat(
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
        self.gtCamRot = th.SO3(randomSmallQuaternion(max_degrees=20), name="gtCamRot")
        self.gtCamTr = (-self.gtCamRot.rotate(gtCamPos)).copy(new_name="gtCamTr")

        camPoints = self.gtCamRot.rotate(self.worldPoints) + self.gtCamTr
        projPoints = camPoints[:, :2] / camPoints[:, 2:3] * self.focalLength.data
        self.image_feature_points = add_noise_and_outliers(projPoints)

        smallRot = th.SO3(randomSmallQuaternion(max_degrees=0.3))
        smallTr = torch.rand(3, dtype=torch.float64) * 0.1
        self.obsCamRot = smallRot.compose(self.gtCamRot).copy(new_name="obsCamRot")
        self.obsCamTr = (smallRot.rotate(self.gtCamTr) + smallTr).copy(
            new_name="obsCamTr"
        )


localization_sample = LocalizationSample()


# create optimization problem
camera_rotation = localization_sample.obsCamRot.copy(new_name="camera_rotation")
camera_translation = localization_sample.obsCamTr.copy(new_name="camera_translation")
loss_radius = th.Vector(1, name="loss_radius", dtype=torch.float64)
focal_length = th.Vector(1, name="focal_length", dtype=torch.float64)

# NOTE: if not set explicitly will crash using a weight of wrong type `float32`
weight = th.ScaleCostWeight(
    th.Vector(data=torch.tensor([1.0], dtype=torch.float64), name="weight")
)

# Set up objective
objective = th.Objective(dtype=torch.float64)

for i in range(len(localization_sample.worldPoints)):
    world_point = th.Point3(
        data=localization_sample.worldPoints[i], name=f"world_point_{i}"
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
            data=torch.cat([ls.obsCamRot.data for ls in batch_ls])
        ),
        "camera_translation": th.Point3(
            data=torch.cat([ls.obsCamTr.data for ls in batch_ls])
        ),
        "focal_length": th.Vector(
            data=torch.cat([ls.focalLength.data.unsqueeze(1) for ls in batch_ls]),
            name="focal_length",
        ),
    }

    # batch of 3d points and 2d feature points
    for i in range(len(batch_ls[0].worldPoints)):
        batch_data[f"world_point_{i}"] = th.Point3(
            data=torch.cat([ls.worldPoints[i : i + 1].data for ls in batch_ls]),
            name=f"world_point_{i}",
        )
        batch_data[f"image_feature_point_{i}"] = th.Point2(
            data=torch.cat(
                [ls.image_feature_points[i : i + 1].data for ls in batch_ls]
            ),
            name=f"image_feature_point_{i}",
        )

    gtCamRot = th.SO3(data=torch.cat([ls.gtCamRot.data for ls in batch_ls]))
    gtCamTr = th.Point3(data=torch.cat([ls.gtCamTr.data for ls in batch_ls]))
    return batch_data, gtCamRot, gtCamTr


# Outer optimization loop
loss_radius_tensor = torch.nn.Parameter(torch.tensor([-3], dtype=torch.float64))
model_optimizer = torch.optim.Adam([loss_radius_tensor], lr=1.0)


num_epochs = 10
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
