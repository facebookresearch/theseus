import math
from typing import cast

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
camRot = th.SO3(
    torch.cat([randomSmallQuaternion(max_degrees=20).unsqueeze(0) for _ in range(4)]),
    name="camRot",
)
camTr = th.Point3(data=torch.zeros((4, 3), dtype=torch.float64), name="camTr")
camTr.data[:, 2] += 5.0
focalLenght = th.Vector(
    data=torch.tensor([1000], dtype=torch.float64).repeat(4).unsqueeze(1),
    name="focalLength",
)
lossRadius = th.Vector(
    data=torch.tensor([0], dtype=torch.float64).repeat(4).unsqueeze(1),
    name="loss_radius",
)
worldPoint = th.Point3(data=torch.rand((4, 3), dtype=torch.float64), name="worldPoint")
camPoint = camRot.rotate(worldPoint) + camTr
imageFeaturePoint = th.Point3(
    data=camPoint[:, :2] / camPoint[:, 2:] + torch.rand((4, 2)) * 50,
    name="imageFeaturePoint",
)
r = theg.ReprojectionError(
    camRot, camTr, focalLenght, lossRadius, worldPoint, imageFeaturePoint
)

baseVal = r.error()
baseCamRot = r.camera_rotation.copy()
baseCamTr = r.camera_translation.copy()
nErr = baseVal.shape[1]
nJac = torch.zeros((r.camera_rotation.data.shape[0], nErr, 6), dtype=torch.float64)
epsilon = 1e-8
for i in range(6):
    if i >= 3:
        r.camera_translation = baseCamTr.copy()
        r.camera_translation.data[:, i - 3] += epsilon
        r.camera_rotation = baseCamRot.copy()
    else:
        r.camera_translation = baseCamTr.copy()
        v = torch.zeros((r.camera_rotation.data.shape[0], 3), dtype=torch.float64)
        v[:, i] += epsilon
        r.camera_rotation = cast(th.SO3, baseCamRot.retract(v))
    pertVal = r.error()
    nJac[:, :, i] = (pertVal - baseVal) / epsilon

rotNumJac = nJac[:, :, :3]
trNumJac = nJac[:, :, 3:]

(rotJac, trJac), _ = r.jacobians()

print(
    "|numJac-analiticJac|: ",
    float(torch.norm(rotNumJac - rotJac)),
    float(torch.norm(trNumJac - trJac)),
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
            data=torch.tensor([focalLength], dtype=torch.float64), name="focalLength"
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
        self.imageFeaturePoints = add_noise_and_outliers(projPoints)

        smallRot = th.SO3(randomSmallQuaternion(max_degrees=0.3))
        smallTr = torch.rand(3, dtype=torch.float64) * 0.1
        self.obsCamRot = smallRot.compose(self.gtCamRot).copy(new_name="obsCamRot")
        self.obsCamTr = (smallRot.rotate(self.gtCamTr) + smallTr).copy(
            new_name="obsCamTr"
        )


localization_sample = LocalizationSample()


# create optimization problem
camRot = localization_sample.obsCamRot.copy(new_name="camRot")
camTr = localization_sample.obsCamTr.copy(new_name="camTr")
lossRadius = th.Vector(1, name="loss_radius", dtype=torch.float64)
focalLength = th.Vector(1, name="focal_length", dtype=torch.float64)

# NOTE: if not set explicitly will crash using a weight of wrong type `float32`
weight = th.ScaleCostWeight(
    th.Vector(data=torch.tensor([1.0], dtype=torch.float64), name="weight")
)

# Set up objective
objective = th.Objective(dtype=torch.float64)

for i in range(len(localization_sample.worldPoints)):
    worldPoint = th.Point3(
        data=localization_sample.worldPoints[i], name=f"worldPoint_{i}"
    )
    imageFeaturePoint = th.Point3(
        data=localization_sample.imageFeaturePoints[i], name=f"imageFeaturePoint_{i}"
    )

    # optim_vars = [camRot, camTr]
    # aux_vars = [lossRadius, focalLength, worldPoint, imageFeaturePoint]
    cost_function = theg.ReprojectionError(
        camRot,
        camTr,
        focalLength,
        lossRadius,
        worldPoint,
        imageFeaturePoint,
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
# NOTE: composition of SO3 rotations is often not a valid rotation (.copy fails)
loc_samples = [LocalizationSample() for _ in range(16)]
batch_size = 4
num_batches = (len(loc_samples) + batch_size - 1) // batch_size


def get_batch(b):
    assert b * batch_size < len(loc_samples)
    batch_ls = loc_samples[b * batch_size : (b + 1) * batch_size]
    batch_data = {
        "camRot": th.SO3(data=torch.cat([ls.obsCamRot.data for ls in batch_ls])),
        "camTr": th.Point3(data=torch.cat([ls.obsCamTr.data for ls in batch_ls])),
        "focal_length": th.Vector(
            data=torch.cat([ls.focalLength.data.unsqueeze(1) for ls in batch_ls]),
            name="focalLength",
        ),
    }

    # batch of 3d points and 2d feature points
    for i in range(len(batch_ls[0].worldPoints)):
        batch_data[f"worldPoint_{i}"] = th.Point3(
            data=torch.cat([ls.worldPoints[i : i + 1].data for ls in batch_ls]),
            name=f"worldPoint_{i}",
        )
        batch_data[f"imageFeaturePoint_{i}"] = th.Point3(
            data=torch.cat([ls.imageFeaturePoints[i : i + 1].data for ls in batch_ls]),
            name=f"imageFeaturePoint_{i}",
        )

    gtCamRot = th.SO3(data=torch.cat([ls.gtCamRot.data for ls in batch_ls]))
    gtCamTr = th.Point3(data=torch.cat([ls.gtCamTr.data for ls in batch_ls]))
    return batch_data, gtCamRot, gtCamTr


# Outer optimization loop
lossRadius_tensor = torch.nn.Parameter(torch.tensor([-3], dtype=torch.float64))
model_optimizer = torch.optim.Adam([lossRadius_tensor], lr=1.0)


num_epochs = 10
camRotVar = theseus_optim.objective.optim_vars["camRot"]
camTrVar = theseus_optim.objective.optim_vars["camTr"]
for epoch in range(num_epochs):
    print(" ******************* EPOCH {epoch} ******************* ")
    epoch_loss = 0.0
    for i in range(num_batches):
        print(f"BATCH {i}/{num_batches}")
        model_optimizer.zero_grad()
        theseus_inputs, gtCamRot, gtCamTr = get_batch(i)
        theseus_inputs["loss_radius"] = lossRadius_tensor.repeat(
            gtCamTr.data.shape[0]
        ).unsqueeze(1)

        theseus_outputs, info = theseus_optim.forward(
            theseus_inputs, optimizer_kwargs={"verbose": False}
        )

        cam_rot_loss = th.local(camRotVar, gtCamRot).norm(dim=1)
        cam_tr_loss = th.local(camTrVar, gtCamTr).norm(dim=1, p=1)
        loss = 100 * cam_rot_loss + cam_tr_loss
        loss = torch.where(loss < 10e5, loss, 0.0).sum()
        loss.backward()
        model_optimizer.step()

        loss_value = torch.sum(loss.detach()).item()
        epoch_loss += loss_value

    print(
        f"Epoch: {epoch} Loss: {epoch_loss} "
        f"Kernel Radius: exp({lossRadius_tensor.data.item()})="
        f"{torch.exp(lossRadius_tensor.data).item()}"
    )
