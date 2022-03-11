import torch
import theseus as th

import theseus.utils.examples as theg
from theseus.utils.examples.bundle_adjustment.util import random_small_quaternion

# unit test for Cost term
batch_size = 4
camRot = th.SO3(torch.cat([random_small_quaternion(max_degrees = 20).unsqueeze(0) for _ in range(batch_size)]), name="camRot")
camTr = th.Point3(data=torch.zeros((batch_size, 3), dtype=torch.float64), name="camTr")
camTr.data[:, 2] += 5.0
focalLenght = th.Vector(data=torch.tensor([1000], dtype=torch.float64).repeat(batch_size).unsqueeze(1), name="focalLength")
lossRadius = th.Vector(data=torch.tensor([0], dtype=torch.float64).repeat(batch_size).unsqueeze(1), name="lossRadius")
worldPoint = th.Vector(data=torch.rand((batch_size, 3), dtype=torch.float64), name="worldPoint")
camPoint = camRot.rotate(worldPoint) + camTr
imageFeaturePoint = th.Vector(data=camPoint[:, :2] / camPoint[:, 2:] + torch.rand((batch_size,2)) * 50, name="imageFeaturePoint")
r = theg.ReprojectionError(
          camera_rotation=camRot, camera_translation=camTr,
          focal_length=focalLenght,
          loss_radius=lossRadius,
          world_point=worldPoint,
          image_feature_point=imageFeaturePoint)

baseVal = r.error()
baseCamRot = r.camera_rotation.copy()
baseCamTr = r.camera_translation.copy()
nErr = baseVal.shape[1]
nJac = torch.zeros((r.camera_rotation.data.shape[0], nErr, 6), dtype=torch.float64)
epsilon = 1e-8
for i in range(6):
    if i >= 3:
        r.camTr = baseCamTr.copy()
        r.camTr.data[:, i - 3] += epsilon
        r.camRot = baseCamRot.copy()
    else:
        r.camTr = baseCamTr.copy()
        v = torch.zeros((r.camera_rotation.data.shape[0], 3), dtype=torch.float64)
        v[:, i] += epsilon
        r.camRot = baseCamRot.retract(v)
    pertVal = r.error()
    nJac[:, :, i] = (pertVal - baseVal) / epsilon

rotNumJac = nJac[:, :, :3]
trNumJac = nJac[:, :, 3:]

(rotJac, trJac), _ = r.jacobians()

print("|numJac-analiticJac|: ",
    float(torch.norm(rotNumJac - rotJac)), float(torch.norm(trNumJac - trJac)))