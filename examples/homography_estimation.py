# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import hydra
from hydra.utils import get_original_cwd
import kornia
from typing import List
import glob
import theseus as th
import torch
from PIL import Image
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset


from libs.easyaug import RandomGeoAug, GeoAugParam

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SZ = 0.5
FONT_PT = (5, 15)


# TODO
# - batch the masked_select

class HomographyDataset(Dataset):
    def __init__(self, img_dir, imgH, imgW):
        self.img_dir = img_dir
        self.imgH = imgH
        self.imgW = imgW
        self.img_paths = glob.glob(img_dir + '/**/*.jpg', recursive=True)
        assert len(self.img_paths) > 0, "no images found"
        sc = 0.3
        self.rga = RandomGeoAug(
            rotate_param=GeoAugParam(min=-30 * sc, max=30 * sc),
            scale_param=GeoAugParam(min=0.8 * (1.0 + sc), max=1.2 * (1.0 + sc)),
            translate_x_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            translate_y_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            shear_x_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            shear_y_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            perspective_param=GeoAugParam(min=-0.1 * sc, max=0.1 * sc),
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img1 = np.asarray(Image.open(img_path).resize(size=(self.imgW, self.imgH)))
        img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        img2, H_1_2 = self.rga.forward(
            img1, return_transform=True, normalize_returned_transform=True
        )
        data = {"img1": img1,
                "img2": img2,
                "H_1_2": H_1_2}

        return data


def warp_perspective_norm(H, img):
    height, width = img.shape[-2:]
    grid = kornia.utils.create_meshgrid(
        height, width, normalized_coordinates=True, device=H.device
    )
    Hinv = torch.inverse(H)
    warped_grid = kornia.geometry.transform.homography_warper.warp_grid(grid, Hinv)
    grid_sample = torch.nn.functional.grid_sample
    img2 = grid_sample(img, warped_grid, mode="bilinear", align_corners=True)
    return img2


def homography_error_fn(optim_vars: List[th.Manifold], aux_vars: List[th.Variable]):
    H_1_2 = optim_vars[0]
    img1, img2 = aux_vars

    img1_dst = warp_perspective_norm(H_1_2.data.reshape(-1, 3, 3), img1.data)
    loss = torch.nn.functional.mse_loss(img1_dst, img2.data, reduction="none")
    ones = warp_perspective_norm(
        H_1_2.data.reshape(-1, 3, 3), torch.ones_like(img1.data)
    )
    loss = loss.masked_select((ones > 0.9)).mean()
    loss = loss.reshape(1, 1)
    return loss


def put_text(img, text, top=True):
    if top:
        pt = FONT_PT
    else:
        pt = FONT_PT[0], int(img.shape[0] * 1.08 - FONT_PT[1])
    cv2.putText(img, text, pt, FONT, FONT_SZ, (255, 255, 255), 2, lineType=16)
    cv2.putText(img, text, pt, FONT, FONT_SZ, (0, 0, 0), 1, lineType=16)
    return img


def torch2cv2(img):
    out = (
        (img[0].permute(1, 2, 0) * 255.0)
        .data.cpu()
        .numpy()
        .astype(np.uint8)[:, :, ::-1]
    )
    out = np.ascontiguousarray(out)
    return out


def viz_warp(log_dir, img1, img2, img1_w, iteration, err):
    img_diff = torch2cv2(torch.abs(img1_w - img2))
    img1 = torch2cv2(img1)
    img2 = torch2cv2(img2)
    img1_w = torch2cv2(img1_w)
    img1 = put_text(img1, "image I")
    img2 = put_text(img2, "image I'")
    img1_w = put_text(img1_w, "I warped to I'")
    img_diff = put_text(img_diff, "L2 diff")
    out = np.concatenate([img1, img2, img1_w, img_diff], axis=1)
    out = put_text(out, "iter: %05d, loss: %.3f" % (iteration, err), top=False)
    path = os.path.join(log_dir, "out_%05d.png" % iteration)
    cv2.imwrite(path, out)


def run(cfg):

    dataset_root = os.path.join(get_original_cwd(), "data")
    chunk = "revisitop1m.1"
    dataset_path = os.path.join(dataset_root, chunk)
    if not os.path.exists(dataset_path):
        print("Downloading data")
        url_root = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/"
        tar = "%s.tar.gz" % chunk
        cmd = "wget %s/%s -O %s/%s" % (url_root, tar, dataset_root, tar)
        print("Running command: ", cmd)
        os.system(cmd)
        os.makedirs(dataset_path)
        cmd = "tar -xf %s/%s -C %s" % (dataset_root, tar, dataset_path)
        print("Running command: ", cmd)
        os.system(cmd)

    imgH, imgW = 160, 200
    dataset = HomographyDataset(dataset_path, imgH, imgW)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data = next(iter(dataloader))

    # TODO: support batch dims.
    img1 = data["img1"][0]
    img2 = data["img2"][0]
    H_1_2 = data["H_1_2"][0]

    log_dir = "viz"
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"

    objective = th.Objective()

    H_init = torch.eye(3).reshape(1, 9)
    H_1_2 = th.Vector(data=H_init, name="H_1_2")

    img1 = th.Variable(data=img1, name="img1")
    img2 = th.Variable(data=img2, name="img2")

    homography_cf = th.AutoDiffCostFunction(
        optim_vars=[H_1_2],
        err_fn=homography_error_fn,
        dim=1,
        aux_vars=[img1, img2],
    )

    homography_cf.to(device)

    objective.add(homography_cf)

    optimizer = th.LevenbergMarquardt(  # GaussNewton(
        objective,
        max_iterations=1000,
        step_size=0.1,
    )

    theseus_layer = th.TheseusLayer(optimizer)
    theseus_layer.to(device)

    inputs = {
        "H_1_2": H_1_2.data,
        "img1": img1.data,
        "img2": img2.data,
    }
    info = theseus_layer.forward(inputs, optimizer_kwargs={"verbose": True})

    # save warped image
    img1_dst = warp_perspective_norm(H_1_2.data.reshape(-1, 3, 3), img1.data)
    viz_warp(
        log_dir,
        img1.data,
        img2.data,
        img1_dst,
        info[1].converged_iter.item(),
        float(objective.error().item()),
    )


@ hydra.main(config_path="./configs/", config_name="homography_estimation")
def main(cfg):
    torch.manual_seed(cfg.seed)
    run(cfg)


if __name__ == "__main__":
    main()
