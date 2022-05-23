# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import kornia
from typing import List
import glob
import theseus as th
import torch
from PIL import Image
import cv2
import numpy as np
import os
import time
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import shutil
from torch.utils.tensorboard import SummaryWriter

from libs.easyaug import RandomGeoAug, GeoAugParam, RandomPhotoAug

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SZ = 0.5
FONT_PT = (5, 15)


BACKWARD_MODE = {
    "implicit": th.BackwardMode.IMPLICIT,
    "full": th.BackwardMode.FULL,
    "truncated": th.BackwardMode.TRUNCATED,
}


def prepare_data():
    #dataset_root = os.path.join(get_original_cwd(), "data")
    dataset_root = os.path.join(os.getcwd(), "data")
    chunks = [
        "revisitop1m.1",
        "revisitop1m.2",
        "revisitop1m.3",
        "revisitop1m.4",
        "revisitop1m.5",
    ]
    dataset_paths = []
    for chunk in chunks:
        dataset_path = os.path.join(dataset_root, chunk)
        dataset_paths.append(dataset_path)
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

    return dataset_paths


class HomographyDataset(Dataset):
    def __init__(self, img_dirs, imgH, imgW, photo_aug=True, train=True):
        self.imgH = imgH
        self.imgW = imgW
        self.img_paths = []
        for direc in img_dirs:
            self.img_paths.extend(glob.glob(direc + "/**/*.jpg", recursive=True))
        assert len(self.img_paths) > 0, "no images found"
        print("Found %d total images in dataset" % len(self.img_paths))
        sc = 0.3
        self.rga = RandomGeoAug(
            rotate_param=GeoAugParam(min=-30 * sc, max=30 * sc),
            scale_param=GeoAugParam(min=0.8 * (1.0 - sc), max=1.2 * (1.0 + sc)),
            translate_x_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            translate_y_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            shear_x_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            shear_y_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            perspective_param=GeoAugParam(min=-0.1 * sc, max=0.1 * sc),
        )
        self.photo_aug = photo_aug
        if self.photo_aug:
            self.rpa = RandomPhotoAug()

        # train test split
        self.img_paths.sort()
        max_images = 99999
        self.img_paths = self.img_paths[:max_images]
        split_ix = int(0.9 * len(self.img_paths))
        if train:
            self.img_paths = self.img_paths[:split_ix]
        else:
            self.img_paths = self.img_paths[split_ix:]
        self.train = train
        if self.train:
            print("Using %d images for training" % len(self.img_paths))
        else:
            print("Using %d images for testing" % len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img1 = np.asarray(Image.open(img_path).resize(size=(self.imgW, self.imgH)))
        assert img1.shape == (self.imgH, self.imgW, 3)
        img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        img2, H_1_2 = self.rga.forward(
            img1, return_transform=True, normalize_returned_transform=True
        )

        # apply random photometric augmentations
        if self.photo_aug:
            seed1, seed2 = None, None
            if self.train is False:
                seed1 = idx
                seed2 = idx + 1
            img1 = torch.clamp(img1, 0.0, 1.0)
            img2 = torch.clamp(img2, 0.0, 1.0)
            img1 = self.rpa.forward(img1, seed=seed1)
            img2 = self.rpa.forward(img2, seed=seed2)

        data = {"img1": img1[0], "img2": img2[0], "H_1_2": H_1_2[0]}

        return data


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(
        image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)
    )

    out_val = (
        nw_val.view(N, C, H, W) * nw.view(N, 1, H, W)
        + ne_val.view(N, C, H, W) * ne.view(N, 1, H, W)
        + sw_val.view(N, C, H, W) * sw.view(N, 1, H, W)
        + se_val.view(N, C, H, W) * se.view(N, 1, H, W)
    )

    return out_val


def warp_perspective_norm(H, img):
    height, width = img.shape[-2:]
    grid = kornia.utils.create_meshgrid(
        height, width, normalized_coordinates=True, device=H.device
    )
    Hinv = torch.inverse(H)
    warped_grid = kornia.geometry.transform.homography_warper.warp_grid(grid, Hinv)
    img2 = grid_sample(img, warped_grid)
    return img2


def homography_error_fn(optim_vars: List[th.Manifold], aux_vars: List[th.Variable]):
    H_1_2 = optim_vars[0]
    img1, img2 = aux_vars

    ones = torch.ones(H_1_2.shape[0], 1, device=H_1_2.device, dtype=H_1_2.dtype)
    H_1_2_mat = torch.cat((H_1_2.data, ones), dim=1).reshape(-1, 3, 3)
    img1_dst = warp_perspective_norm(H_1_2_mat, img1.data)
    loss = torch.nn.functional.mse_loss(img1_dst, img2.data, reduction="none")
    mask = warp_perspective_norm(H_1_2_mat, torch.ones_like(img1.data))
    mask = mask > 0.9
    loss = (loss * mask).mean(dim=1)
    loss = loss.reshape(loss.shape[0], -1)
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
    out = (img.permute(1, 2, 0) * 255.0).data.cpu().numpy().astype(np.uint8)[:, :, ::-1]
    out = np.ascontiguousarray(out)
    return out


def torch2cv2_border(img):
    out = (img.permute(1, 2, 0) * 255.0).data.cpu().numpy().astype(np.uint8)[:, :, ::-1]
    out = np.ascontiguousarray(out)
    h, w = out.shape[:2]
    blank = np.full([h * 2, w * 2, 3], 255).astype(np.uint8)
    blank[int(h / 2) : int(3 * h / 2), int(w / 2) : int(3 * w / 2)] = out
    return blank


def viz_warp(path, img1, img2, img1_w, iteration, err):
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
    cv2.imwrite(path, out)

def write_gif_batch(save_dir, state_history, img1, img2, loss, ix=0):
    for it in state_history:
        H = state_history[it]["H_1_2"].data
        ones = torch.ones(H.shape[0], 1, device=H.device, dtype=H.dtype)
        H_1_2_mat = torch.cat((H.data, ones), dim=1).reshape(-1, 3, 3)
        img1_dsts = warp_perspective_norm(H_1_2_mat, img1)

        for j, H in enumerate(H_1_2_mat):
            if it == 0:
                os.makedirs(f"{save_dir}/out{j:03d}")
            path = os.path.join(save_dir, f"out{j:03d}/{it:05d}.png")
            viz_warp(
                path,
                img1[j],
                img2[j],
                img1_dsts[j],
                it,
                loss[j].item(),
            )

    for j in range(len(H_1_2_mat)):
        cmd = f"convert -delay 10 -loop 0 {save_dir}/out{j:03d}/*.png {save_dir}/img{ix:03d}.gif"
        print(cmd)
        os.system(cmd)
        shutil.rmtree(f"{save_dir}/out{j:03d}")
        ix += 1



#class SimpleCNN(nn.Module):
#    def __init__(self, D=32):
#        super(SimpleCNN, self).__init__()
#        self.relu = nn.ReLU(inplace=True)
#        self.conv1 = nn.Conv2d(3, D, kernel_size=3, stride=1, padding=1)
#        self.conv2 = nn.Conv2d(D, 3, kernel_size=3, stride=1, padding=1)
#        self.bn1 = nn.BatchNorm2d(D)
#
#    def forward(self, img):
#        x = self.relu(self.bn1(self.conv1(img)))
#        return self.conv2(x)

def run():

    B = 1
    C = 3
    max_iters = 3
    #imgH, imgW = 120, 160
    imgH, imgW = 60, 80
    use_gpu = True

    dataset_paths = prepare_data()

    train_dataset = HomographyDataset(dataset_paths, imgH, imgW, train=True)



    log_dir = "viz"
    os.makedirs(log_dir, exist_ok=True)
    print("Writing to tensorboard at", os.getcwd())
    writer = SummaryWriter(os.getcwd())

    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    print("Using device %s" % device)

    ## A simple 2-layer CNN network that maintains the original image size.
    #cnn_model = SimpleCNN()
    #cnn_model.to(device)
    #cnn_model_feat = torch.nn.Sequential(*list(cnn_model.children()))
    #fixed_feat = torch.nn.Sequential(*list(cnn_model.children()))

    train_loader = DataLoader(
        train_dataset, batch_size=B, shuffle=True
    )
    data = next(iter(train_loader))

    #D = cnn_model(data["img1"].to(device)).shape[1]

    #l1_loss = torch.nn.L1Loss()

    # Setup theseus.
    objective = th.Objective()
    H_init = torch.zeros(B, 8)
    H_1_2 = th.Vector(data=H_init, name="H_1_2")
    img_init = torch.zeros(B, C, imgH, imgW)
    img1 = th.Variable(data=img_init, name="img1")
    img2 = th.Variable(data=img_init, name="img2")
    error_dim = img_init[0, 0].numel()
    print(f"Error dim {error_dim}")
    def error_fn(optim_vars: List[th.Manifold], aux_vars: List[th.Variable]):
        return homography_error_fn(optim_vars, aux_vars)
    homography_cf = th.AutoDiffCostFunction(
        optim_vars=[H_1_2],
        err_fn=error_fn,
        dim=error_dim,
        aux_vars=[img1, img2],
    )
    objective.add(homography_cf)
    inner_optimizer = th.GaussNewton(
        objective,
        max_iterations=max_iters,
        step_size=0.1,
    )
    theseus_layer = th.TheseusLayer(inner_optimizer)
    theseus_layer.to(device)

    #cnn_model.train()

    H_1_2_gt = data["H_1_2"].to(device)
    img1 = data["img1"].to(device)
    img2 = data["img2"].to(device)
    #feat1 = cnn_model(img1)
    #feat2 = cnn_model(img2)
    feat1 = img1
    feat2 = img2
    H_init = torch.eye(3).reshape(1, 9).repeat(img1.shape[0], 1).to(device)
    H_init = H_init[:, :-1]
    inputs = {"H_1_2": H_init, "img1": feat1, "img2": feat2,}
    theseus_layer.forward(
        inputs,
        optimizer_kwargs={
            "verbose": True,
            "backward_mode": BACKWARD_MODE["truncated"],
            "backward_num_iterations": max_iters,
            "__keep_final_step_size__": True,
        },
    )

    H_1_2_gt = H_1_2_gt.reshape(-1, 9)[:, :-1]
    H_1_2 = theseus_layer.objective.get_optim_var("H_1_2")
    import pdb
    pdb.set_trace()


def main():
    torch.manual_seed(0)
    run()


if __name__ == "__main__":
    main()
