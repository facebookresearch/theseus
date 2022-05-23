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
    dataset_root = os.path.join(get_original_cwd(), "data")
    chunks = [
        "revisitop1m.1",
        "revisitop1m.2",
        "revisitop1m.3",
        "revisitop1m.4",
        "revisitop1m.5",
        # "revisitop1m.6",
        # "revisitop1m.7",
        # "revisitop1m.8",
        #"revisitop1m.9",
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

    bad_files = [
        "/revisitop1m.9/171/1712c98e7f971fb9a272ad61c604ee2.jpg",
        "/revisitop1m.9/176/176185b2431ac72f6419ab30bd78705c.jpg",
        "/revisitop1m.9/162/162b144da5bf789a5e23feb1dfa6a391.jpg",
        "/revisitop1m.9/173/1733685218bf22d514b9c3b4bf2c2027.jpg",
        "/revisitop1m.9/158/1580dc2f3479ae44531a477f892850.jpg",
        "/revisitop1m.9/16f/16f9772f0da348ee99cdf8f0e975d3.jpg",
        "/revisitop1m.9/152/152f374b398b3ba3a1c843df036df.jpg",
        "/revisitop1m.9/165/165f26d34914dbbab7bbdd3eac694333.jpg",
        "/revisitop1m.9/174/17442dbe499594c7a54cca7bd171b58.jpg",
        "/revisitop1m.9/16a/16acaf59321debe9250c0f6bc75e56d.jpg",
    ]
    for f in bad_files:
        if os.path.exists(dataset_root + f):
            os.remove(dataset_root + f)

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
        max_images = 500
        #max_images = 99999
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


def homography_error_fn(
    optim_vars: List[th.Manifold], aux_vars: List[th.Variable], pool_size: int
):
    H_1_2 = optim_vars[0]
    img1, img2 = aux_vars

    ones = torch.ones(H_1_2.shape[0], 1, device=H_1_2.device, dtype=H_1_2.dtype)
    H_1_2_mat = torch.cat((H_1_2.data, ones), dim=1).reshape(-1, 3, 3)
    img1_dst = warp_perspective_norm(H_1_2_mat, img1.data)
    loss = torch.nn.functional.mse_loss(img1_dst, img2.data, reduction="none")
    mask = warp_perspective_norm(H_1_2_mat, torch.ones_like(img1.data))
    mask = mask > 0.9
    loss = (loss * mask).mean(dim=1)
    loss = torch.nn.functional.avg_pool2d(loss, pool_size, stride=pool_size)
    loss = loss.reshape(loss.shape[0], -1)
    return loss


def four_corner_dist(H_1_2, H_1_2_gt, height, width):
    Hinv_gt = torch.inverse(H_1_2_gt)
    Hinv = torch.inverse(H_1_2)
    grid = kornia.utils.create_meshgrid(2, 2, device=Hinv.device)
    warped_grid = kornia.geometry.transform.homography_warper.warp_grid(grid, Hinv)
    warped_grid_gt = kornia.geometry.transform.homography_warper.warp_grid(
        grid, Hinv_gt
    )
    warped_grid = (warped_grid + 1) / 2
    warped_grid_gt = (warped_grid_gt + 1) / 2
    warped_grid[..., 0] *= width
    warped_grid[..., 1] *= height
    warped_grid_gt[..., 0] *= width
    warped_grid_gt[..., 1] *= height

    dist = torch.square(torch.norm(warped_grid - warped_grid_gt, dim=-1))
    dist = dist.mean(dim=-1).mean(dim=-1)
    return dist, warped_grid, warped_grid_gt


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


def viz_four_corner(path, img1, img2, H_1_2_mat, warped_grid, warped_grid_gt):
    h, w = img1.shape[1:]
    img1_w = warp_perspective_norm(H_1_2_mat[None, ...], img1[None, ...])[0]
    img_diff = torch2cv2_border(torch.abs(img1_w - img2))
    img1 = torch2cv2_border(img1)
    img2 = torch2cv2_border(img2)
    img1_w = torch2cv2_border(img1_w)
    img2 = put_text(img2, "image I'")
    img1_w = put_text(img1_w, "I warped to I'")
    img_diff = put_text(img_diff, "L2 diff")

    src_grid = kornia.utils.create_meshgrid(2, 2, normalized_coordinates=False)[0]
    src_grid[..., 0] *= w
    src_grid[..., 1] *= h
    offset = np.array([w / 2, h / 2]).astype(int)
    for point in src_grid.reshape(-1, 2):
        point = point.numpy().astype(int) + offset
        cv2.circle(img1, point, 3, (0, 0, 255), thickness=-1)
    for point in warped_grid.cpu().reshape(-1, 2):
        point = point.numpy().astype(int) + offset
        cv2.circle(img1, point, 3, (255, 0, 0), thickness=-1)
    for point in warped_grid_gt.cpu().reshape(-1, 2):
        point = point.numpy().astype(int) + offset
        cv2.circle(img1, point, 3, (0, 255, 0), thickness=-1)
    out = np.concatenate([img1, img2, img1_w, img_diff], axis=1)

    dist = torch.norm(warped_grid - warped_grid_gt, dim=-1)
    dist = dist.mean(dim=-1).mean(dim=-1)
    out = put_text(out, "four corner dist: %.3f" % (dist.item()))
    cv2.imwrite(path, out)


def four_corner_dist_hist(path, fcd_pho, fcd_fix, fcd_opt):
    plt.figure()
    plt.hist(
        fcd_pho[fcd_pho < 1000], bins=50, label="Photometric", color="C0", alpha=0.5
    )
    plt.hist(
        fcd_fix[fcd_fix < 1000],
        bins=50,
        label="Feature-metric fixed",
        color="C1",
        alpha=0.5,
    )
    plt.hist(
        fcd_opt[fcd_opt < 1000],
        bins=50,
        label="Feature-metric optimized",
        color="C2",
        alpha=0.5,
    )
    plt.xlabel("Four corner distance (pixels)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(path)


def four_corner_thresh(path, fcd_pho, fcd_fix, fcd_opt):
    thresholds = np.arange(1, 401)
    count_pho = [len(fcd_pho[fcd_pho < thr]) / len(fcd_pho) for thr in thresholds]
    count_fix = [len(fcd_fix[fcd_fix < thr]) / len(fcd_fix) for thr in thresholds]
    count_opt = [len(fcd_opt[fcd_opt < thr]) / len(fcd_opt) for thr in thresholds]
    plt.figure()
    plt.plot(thresholds, count_pho, label="Photometric", color="C0")
    plt.plot(thresholds, count_fix, label="Feature-metric fixed", color="C1")
    plt.plot(thresholds, count_opt, label="Feature-metric optimized", color="C2")
    plt.xlabel("Threshold four corner distance (pixels)")
    plt.ylabel("Proportion below threshold")
    # plt.xscale('log')
    plt.legend()
    plt.savefig(path)


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


def setup_homgraphy_layer(cfg, device, batch_size, channels):

    objective = th.Objective()

    H_init = torch.zeros(batch_size, 8)
    H_1_2 = th.Vector(data=H_init, name="H_1_2")

    img_init = torch.zeros(batch_size, channels, cfg.imgH, cfg.imgW)
    img1 = th.Variable(data=img_init, name="img1")
    img2 = th.Variable(data=img_init, name="img2")

    # loss is pooled so error dim is not too large
    #pool_size = int(cfg.imgH // 3)
    pool_size = 1
    pooled = torch.nn.functional.avg_pool2d(img_init, pool_size, stride=pool_size)
    error_dim = pooled[0, 0].numel()
    print(f"Pool size {pool_size}, error dim {error_dim}")

    def error_fn(optim_vars: List[th.Manifold], aux_vars: List[th.Variable]):
        return homography_error_fn(optim_vars, aux_vars, pool_size)

    homography_cf = th.AutoDiffCostFunction(
        optim_vars=[H_1_2],
        err_fn=error_fn,
        dim=error_dim,
        aux_vars=[img1, img2],
    )
    objective.add(homography_cf)

    reg_w = th.ScaleCostWeight(np.sqrt(cfg.inner_optim.reg_w))
    reg_w.to(dtype=H_init.dtype)
    H_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    identity_homography = th.Vector(data=H_identity, name="identity")
    objective.add(
        th.eb.VariableDifference(
            H_1_2, reg_w, identity_homography, name="reg_homography"
        )
    )

    optimizer = th.GaussNewton(
        objective,
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
    )

    theseus_layer = th.TheseusLayer(optimizer)
    theseus_layer.to(device)

    return theseus_layer


def train_loop(cfg, device, train_dataset, feat_model, writer=None, train_chkpt=None, save_dir=None):
    if cfg.save_model:
        os.makedirs("chkpts")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True
    )
    sample = next(iter(train_loader))
    feat_channels = feat_model(sample["img1"].to(device)).shape[1]

    model_optimizer = torch.optim.Adam(feat_model.parameters(), lr=cfg.outer_optim.lr)

    start_epoch = 0
    if train_chkpt is not None:
        if os.path.exists(train_chkpt):
            state_dict = torch.load(train_chkpt)
            feat_model.load_state_dict(state_dict["model_state_dict"])
            model_optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            start_epoch = state_dict["epoch"]
            print("Loading from checkpoint ", cfg.train_chkpt)

    l1_loss = torch.nn.L1Loss()

    layer_feat = setup_homgraphy_layer(
        cfg,
        device,
        cfg.train_batch_size,
        feat_channels,
    )

    feat_model.train()

    all_losses = []
    epoch_losses = []
    print(f"\n Starting training for {cfg.outer_optim.num_epochs - start_epoch} epochs")
    for epoch in range(start_epoch, cfg.outer_optim.num_epochs):
        running_losses = []
        start_time = time.time()
        for t, data in enumerate(train_loader):
            print("starting iteration %d" % t)
            start = time.time()

            H_1_2_gt = data["H_1_2"].to(device)
            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)
            feat1 = feat_model(img1)
            feat2 = feat_model(img2)

            H_init = torch.eye(3).reshape(1, 9).repeat(img1.shape[0], 1).to(device)
            H_init = H_init[:, :-1]
            inputs = {
                "H_1_2": H_init,
                "img1": feat1,
                "img2": feat2,
            }

            layer_feat.forward(
                inputs,
                optimizer_kwargs={
                    "verbose": False,
                    "backward_mode": BACKWARD_MODE[cfg.inner_optim.backward_mode],
                    "__keep_final_step_size__": True,
                },
            )

            # no loss on last element as set to one
            H_1_2_gt = H_1_2_gt.reshape(-1, 9)[:, :-1]
            H_1_2 = layer_feat.objective.get_optim_var("H_1_2")
            loss = l1_loss(H_1_2.data, H_1_2_gt)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            running_losses.append(loss.item())
            all_losses.append(loss.item())

            elapsed = time.time() - start
            print(
                f"Step {t} / {len(train_loader)}. Loss {loss.item():.4f}. Time {elapsed}."
            )
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), t)

            if t % 200 == 0 and cfg.save_model:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": feat_model.state_dict(),
                        "optimizer_state_dict": model_optimizer.state_dict(),
                    },
                    f"chkpts/epoch_{epoch:03d}_step_{t:03d}.pt",
                )

            B = img1.shape[0]
            if cfg.save_gifs and t % 5 == 0:
                write_gif_batch(
                    save_dir,
                    layer_feat.optimizer.state_history,
                    img1,
                    img2,
                    loss.reshape(B),
                    t,
                )

        epoch_time = time.time() - start_time
        epoch_loss = np.mean(running_losses).item()
        epoch_losses.append(epoch_loss)
        print(
            f"******* Epoch {epoch}. Average loss: {epoch_loss:.4f}. "
            f"Epoch time {epoch_time}*******"
        )

    return feat_model


class SimpleCNN(nn.Module):
    def __init__(self, D=32):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, D, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(D, 3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(3)

    def forward(self, img):
        x = self.relu(self.bn1(self.conv1(img)))
        return self.conv2(x)

def run(cfg):

    dataset_paths = prepare_data()

    train_dataset = HomographyDataset(dataset_paths, cfg.imgH, cfg.imgW, train=True)
    test_dataset = HomographyDataset(dataset_paths, cfg.imgH, cfg.imgW, train=False)

    log_dir = "viz"
    os.makedirs(log_dir, exist_ok=True)
    print("Writing to tensorboard at", os.getcwd())
    writer = SummaryWriter(os.getcwd())

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"
    print("Using device %s" % device)

    # Resnet18, chopping off the final layers, resulting in smaller output image size.
    #net = models.resnet18(pretrained=True)
    #net.to(device)
    #upsample = torch.nn.UpsamplingBilinear2d(size=[cfg.imgH, cfg.imgW])
    #net_feat = torch.nn.Sequential(*list(net.children())[:-4] + [upsample])
    #fixed_feat = torch.nn.Sequential(*list(net.children())[:-4] + [upsample])
    # A simple 2-layer CNN network that maintains the original image size.
    net = SimpleCNN()
    net.to(device)
    net_feat = torch.nn.Sequential(*list(net.children()))
    fixed_feat = torch.nn.Sequential(*list(net.children()))

    # Training loop to refine pretrained features
    net_feat = train_loop(
        cfg,
        device,
        train_dataset,
        net_feat,
        writer=writer,
        train_chkpt=cfg.train_chkpt,
        save_dir=log_dir + "/viz",
    )

    ## Test loop
    #res_dir = "res"
    #os.makedirs(res_dir)

    #loss_pho, fcd_pho = run_eval(cfg, device, test_dataset, save_dir=log_dir + "/pho")
    #np.savetxt(f"{res_dir}/loss_pho.txt", loss_pho)
    #np.savetxt(f"{res_dir}/fcd_pho.txt", fcd_pho)
    ## loss_pho = np.loadtxt(get_original_cwd() + "/outputs/homography_res/loss_pho.txt")
    ## fcd_pho = np.loadtxt(get_original_cwd() + "/outputs/homography_res/fcd_pho.txt")

    #loss_fix, fcd_fix = run_eval(
    #    cfg,
    #    device,
    #    test_dataset,
    #    save_dir=log_dir + "/feat_fix",
    #    feat_model=fixed_feat,
    #)
    #np.savetxt(f"{res_dir}/loss_fix.txt", loss_fix)
    #np.savetxt(f"{res_dir}/fcd_fix.txt", fcd_fix)
    ## loss_fix = np.loadtxt(get_original_cwd() + "/outputs/homography_res/loss_fix.txt")
    ## fcd_fix = np.loadtxt(get_original_cwd() + "/outputs/homography_res/fcd_fix.txt")

    ## print("Loading eval model from checkpoint ", cfg.train_chkpt)
    ## net_feat.load_state_dict(torch.load(cfg.eval_chkpt)["model_state_dict"])
    #loss_opt, fcd_opt = run_eval(
    #    cfg,
    #    device,
    #    test_dataset,
    #    save_dir=log_dir + "/feat_opt",
    #    feat_model=net_feat,
    #)
    #np.savetxt(f"{res_dir}/loss_opt.txt", loss_opt)
    #np.savetxt(f"{res_dir}/fcd_opt.txt", fcd_opt)
    # loss_opt = np.loadtxt(get_original_cwd() + "/outputs/homography_res/loss_opt.txt")
    # fcd_opt = np.loadtxt(get_original_cwd() + "/outputs/homography_res/fcd_opt.txt")

    # print("\n\nResults ---------------------------------------")
    # print("\nMean and median photometric loss over the test dataset:")
    # print(f"Photometric: {np.mean(loss_pho):.3f}, {np.median(loss_pho):.3f}")
    # print(f"Feature-metric fixed: {np.mean(loss_fix):.3f}, {np.median(loss_fix):.3f}")
    # print(f"Feature-metric optimised: {np.mean(loss_opt):.3f}, {np.median(loss_opt):.3f}")

    # print("\nMean and median four corner distance over the test dataset (pixels):")
    # print(f"Photometric: {np.mean(fcd_pho):.3f}, {np.median(fcd_pho):.3f}")
    # print(f"Feature-metric fixed: {np.mean(fcd_fix):.3f}, {np.median(fcd_fix):.3f}")
    # print(f"Feature-metric optimised: {np.mean(fcd_opt):.3f}, {np.median(fcd_opt):.3f}")

    # # four corner plots
    # plot_path = get_original_cwd() + "/outputs/homography_res/"
    # four_corner_dist_hist(plot_path + "fcd_hist.png", fcd_pho, fcd_fix, fcd_opt)
    # four_corner_thresh(plot_path + "fcd_threshold.png", fcd_pho, fcd_fix, fcd_opt)


@hydra.main(config_path="./configs/", config_name="homography_estimation")
def main(cfg):
    torch.manual_seed(cfg.seed)
    run(cfg)


if __name__ == "__main__":
    main()
