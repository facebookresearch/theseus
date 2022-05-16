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
import torchvision.models as models
import matplotlib.pyplot as plt
import shutil

from libs.easyaug import RandomGeoAug, GeoAugParam, RandomPhotoAug

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SZ = 0.5
FONT_PT = (5, 15)


BACKWARD_MODE = {
    "implicit": th.BackwardMode.IMPLICIT,
    "full": th.BackwardMode.FULL,
    "truncated": th.BackwardMode.TRUNCATED,
}


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
            scale_param=GeoAugParam(min=0.8 * (1.0 + sc), max=1.2 * (1.0 + sc)),
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
        cmd = f"convert -delay 10 -loop 0 {save_dir}/out{j:03d}/*.png {save_dir}/img{ix}.gif"
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
    pool_size = int(cfg.imgH // 3)
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

    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
    )

    theseus_layer = th.TheseusLayer(optimizer)
    theseus_layer.to(device)

    return theseus_layer


def train_loop(cfg, device, train_dataset, feat_model):
    if cfg.save_model:
        os.makedirs("chkpts")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True
    )
    sample = next(iter(train_loader))
    feat_channels = feat_model(sample["img1"].to(device)).shape[1]

    model_optimizer = torch.optim.Adam(feat_model.parameters(), lr=cfg.outer_optim.lr)

    l1_loss = torch.nn.L1Loss()

    layer_feat = setup_homgraphy_layer(
        cfg,
        device,
        cfg.train_batch_size,
        feat_channels,
    )

    ax = plt.axes()

    all_losses = []
    epoch_losses = []
    for epoch in range(cfg.outer_optim.num_epochs):
        running_losses = []
        start_time = time.time()
        for t, data in enumerate(train_loader):

            H_1_2_gt = data["H_1_2"].to(device)
            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)
            feat1 = feat_model(img1)
            feat2 = feat_model(img2)

            H_init = (
                torch.eye(3).reshape(1, 9).repeat(cfg.train_batch_size, 1).to(device)
            )
            H_init = H_init[:, :-1]
            inputs = {
                "H_1_2": H_init,
                "img1": feat1,
                "img2": feat2,
            }

            success = True
            try:
                layer_feat.forward(
                    inputs,
                    optimizer_kwargs={
                        "verbose": True,
                        "backward_mode": BACKWARD_MODE[cfg.inner_optim.backward_mode],
                        "damping": cfg.inner_optim.lm_damping,
                        "__keep_final_step_size__": True,
                    },
                )
            except Exception as e:
                print(e)
                success = False
                with torch.no_grad():
                    torch.cuda.empty_cache()

            # no loss on last element as set to one
            H_1_2_gt = H_1_2_gt.reshape(-1, 9)[:, :-1]
            H_1_2 = layer_feat.objective.get_optim_var("H_1_2")
            loss = l1_loss(H_1_2.data, H_1_2_gt)
            model_optimizer.zero_grad()
            if success:
                loss.backward()
            model_optimizer.step()
            running_losses.append(loss.item())
            all_losses.append(loss.item())

            print(f"Step {t} / {len(train_loader)}. Loss {loss.item():.4f}")

            ax.scatter(np.arange(len(all_losses)), all_losses, color="C0")
            plt.xlim(0, len(all_losses) + 1)
            plt.ylim(0, np.max(all_losses))
            plt.savefig("loss_curve.png")

        epoch_time = time.time() - start_time
        epoch_loss = np.mean(running_losses).item()
        epoch_losses.append(epoch_loss)
        print(
            f"******* Epoch {epoch}. Average loss: {epoch_loss:.4f}. "
            f"Epoch time {epoch_time}*******"
        )

        if cfg.save_model:
            torch.save(feat_model.state_dict(), f"chkpts/epoch_{epoch:03d}.pt")

    return feat_model


def run_eval(cfg, device, test_dataset, save_dir=None, feat_model=None):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.test_batch_size, shuffle=False
    )
    channels = 3
    if feat_model is not None:
        sample = next(iter(test_loader))
        channels = feat_model(sample["img1"].to(device)).shape[1]

    theseus_layer = setup_homgraphy_layer(
        cfg,
        device,
        cfg.test_batch_size,
        channels,
    )
    theseus_layer.optimizer.params.max_iterations = cfg.test_max_iters

    ix = 0
    with torch.no_grad():
        losses = []
        for t, data in enumerate(test_loader):

            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)
            H_init = torch.eye(3).reshape(1, 9).repeat(img1.shape[0], 1).to(device)
            H_init = H_init[:, :-1]

            inputs = {
                "H_1_2": H_init,
                "img1": img1,
                "img2": img2,
            }

            if feat_model is not None:
                inputs["img1"] = feat_model(img1)
                inputs["img2"] = feat_model(img2)

            info = theseus_layer.forward(
                inputs, {"verbose": False, "damping": cfg.inner_optim.lm_damping}
            )
            H_1_2 = info[0]["H_1_2"]

            # Compute evalution metric, for now photometric loss
            img1_var = th.Variable(data=img1, name="img1")
            img2_var = th.Variable(data=img2, name="img2")

            loss = homography_error_fn([H_1_2], [img1_var, img2_var], pool_size=1)
            loss = loss.mean(dim=1)
            losses.append(loss)

            print(f"Test loss ({t} / {len(test_loader)}): {loss.mean().item()}")

            if cfg.save_gifs and ix < 20:
                write_gif_batch(
                    save_dir,
                    theseus_layer.optimizer.state_history,
                    img1,
                    img2,
                    loss,
                    ix,
                )
                ix += cfg.test_batch_size

        return torch.cat(losses).mean()


def run(cfg):

    dataset_root = os.path.join(get_original_cwd(), "data")
    chunks = [
        # "revisitop1m.1",
        # "revisitop1m.2",
        # "revisitop1m.3",
        # "revisitop1m.4",
        # "revisitop1m.5",
        # "revisitop1m.6",
        # "revisitop1m.7",
        # "revisitop1m.8",
        "revisitop1m.9",
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

    train_dataset = HomographyDataset(dataset_paths, cfg.imgH, cfg.imgW, train=True)
    test_dataset = HomographyDataset(dataset_paths, cfg.imgH, cfg.imgW, train=False)

    log_dir = "viz"
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"

    resnet = models.resnet18(pretrained=True)
    resnet.to(device)
    upsample = torch.nn.UpsamplingBilinear2d(size=[cfg.imgH, cfg.imgW])
    resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-4] + [upsample])
    fixed_feat = torch.nn.Sequential(*list(resnet.children())[:-4] + [upsample])

    # Training loop to refine pretrained features

    resnet_feat = train_loop(cfg, device, train_dataset, resnet_feat)

    # Test loop

    loss_pho = run_eval(cfg, device, test_dataset, save_dir=log_dir + "/pho")
    print(f"Photo-metric loss {loss_pho.item():.3f}")  # 0.031

    loss_fix = run_eval(
        cfg,
        device,
        test_dataset,
        save_dir=log_dir + "/feat_fix",
        feat_model=fixed_feat,
    )
    print(f"Feature-metric fixed loss {loss_fix.item():.3f}")  # 0.04

    loss_opt = run_eval(
        cfg,
        device,
        test_dataset,
        cfg.test_batch_size,
        save_dir=log_dir + "/feat_opt",
        feat_model=resnet_feat,
    )
    print(f"Feature-metric optimised loss {loss_opt.item():.3f}")


@hydra.main(config_path="./configs/", config_name="homography_estimation")
def main(cfg):
    torch.manual_seed(cfg.seed)
    run(cfg)


if __name__ == "__main__":
    main()
