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
import torchvision.models as models

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
    def __init__(self, img_dir, imgH, imgW, photo_aug=True):
        self.img_dir = img_dir
        self.imgH = imgH
        self.imgW = imgW
        self.img_paths = glob.glob(img_dir + "/**/*.jpg", recursive=True)
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
        self.photo_aug = photo_aug
        if self.photo_aug:
            self.rpa = RandomPhotoAug()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img1 = np.asarray(Image.open(img_path).resize(size=(self.imgW, self.imgH)))
        img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        img2, H_1_2 = self.rga.forward(
            img1, return_transform=True, normalize_returned_transform=True
        )

        # apply random photometric augmentations
        if self.photo_aug:
            img1 = torch.clamp(img1, 0.0, 1.0)
            img2 = torch.clamp(img2, 0.0, 1.0)
            img1 = self.rpa.forward(img1)
            img2 = self.rpa.forward(img2)

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

    img1_dst = warp_perspective_norm(H_1_2.data.reshape(-1, 3, 3), img1.data)
    loss = torch.nn.functional.mse_loss(img1_dst, img2.data, reduction="none")
    ones = warp_perspective_norm(
        H_1_2.data.reshape(-1, 3, 3), torch.ones_like(img1.data)
    )
    mask = ones > 0.9
    loss = loss.view(loss.shape[0], -1)
    mask = mask.view(loss.shape[0], -1)
    loss = (loss * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
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
    # cv2.imshow("i", out)
    # cv2.waitKey(0)


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
    channels = 3
    dataset = HomographyDataset(dataset_path, imgH, imgW)
    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data = next(iter(dataloader))

    log_dir = "viz"
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"

    if cfg.use_feats:
        resnet = models.resnet18(pretrained=True)
        resnet.to(device)
        resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-4])
        imgH, imgW = 20, 25
        channels = 128

        model_optimizer = torch.optim.Adam(
            resnet_feat.parameters(), lr=cfg.outer_optim.lr
        )

        l1_loss = torch.nn.L1Loss()

    objective = th.Objective()

    H_init = torch.eye(3).reshape(1, 9).repeat(batch_size, 1).to(device)
    H_1_2 = th.Vector(data=H_init, name="H_1_2")

    img_init = torch.zeros(batch_size, channels, imgH, imgW)
    img1 = th.Variable(data=img_init, name="img1")
    img2 = th.Variable(data=img_init, name="img2")

    homography_cf = th.AutoDiffCostFunction(
        optim_vars=[H_1_2],
        err_fn=homography_error_fn,
        dim=1,
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

    for epoch in range(cfg.outer_optim.num_epochs):
        epoch_losses = []
        for t, data in enumerate(dataloader):

            H_1_2_gt = data["H_1_2"].to(device)
            img1_init = data["img1"].to(device)
            img2_init = data["img2"].to(device)

            if cfg.use_feats:
                img1_init = resnet_feat(img1_init)
                img2_init = resnet_feat(img2_init)

            inputs = {
                "H_1_2": H_init,
                "img1": img1_init,
                "img2": img2_init,
            }

            info = theseus_layer.forward(
                inputs,
                optimizer_kwargs={
                    "verbose": False,
                    "backward_mode": BACKWARD_MODE[cfg.inner_optim.backward_mode],
                    "damping": 1e-2,
                },
            )

            if cfg.use_feats:
                # normalise last element to 1
                H_1_2.update(data=H_1_2 / H_1_2[:, -1][:, None])
                H_1_2_gt = H_1_2_gt.reshape(-1, 9)
                loss = l1_loss(H_1_2.data, H_1_2_gt)
                loss.backward()
                model_optimizer.step()
                epoch_losses.append(loss.item())

                print(f"Step {t}. Loss {loss.item():.4f}")

            if cfg.vis_training and cfg.use_feats is False:
                img1_dst = warp_perspective_norm(
                    H_1_2.data.reshape(-1, 3, 3), img1.data
                )
                for i in range(batch_size):
                    viz_warp(
                        log_dir,
                        img1.data[i],
                        img2.data[i],
                        img1_dst[i],
                        info[1].converged_iter[i].item(),
                        float(objective.error()[i].item()),
                    )

        print(f"******* Epoch {epoch}. Loss: {np.mean(epoch_losses):.4f} *******")


@hydra.main(config_path="./configs/", config_name="homography_estimation")
def main(cfg):
    torch.manual_seed(cfg.seed)
    run(cfg)


if __name__ == "__main__":
    main()
