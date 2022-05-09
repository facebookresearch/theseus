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
    def __init__(self, img_dir, imgH, imgW, photo_aug=True, train=True):
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

        # train test split
        self.img_paths.sort()
        if train:
            self.img_paths = self.img_paths[:-100]
        else:
            self.img_paths = self.img_paths[-100:]

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


def make_viz(img1, img2, img1_w, iteration, err):
    img_diff = torch2cv2(torch.abs(img1_w - img2))
    img1 = torch2cv2(img1)
    img2 = torch2cv2(img2)
    img1_w = torch2cv2(img1_w)
    img1 = put_text(img1, "image I")
    img2 = put_text(img2, "image I'")
    img1_w = put_text(img1_w, "I warped to I'")
    img1_w = put_text(img1_w, "iter: %05d, loss: %.3f" % (iteration, err), top=False)
    img_diff = put_text(img_diff, "L2 diff")
    out = np.concatenate([img1, img2, img1_w, img_diff], axis=1)
    return out


def viz_warp(log_dir, img1, img2, img1_w, iteration, err):
    out = make_viz(img1, img2, img1_w, iteration, err)
    path = os.path.join(log_dir, "out_%05d.png" % iteration)
    cv2.imwrite(path, out)


def setup_homgraphy_layer(cfg, device, batch_size, imgH, imgW, channels):

    objective = th.Objective()

    H_init = torch.zeros(batch_size, 8)
    H_1_2 = th.Vector(data=H_init, name="H_1_2")

    img_init = torch.zeros(batch_size, channels, imgH, imgW)
    img1 = th.Variable(data=img_init, name="img1")
    img2 = th.Variable(data=img_init, name="img2")

    # loss is pooled so error dim is not too large
    pool_size = int(imgH // 4)
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
    batch_size = 5
    train_dataset = HomographyDataset(dataset_path, imgH, imgW, train=True)
    test_dataset = HomographyDataset(dataset_path, imgH, imgW, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    log_dir = "viz"
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu"

    resnet = models.resnet18(pretrained=True)
    resnet.to(device)
    resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-4])
    fixed_feat = torch.nn.Sequential(*list(resnet.children())[:-4])
    featH, featW = 20, 25
    feat_channels = 128

    model_optimizer = torch.optim.Adam(resnet_feat.parameters(), lr=cfg.outer_optim.lr)

    l1_loss = torch.nn.L1Loss()

    layer_feat = setup_homgraphy_layer(
        cfg,
        device,
        batch_size,
        featH,
        featW,
        feat_channels,
    )

    # Training loop to refine pretrained features

    for epoch in range(cfg.outer_optim.num_epochs):
        epoch_losses = []
        start_time = time.time()
        for t, data in enumerate(train_loader):

            H_1_2_gt = data["H_1_2"].to(device)
            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)
            feat1 = resnet_feat(img1)
            feat2 = resnet_feat(img2)

            H_init = torch.eye(3).reshape(1, 9).repeat(batch_size, 1).to(device)
            H_init = H_init[:, :-1]
            inputs = {
                "H_1_2": H_init,
                "img1": feat1,
                "img2": feat2,
            }

            info = layer_feat.forward(
                inputs,
                optimizer_kwargs={
                    "verbose": False,
                    "backward_mode": BACKWARD_MODE[cfg.inner_optim.backward_mode],
                    # "damping": 1e-2,
                },
            )

            # no loss on last element as set to one
            H_1_2_gt = H_1_2_gt.reshape(-1, 9)[:, :-1]
            H_1_2 = info[0]["H_1_2"]
            loss = l1_loss(H_1_2.data, H_1_2_gt)
            loss.backward()
            model_optimizer.step()
            epoch_losses.append(loss.item())

            if t % 10 == 0:
                print(f"Step {t}. Loss {loss.item():.4f}")

        epoch_time = time.time() - start_time
        print(
            f"******* Epoch {epoch}. Average loss: {np.mean(epoch_losses):.4f}. "
            f"Epoch time {epoch_time}*******"
        )

    # Test loop

    print("\n**** Test dataset evaluation ****")

    layer_pho = setup_homgraphy_layer(
        cfg,
        device,
        batch_size,
        imgH,
        imgW,
        3,
    )

    with torch.no_grad():
        for t, data in enumerate(test_loader):

            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)
            feat1 = resnet_feat(img1)
            feat2 = resnet_feat(img2)
            fixed_feat1 = fixed_feat(img1)
            fixed_feat2 = fixed_feat(img2)

            H_init = torch.eye(3).reshape(1, 9).repeat(batch_size, 1).to(device)
            H_init = H_init[:, :-1]

            inputs_img = {
                "H_1_2": H_init,
                "img1": img1,
                "img2": img2,
            }

            inputs_feat = {
                "H_1_2": H_init,
                "img1": feat1,
                "img2": feat2,
            }

            inputs_fixed_feat = {
                "H_1_2": H_init,
                "img1": fixed_feat1,
                "img2": fixed_feat2,
            }

            layer_feat.optimizer.params.max_iterations = cfg.test_max_iters
            layer_pho.optimizer.params.max_iterations = cfg.test_max_iters

            info_opt = layer_feat.forward(inputs_feat)
            H_opt = info_opt[0]["H_1_2"]
            info_fix = layer_feat.forward(inputs_fixed_feat)
            H_fix = info_fix[0]["H_1_2"]
            info_pho = layer_pho.forward(inputs_img)
            H_pho = info_pho[0]["H_1_2"]

            # Compute evalution metric, for now photometric loss
            img1_var = th.Variable(data=img1, name="img1")
            img2_var = th.Variable(data=img2, name="img2")

            loss_opt = homography_error_fn([H_opt], [img1_var, img2_var], pool_size=1)
            loss_fix = homography_error_fn([H_fix], [img1_var, img2_var], pool_size=1)
            loss_pho = homography_error_fn([H_pho], [img1_var, img2_var], pool_size=1)
            loss_opt = loss_opt.mean(dim=1)
            loss_fix = loss_fix.mean(dim=1)
            loss_pho = loss_pho.mean(dim=1)

            print(
                f"Photometric errors: optimised feats {loss_opt.mean():.3f}, "
                f"fixed feats {loss_fix.mean():.3f}, photometric {loss_pho.mean():.3f}"
            )

            if cfg.vis_res:
                Hs = [H_opt, H_fix, H_pho]
                losses = [loss_opt, loss_fix, loss_pho]
                infos = [info_opt, info_fix, info_pho]
                labels = [
                    "Optimised featuremetric",
                    "Pretrained featuremetric",
                    "Photometric",
                ]
                img1_dsts = []
                for H in Hs:
                    ones = torch.ones(H.shape[0], 1, device=H.device, dtype=H.dtype)
                    H_1_2_mat = torch.cat((H.data, ones), dim=1).reshape(-1, 3, 3)
                    img1_dsts.append(warp_perspective_norm(H_1_2_mat, img1))

                for i in range(batch_size):
                    imgs = []
                    for j in range(3):
                        viz = make_viz(
                            img1[i],
                            img2[i],
                            img1_dsts[j][i],
                            infos[j][1].converged_iter[i],
                            float(losses[j][i].item()),
                        )
                        viz = put_text(viz, labels[j], top=False)
                        imgs.append(viz)
                    imgs = np.vstack((imgs))
                    cv2.imshow("viz", imgs)
                    cv2.waitKey(0)


@hydra.main(config_path="./configs/", config_name="homography_estimation")
def main(cfg):
    torch.manual_seed(cfg.seed)
    run(cfg)


if __name__ == "__main__":
    main()
