# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import pathlib
import shutil
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import cv2
import hydra
import kornia
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import theseus as th
from theseus.core.cost_function import ErrFnType
from theseus.third_party.easyaug import GeoAugParam, RandomGeoAug, RandomPhotoAug
from theseus.third_party.utils import grid_sample

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=UserWarning)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SZ = 0.5
FONT_PT = (5, 15)


# Logger
logger = logging.getLogger(__name__)


# Download and extract data
def prepare_data():
    dataset_root = os.path.join(os.getcwd(), "data")
    chunks = [
        "revisitop1m.1",
        # "revisitop1m.2", # Uncomment for more data.
        # "revisitop1m.3",
        # "revisitop1m.4",
        # "revisitop1m.5",
    ]
    dataset_paths = []
    for chunk in chunks:
        dataset_path = os.path.join(dataset_root, chunk)
        dataset_paths.append(dataset_path)
        if not os.path.exists(dataset_path):
            logger.info("Downloading data")
            url_root = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/"
            tar = "%s.tar.gz" % chunk
            os.makedirs(dataset_path)
            cmd = "wget %s/%s -O %s/%s" % (url_root, tar, dataset_root, tar)
            logger.info("Running command: ", cmd)
            os.system(cmd)
            cmd = "tar -xf %s/%s -C %s" % (dataset_root, tar, dataset_path)
            logger.info("Running command: ", cmd)
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
        logger.info("Found %d total images in dataset" % len(self.img_paths))
        sc = 0.1
        self.rga = RandomGeoAug(
            rotate_param=GeoAugParam(min=-30 * sc, max=30 * sc),
            scale_param=GeoAugParam(min=(1.0 - 0.8 * sc), max=(1.0 + 1.2 * sc)),
            translate_x_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            translate_y_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            shear_x_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            shear_y_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            perspective_param=GeoAugParam(min=-0.1 * sc, max=0.1 * sc),
        )
        self.photo_aug = photo_aug
        if self.photo_aug:
            self.rpa = RandomPhotoAug()
            prob = 0.2  # Probability of augmentation applied.
            mag = 0.2  # Magnitude of augmentation [0: none, 1: max]
            self.rpa.set_all_probs(prob)
            self.rpa.set_all_mags(mag)

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
            logger.info("Using %d images for training" % len(self.img_paths))
        else:
            logger.info("Using %d images for testing" % len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img1 = np.asarray(Image.open(img_path).resize(size=(self.imgW, self.imgH)))
        # Convert file to rgb if it is grayscale.
        if img1.shape != (self.imgH, self.imgW, 3):
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1)[None]

        # apply random geometric augmentations to create homography problem
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


def warp_perspective_norm(H, img):
    height, width = img.shape[-2:]
    grid = kornia.utils.create_meshgrid(
        height, width, normalized_coordinates=True, device=H.device
    )
    Hinv = torch.inverse(H)
    warped_grid = kornia.geometry.transform.homography_warper.warp_grid(grid, Hinv)
    # Using custom implementation, above will throw error with outer loop optim.
    img2 = grid_sample(img, warped_grid)
    return img2


# loss is difference between warped and target image
def homography_error_fn(optim_vars: Tuple[th.Manifold], aux_vars: Tuple[th.Variable]):
    H8_1_2 = optim_vars[0].tensor.reshape(-1, 8)
    # Force the last element H[2,2] to be 1.
    H_1_2 = torch.cat([H8_1_2, H8_1_2.new_ones(H8_1_2.shape[0], 1)], dim=-1)  # type: ignore
    img1: th.Variable = aux_vars[0]
    img2: th.Variable = aux_vars[-1]
    img1_dst = warp_perspective_norm(H_1_2.reshape(-1, 3, 3), img1.tensor)
    loss = torch.nn.functional.mse_loss(img1_dst, img2.tensor, reduction="none")
    ones = warp_perspective_norm(
        H_1_2.data.reshape(-1, 3, 3), torch.ones_like(img1.tensor)
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


def viz_warp(path, img1, img2, img1_w, iteration, err=-1.0, fc_err=-1.0):
    img_diff = torch2cv2(torch.abs(img1_w - img2))
    img1 = torch2cv2(img1)
    img2 = torch2cv2(img2)
    img1_w = torch2cv2(img1_w)
    factor = 2
    new_sz = int(factor * img1.shape[1]), int(factor * img1.shape[0])
    img1 = cv2.resize(img1, new_sz, interpolation=cv2.INTER_NEAREST)
    img1 = cv2.resize(img1, new_sz)
    img2 = cv2.resize(img2, new_sz, interpolation=cv2.INTER_NEAREST)
    img1_w = cv2.resize(img1_w, new_sz, interpolation=cv2.INTER_NEAREST)
    img_diff = cv2.resize(img_diff, new_sz, interpolation=cv2.INTER_NEAREST)
    img1 = put_text(img1, "image I")
    img2 = put_text(img2, "image I'")
    img1_w = put_text(img1_w, "I warped to I'")
    img_diff = put_text(img_diff, "L2 diff")
    out = np.concatenate([img1, img2, img1_w, img_diff], axis=1)
    out = put_text(
        out,
        "iter: %05d, loss: %.8f, fc_err: %.3f px" % (iteration, err, fc_err),
        top=False,
    )
    cv2.imwrite(path, out)


# write gif showing source image being warped onto target through optimisation
def write_gif_batch(log_dir, img1, img2, H_hist, Hgt_1_2, err_hist):
    anim_dir = f"{log_dir}/animation"
    os.makedirs(anim_dir, exist_ok=True)
    subsample_anim = 1
    H8_1_2_hist = H_hist["H8_1_2"]
    num_iters = (~err_hist[0].isinf()).sum().item()
    for it in range(num_iters):
        if it % subsample_anim != 0:
            continue
        # Visualize only first element in batch.
        H8_1_2 = H8_1_2_hist[..., it]
        H_1_2 = torch.cat([H8_1_2, H8_1_2.new_ones(H8_1_2.shape[0], 1)], dim=-1).to(
            Hgt_1_2.device
        )
        H_1_2_mat = H_1_2[0].reshape(1, 3, 3)
        Hgt_1_2_mat = Hgt_1_2[0].reshape(1, 3, 3)
        imgH, imgW = img1.shape[-2], img1.shape[-1]
        fc_err = four_corner_dist(H_1_2_mat, Hgt_1_2_mat, imgH, imgW)
        err = float(err_hist[0][it])
        img1 = img1[0][None, ...]
        img2 = img2[0][None, ...]
        img1_dsts = warp_perspective_norm(H_1_2_mat, img1)
        path = os.path.join(anim_dir, f"{it:05d}.png")
        viz_warp(path, img1[0], img2[0], img1_dsts[0], it, err=err, fc_err=fc_err)
    anim_path = os.path.join(log_dir, "animation.gif")
    cmd = f"convert -delay 10 -loop 0 {anim_dir}/*.png {anim_path}"
    logger.info("Generating gif here: %s" % anim_path)
    os.system(cmd)
    shutil.rmtree(anim_dir)
    return


# L1 distance between 4 corners of source image warped using GT homography
# and estimated homography transform
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
    dist = torch.norm(warped_grid - warped_grid_gt, p=2, dim=-1)
    dist = dist.mean(dim=-1).mean(dim=-1)
    return dist


class SimpleCNN(nn.Module):
    def __init__(self, D=32):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, D, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(D, 3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(D)

    def forward(self, img):
        x = self.relu(self.bn1(self.conv1(img)))
        return self.conv2(x)


def run(
    batch_size: int = 64,
    num_epochs: int = 999,
    outer_lr: float = 1e-4,
    max_iterations: int = 50,
    step_size: float = 0.1,
    autograd_mode: str = "vmap",
    benchmarking_costs: bool = False,
    linear_solver_info: Optional[
        Tuple[Type[th.LinearSolver], Type[th.Linearization]]
    ] = None,
) -> List[List[Dict[str, Any]]]:
    logger.info(
        "==============================================================="
        "==========================="
    )
    logger.info(f"Batch Size: {batch_size}, " f"Autograd Mode: {autograd_mode}, ")

    logger.info(
        "---------------------------------------------------------------"
        "---------------------------"
    )
    verbose = False
    imgH, imgW = 60, 80
    use_gpu = True
    viz_every = 10
    save_every = 100
    use_cnn = True

    log_dir = os.path.join(os.getcwd(), "viz")
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    dataset_paths = prepare_data()
    dataset = HomographyDataset(dataset_paths, imgH, imgW)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # A simple 2-layer CNN network that maintains the original image size.
    cnn_model = SimpleCNN()
    cnn_model.to(device)

    objective = th.Objective()

    data = next(iter(dataloader))
    H8_init = torch.eye(3).reshape(1, 9)[:, :-1].repeat(batch_size, 1)
    feats = torch.zeros_like(data["img1"])
    H8_1_2 = th.Vector(tensor=H8_init, name="H8_1_2")
    feat1 = th.Variable(tensor=feats, name="feat1")
    feat2 = th.Variable(tensor=feats, name="feat2")

    # Set up inner loop optimization.
    homography_cf = th.AutoDiffCostFunction(
        optim_vars=[H8_1_2],
        err_fn=cast(ErrFnType, homography_error_fn),
        dim=1,
        aux_vars=[feat1, feat2],
        autograd_mode=autograd_mode,
    )
    objective.add(homography_cf)

    # Regularization helps avoid crash with using implicit mode.
    reg_w_value = 1e-2
    reg_w = th.ScaleCostWeight(np.sqrt(reg_w_value))
    reg_w.to(dtype=H8_init.dtype)
    vals = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    H8_1_2_id = th.Vector(tensor=vals, name="identity")
    reg_cf = th.Difference(
        H8_1_2, target=H8_1_2_id, cost_weight=reg_w, name="reg_homography"
    )
    objective.add(reg_cf)

    if linear_solver_info is not None:
        linear_solver_cls, linearization_cls = linear_solver_info
    else:
        linear_solver_cls, linearization_cls = None, None
    inner_optim = th.LevenbergMarquardt(
        objective,
        linear_solver_cls=linear_solver_cls,
        linearization_cls=linearization_cls,
        max_iterations=max_iterations,
        step_size=step_size,
    )
    theseus_layer = th.TheseusLayer(inner_optim)
    theseus_layer.to(device)

    # Set up outer loop optimization.
    outer_optim = torch.optim.Adam(cnn_model.parameters(), lr=outer_lr)

    itr = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    logger.info(
        "---------------------------------------------------------------"
        "---------------------------"
    )
    # benchmark_results[i][j] has the results (time/mem) for epoch i and batch j
    benchmark_results: List[List[Dict[str, Any]]] = []
    for epoch in range(num_epochs):
        benchmark_results.append([])
        forward_times: List[float] = []
        forward_mems: List[float] = []
        backward_times: List[float] = []
        backward_mems: List[float] = []

        for _, data in enumerate(dataloader):
            benchmark_results[-1].append({})
            outer_optim.zero_grad()

            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)
            Hgt_1_2 = data["H_1_2"].to(device)

            if use_cnn:  # Use cnn features.
                feat1_tensor = cnn_model.forward(img1)
                feat2_tensor = cnn_model.forward(img2)
            else:  # Use image pixels.
                feat1_tensor = img1
                feat2_tensor = img2

            H8_init = torch.eye(3).reshape(1, 9)[:, :-1].repeat(batch_size, 1)
            H8_init = H8_init.to(device)

            inputs: Dict[str, torch.Tensor] = {
                "H8_1_2": H8_init,
                "feat1": feat1_tensor,
                "feat2": feat2_tensor,
            }
            start_event.record()
            torch.cuda.reset_peak_memory_stats()

            if benchmarking_costs:
                objective.update(inputs)
                inner_optim.linear_solver.linearization.linearize()
            else:
                _, info = theseus_layer.forward(
                    inputs,
                    optimizer_kwargs={
                        "verbose": verbose,
                        "track_err_history": True,
                        "track_state_history": True,
                        "backward_mode": "implicit",
                    },
                )
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event)
            forward_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            forward_times.append(forward_time)
            forward_mems.append(forward_mem)
            benchmark_results[-1][-1]["ftime"] = forward_time
            benchmark_results[-1][-1]["fmem"] = forward_mem

            if benchmarking_costs:
                continue

            optimizer_info: th.NonlinearOptimizerInfo = cast(
                th.NonlinearOptimizerInfo, info
            )
            err_hist = optimizer_info.err_history
            H_hist = optimizer_info.state_history
            # print("Finished inner loop in %d iters" % len(H_hist))

            Hgt_1_2 = Hgt_1_2.reshape(-1, 9)
            H8_1_2_tensor = theseus_layer.objective.get_optim_var(
                "H8_1_2"
            ).tensor.reshape(-1, 8)
            H_1_2 = torch.cat(
                [H8_1_2_tensor, H8_1_2_tensor.new_ones(H8_1_2_tensor.shape[0], 1)],
                dim=-1,
            )
            # Loss is on four corner error.
            fc_dist = four_corner_dist(
                H_1_2.reshape(-1, 3, 3), Hgt_1_2.reshape(-1, 3, 3), imgH, imgW
            )
            outer_loss = fc_dist.mean()

            start_event.record()
            torch.cuda.reset_peak_memory_stats()
            outer_loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            backward_time = start_event.elapsed_time(end_event)
            backward_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

            backward_times.append(backward_time)
            backward_mems.append(backward_mem)
            benchmark_results[-1][-1]["btime"] = backward_time
            benchmark_results[-1][-1]["bmem"] = backward_mem

            outer_optim.step()
            logger.info(
                "Epoch %d, iteration %d, outer_loss: %.3f"
                % (epoch, itr, outer_loss.item())
            )

            if itr % viz_every == 0:
                write_gif_batch(log_dir, feat1, feat2, H_hist, Hgt_1_2, err_hist)

            if itr % save_every == 0:
                save_path = os.path.join(log_dir, "last.ckpt")
                torch.save({"itr": itr, "cnn_model": cnn_model}, save_path)

            itr += 1

        logger.info(
            "---------------------------------------------------------------"
            "---------------------------"
        )
        logger.info(f"Forward pass took {sum(forward_times)} ms/epoch.")
        logger.info(f"Forward pass took {sum(forward_mems)/len(forward_mems)} MBs.")
        if not benchmarking_costs:
            logger.info(f"Backward pass took {sum(backward_times)} ms/epoch.")
            logger.info(
                f"Backward pass took {sum(backward_mems)/len(backward_mems)} MBs."
            )
        logger.info(
            "---------------------------------------------------------------"
            "---------------------------"
        )
    return benchmark_results


@hydra.main(config_path="./configs/", config_name="homography_estimation")
def main(cfg):
    benchmark_results = run(
        batch_size=cfg.outer_optim.batch_size,
        outer_lr=cfg.outer_optim.lr,
        num_epochs=cfg.outer_optim.num_epochs,
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
        autograd_mode=cfg.autograd_mode,
        benchmarking_costs=cfg.benchmarking_costs,
        linear_solver_info=cfg.get("linear_solver_info", None),
    )
    torch.save(benchmark_results, pathlib.Path(os.getcwd()) / "benchmark_results.pt")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
