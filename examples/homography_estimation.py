import kornia
from typing import List
import glob
import theseus as th
import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
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
            print("Using %d images for training" % len(self.img_paths))
        else:
            print("Using %d images for testing" % len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img1 = np.asarray(Image.open(img_path).resize(size=(self.imgW, self.imgH)))
        # Convert file to rgb if it is grayscale.
        if img1.shape != (self.imgH, self.imgW, 3):
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
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
    """
    Custom implementation for torch.nn.functional.grid_sample() to avoid this warning:
    > "RuntimeError: derivative for aten::grid_sampler_2d_backward is not implemented"
    """
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
    # grid_sample = torch.nn.functional.grid_sample
    # img2 = grid_sample(img, warped_grid, mode="bilinear", align_corners=True)
    # Using custom implementation, above will throw error with outer loop optim.
    img2 = grid_sample(img, warped_grid)
    return img2


def homography_error_fn(optim_vars: List[th.Manifold], aux_vars: List[th.Variable]):
    H8_1_2 = optim_vars[0].data.reshape(-1, 8)
    # Force the last element H[2,2] to be 1.
    H_1_2 = torch.cat([H8_1_2, H8_1_2.new_ones(H8_1_2.shape[0], 1)], dim=-1)  # type: ignore
    img1, img2 = aux_vars
    img1_dst = warp_perspective_norm(H_1_2.reshape(-1, 3, 3), img1.data)
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


def write_gif_batch(log_dir, img1, img2, H_hist, Hgt_1_2, err_hist=None):
    anim_dir = f"{log_dir}/animation"
    os.makedirs(anim_dir, exist_ok=True)
    subsample_anim = 1
    for it in H_hist:
        if it % subsample_anim != 0:
            continue
        # Visualize only first element in batch.
        H8_1_2 = H_hist[it]["H8_1_2"].data
        H_1_2 = torch.cat([H8_1_2, H8_1_2.new_ones(H8_1_2.shape[0], 1)], dim=-1)
        H_1_2_mat = H_1_2[0].reshape(1, 3, 3)
        Hgt_1_2_mat = Hgt_1_2[0].reshape(1, 3, 3)
        imgH, imgW = img1.shape[-2], img1.shape[-1]
        fc_err = four_corner_dist(H_1_2_mat, Hgt_1_2_mat, imgH, imgW)
        if err_hist is None:
            err = -1
        else:
            err = float(err_hist[0][it])
        img1 = img1[0][None, ...]
        img2 = img2[0][None, ...]
        img1_dsts = warp_perspective_norm(H_1_2_mat, img1)
        path = os.path.join(log_dir, f"{anim_dir}/{it:05d}.png")
        viz_warp(path, img1[0], img2[0], img1_dsts[0], it, err=err, fc_err=fc_err)
    anim_path = os.path.join(log_dir, "animation.gif")
    cmd = f"convert -delay 10 -loop 0 {anim_dir}/*.png {anim_path}"
    print("Generating gif here: %s" % anim_path)
    os.system(cmd)
    shutil.rmtree(anim_dir)
    return


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


def run():

    batch_size = 64
    max_iterations = 50
    step_size = 0.1
    verbose = True
    imgH, imgW = 60, 80
    use_gpu = True
    viz_every = 10
    disp_every = 10
    save_every = 100
    num_epochs = 999
    outer_lr = 1e-4
    use_cnn = True

    log_dir = os.path.join(os.getcwd(), "viz")
    os.makedirs(log_dir, exist_ok=True)
    print("Writing to tensorboard at", os.getcwd())
    writer = SummaryWriter(log_dir)

    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    dataset_paths = prepare_data()
    dataset = HomographyDataset(dataset_paths, imgH, imgW)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # A simple 2-layer CNN network that maintains the original image size.
    cnn_model = SimpleCNN()
    cnn_model.to(device)

    # Set up outer loop optimization.
    outer_optim = torch.optim.Adam(cnn_model.parameters(), lr=outer_lr)

    itr = 0

    for epoch in range(num_epochs):

        for _, data in enumerate(dataloader):

            outer_optim.zero_grad()

            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)
            Hgt_1_2 = data["H_1_2"].to(device)

            objective = th.Objective()

            H8_init = torch.eye(3).reshape(1, 9)[:, :-1].repeat(batch_size, 1)
            H8_1_2 = th.Vector(data=H8_init, name="H8_1_2")

            if use_cnn:  # Use cnn features.
                feat1 = cnn_model.forward(img1)
                feat2 = cnn_model.forward(img2)
            else:  # Use image pixels.
                feat1 = img1
                feat2 = img2

            feat1 = th.Variable(data=feat1, name="feat1")
            feat2 = th.Variable(data=feat2, name="feat2")

            # Set up inner loop optimization.
            homography_cf = th.AutoDiffCostFunction(
                optim_vars=[H8_1_2],
                err_fn=homography_error_fn,
                dim=1,
                aux_vars=[feat1, feat2],
            )
            objective.add(homography_cf)

            # Regularization helps avoid crash with using implicit mode.
            reg_w = 1e-2
            reg_w = th.ScaleCostWeight(np.sqrt(reg_w))
            reg_w.to(dtype=H8_init.dtype)
            vals = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
            H8_1_2_id = th.Vector(data=vals, name="identity")
            reg_cf = th.eb.VariableDifference(
                H8_1_2, reg_w, H8_1_2_id, name="reg_homography"
            )
            objective.add(reg_cf)

            inner_optim = th.LevenbergMarquardt(
                objective,
                max_iterations=max_iterations,
                step_size=step_size,
            )
            theseus_layer = th.TheseusLayer(inner_optim)
            theseus_layer.to(device)

            inputs = {
                "H8_1_2": H8_1_2.data,
                "feat1": feat1.data,
                "feat2": feat2.data,
            }
            if itr % disp_every == 0:
                verbose2 = verbose
            else:
                verbose2 = False
            info = theseus_layer.forward(
                inputs,
                optimizer_kwargs={
                    "verbose": verbose2,
                    "track_err_history": True,
                    # "backward_mode": BACKWARD_MODE["full"]})
                    "backward_mode": BACKWARD_MODE["implicit"],
                },
            )
            err_hist = info[1].err_history
            H_hist = theseus_layer.optimizer.state_history
            # print("Finished inner loop in %d iters" % len(H_hist))

            Hgt_1_2 = Hgt_1_2.reshape(-1, 9)
            H8_1_2 = theseus_layer.objective.get_optim_var("H8_1_2").data.reshape(-1, 8)
            H_1_2 = torch.cat([H8_1_2, H8_1_2.new_ones(H8_1_2.shape[0], 1)], dim=-1)
            # Loss is on four corner error.
            fc_dist = four_corner_dist(
                H_1_2.reshape(-1, 3, 3), Hgt_1_2.reshape(-1, 3, 3), imgH, imgW
            )
            outer_loss = fc_dist.mean()
            outer_loss.backward()
            outer_optim.step()
            print(
                "Epoch %d, iteration %d, outer_loss: %.3f"
                % (epoch, itr, outer_loss.item())
            )
            writer.add_scalar("Loss/train", outer_loss.item(), itr)

            if itr % viz_every == 0:
                write_gif_batch(log_dir, feat1, feat2, H_hist, Hgt_1_2, err_hist)

            if itr % save_every == 0:
                save_path = os.path.join(log_dir, "last.ckpt")
                torch.save({"itr": itr, "cnn_model": cnn_model}, save_path)

            itr += 1


def main():
    run()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
