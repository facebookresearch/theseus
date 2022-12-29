from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

import theseus as th

# ----------------------------------------------------------------------------------- #
# ------------------------------------ Data loading --------------------------------- #
# ----------------------------------------------------------------------------------- #


class TactilePushingDataset:
    def __init__(
        self,
        data_fname: str,
        sdf_fname: str,
        episode_length: int,
        batch_size: int,
        max_episodes: int,
        max_steps: int,
        device: th.DeviceType,
        split_episodes: bool = False,
        data_mode: str = "all",
        val_ratio: float = 0.1,
        seed: int = 1234567,
    ):
        assert data_mode in ["all", "train", "val"]

        batch_size = min(batch_size, max_episodes)
        data = TactilePushingDataset._load_dataset_from_file(
            data_fname, episode_length, max_episodes, device, split_episodes
        )
        (
            self.sdf_data_tensor,
            self.sdf_cell_size,
            self.sdf_origin,
        ) = TactilePushingDataset._load_tactile_sdf_from_file(sdf_fname, device)

        # keys: img_feats, eff_poses, obj_poses, contact_episode, contact_flag
        num_episodes = data["obj_poses"].shape[0]
        if data_mode == "all":
            idx = np.arange(num_episodes)
        else:
            rng = np.random.default_rng(seed)
            order = rng.permutation(num_episodes)
            stop = max(int(np.ceil(num_episodes * val_ratio)), 2)
            idx = order[:stop] if data_mode == "val" else order[stop:]

        self.img_feats = data["img_feats"][idx]  # type: ignore
        self.eff_poses = data["eff_poses"][idx]  # type: ignore
        self.obj_poses = data["obj_poses"][idx]  # type: ignore
        self.contact_episode = data["contact_episode"][idx]  # type: ignore
        self.contact_flag = data["contact_flag"][idx]  # type: ignore
        # Check sizes of the attributes assigned above
        self.dataset_size: int = -1
        for key in data:
            if self.dataset_size == -1:
                self.dataset_size = getattr(self, key).shape[0]
            else:
                assert self.dataset_size == getattr(self, key).shape[0]
        print(f"Dataset for mode '{data_mode}' has size {self.dataset_size}.")

        # obj_poses is shape (num_episodes, episode_length, 3)
        self.time_steps = np.minimum(max_steps, self.obj_poses.shape[1])
        self.batch_size = batch_size
        self.num_batches = (self.dataset_size - 1) // self.batch_size + 1

    @staticmethod
    def _load_dataset_from_file(
        filename: str,
        episode_length: int,
        max_episodes: int,
        device: th.DeviceType,
        split_episodes: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Load all episode data
        with open(filename) as f:
            import json

            data_from_file = json.load(f)

        dataset_all = {}
        dataset_all["obj_poses"] = torch.tensor(
            data_from_file["obj_poses_2d"], device=device
        )
        dataset_all["eff_poses"] = torch.tensor(
            data_from_file["ee_poses_2d"], device=device
        )
        dataset_all["img_feats"] = torch.tensor(
            data_from_file["img_feats"], device=device
        )
        dataset_all["contact_episode"] = torch.tensor(
            data_from_file["contact_episode"], device=device
        )
        dataset_all["contact_flag"] = torch.tensor(
            data_from_file["contact_flag"], device=device
        )

        # Read all episodes and filter those with length less than desired
        episode_indices = [
            idx.item() for idx in dataset_all["contact_episode"].unique()
        ]
        data: Dict[str, List[torch.Tensor]] = dict(
            [(k, []) for k in dataset_all.keys()]
        )

        for episode in episode_indices:
            if len(data["obj_poses"]) >= max_episodes:
                break
            ds_idxs = torch.nonzero(dataset_all["contact_episode"] == episode).squeeze()
            if len(ds_idxs) < episode_length:
                continue
            for key, val in dataset_all.items():
                if split_episodes:
                    tensors = TactilePushingDataset._get_tensor_splits(
                        val[ds_idxs], episode_length
                    )
                else:
                    ds_idxs = ds_idxs[:episode_length]
                    tensors = [val[ds_idxs]]
                for tensor in tensors:
                    data[key].append(tensor)

        # Stack all episode data into single tensors
        data_tensors = {}
        for key in data:
            data_tensors[key] = torch.stack(data[key])
        num_ep_read, len_ep_read = data_tensors[key].shape[:2]
        print(f"Read {num_ep_read} episodes of length {len_ep_read}.")
        return data_tensors

    @staticmethod
    def _get_tensor_splits(
        tensor: torch.Tensor, episode_length: int
    ) -> List[torch.Tensor]:
        squeeze = False
        if tensor.ndim == 1:
            squeeze = True
            tensor = tensor.view(-1, 1)
        length, dof = tensor.shape
        num_splits = length // episode_length
        mod_tensor = tensor[: num_splits * episode_length]
        reshaped_tensor = mod_tensor.view(num_splits, -1, dof)
        if squeeze:
            reshaped_tensor = reshaped_tensor.squeeze(2)
        return [t for t in reshaped_tensor]

    @staticmethod
    def _load_tactile_sdf_from_file(
        filename: str, device: th.DeviceType
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with open(filename) as f:
            import json

            sdf_from_file = json.load(f)
        sdf_data_vec = sdf_from_file["grid_data"]

        sdf_data_mat = np.zeros(
            (sdf_from_file["grid_size_y"], sdf_from_file["grid_size_x"])
        )
        for i in range(sdf_data_mat.shape[0]):
            for j in range(sdf_data_mat.shape[1]):
                sdf_data_mat[i, j] = sdf_data_vec[i][j]

        sdf_data_tensor = torch.tensor(sdf_data_mat, device=device).unsqueeze(0)
        cell_size = torch.tensor([sdf_from_file["grid_res"]], device=device).unsqueeze(
            0
        )
        origin = torch.tensor(
            [sdf_from_file["grid_origin_x"], sdf_from_file["grid_origin_y"]],
            device=device,
        ).unsqueeze(0)

        return sdf_data_tensor, cell_size, origin

    def get_batch(self, batch_idx: int) -> Dict[str, torch.Tensor]:
        assert batch_idx < self.num_batches
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.dataset_size)
        batch = {}
        batch["img_feats"] = self.img_feats[start:end, : self.time_steps]
        batch["eff_poses"] = self.eff_poses[start:end, : self.time_steps]
        batch["obj_poses"] = self.obj_poses[start:end, : self.time_steps]
        batch["obj_poses_gt"] = self.obj_poses[start:end, : self.time_steps, :].clone()
        batch["eff_poses_gt"] = self.eff_poses[start:end, : self.time_steps, :].clone()
        batch["obj_start_pose"] = self.obj_poses[start:end, 0]
        for i in range(self.time_steps):
            batch[f"motion_capture_{i}"] = self.eff_poses[start:end, i]
        return batch


# ----------------------------------------------------------------------------------- #
# ---------------------------------- Visualization ---------------------------------- #
# ----------------------------------------------------------------------------------- #
def _draw_tactile_effector(poses: np.ndarray, label: str = "groundtruth"):
    linestyle = "-"
    color = "tab:orange"
    if label == "groundtruth":
        linestyle = "--"
        color = "tab:gray"

    # plot contact point and normal
    plt.plot(poses[-1][0], poses[-1][1], "k*", linestyle=linestyle)
    ori = poses[-1][2]
    (dx, dy) = (0.03 * -np.sin(ori), 0.03 * np.cos(ori))
    plt.arrow(
        poses[-1][0],
        poses[-1][1],
        dx,
        dy,
        linewidth=2,
        head_width=0.001,
        color=color,
        head_length=0.01,
        fc=color,
        ec=color,
    )

    eff_radius = 0.0075
    circle = mpatches.Circle(
        (poses[-1][0], poses[-1][1]), color=color, radius=eff_radius
    )
    plt.gca().add_patch(circle)


def _draw_tactile_object(
    poses: np.ndarray, rect_len_x: float, rect_len_y: float, label: str = "optimizer"
):
    linestyle = "-"
    color = "tab:orange"
    if label == "groundtruth":
        linestyle = "--"
        color = "tab:gray"

    plt.plot(
        poses[:, 0],
        poses[:, 1],
        color=color,
        linestyle=linestyle,
        label=label,
        linewidth=2,
        alpha=0.9,
    )

    # shape: rect
    yaw = poses[-1][2]
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    offset = np.matmul(R, np.array([[0.5 * rect_len_x], [0.5 * rect_len_y]]))
    xb, yb = poses[-1][0] - offset[0], poses[-1][1] - offset[1]
    rect = mpatches.Rectangle(
        (xb, yb),
        rect_len_x,
        rect_len_y,
        angle=(np.rad2deg(yaw)),
        facecolor="None",
        edgecolor=color,
        linestyle=linestyle,
        linewidth=2,
    )
    plt.gca().add_patch(rect)


def visualize_tactile_push2d(
    obj_poses: torch.Tensor,
    eff_poses: torch.Tensor,
    obj_poses_gt: torch.Tensor,
    eff_poses_gt: torch.Tensor,
    rect_len_x: float,
    rect_len_y: float,
    save_fname: Optional[str] = None,
):
    plt.cla()
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.gca().axis("equal")
    plt.axis("off")

    _draw_tactile_object(
        obj_poses.cpu().detach().numpy(), rect_len_x, rect_len_y, label="optimizer"
    )
    _draw_tactile_effector(eff_poses.cpu().detach().numpy(), label="optimizer")
    _draw_tactile_object(
        obj_poses_gt.cpu().detach().numpy(), rect_len_x, rect_len_y, label="groundtruth"
    )
    _draw_tactile_effector(eff_poses_gt.cpu().detach().numpy(), label="groundtruth")

    plt.show()
    plt.pause(1e-9)

    if save_fname is not None:
        plt.savefig(save_fname)
