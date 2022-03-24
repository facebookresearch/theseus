from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

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
        device: torch.device,
    ):
        batch_size = min(batch_size, max_episodes)
        data = TactilePushingDataset._load_dataset_from_file(
            data_fname, episode_length, max_episodes, device
        )
        (
            self.sdf_data_tensor,
            self.sdf_cell_size,
            self.sdf_origin,
        ) = TactilePushingDataset._load_tactile_sdf_from_file(sdf_fname, device)

        self.img_feats = data["img_feats"]
        self.eff_poses = data["eff_poses"]
        self.obj_poses = data["obj_poses"]
        self.contact_episode = data["contact_episode"]
        self.contact_flag = data["contact_flag"]
        self.dataset_size: int = -1
        for key, val in data.items():
            setattr(self, key, val)
            if self.dataset_size == -1:
                self.dataset_size = val.shape[0]
            else:
                assert self.dataset_size == val.shape[0]
        self.batch_size = batch_size

    @staticmethod
    def _load_dataset_from_file(
        filename: str, episode_length: int, max_episodes: int, device: torch.device
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

        for i, episode in enumerate(episode_indices):
            if i == max_episodes:
                break
            ds_idxs = torch.nonzero(dataset_all["contact_episode"] == episode).squeeze()
            if len(ds_idxs) < episode_length:
                continue
            ds_idxs = ds_idxs[:episode_length]
            for key, val in dataset_all.items():
                data[key].append(val[ds_idxs])

        # Stack all episode data into single tensors
        data_tensors = {}
        for key in data:
            data_tensors[key] = torch.stack(data[key])
        print(f"Read {len(data_tensors[key])} episodes of length {episode_length}.")
        return data_tensors

    @staticmethod
    def _load_tactile_sdf_from_file(
        filename: str, device: torch.device
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

    def get_measurements(
        self, time_steps: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        num_batches = (self.dataset_size - 1) // self.batch_size + 1
        batches = []
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, self.dataset_size)
            batches.append(
                (
                    self.img_feats[start:end, 0:time_steps],
                    self.eff_poses[start:end, 0:time_steps],
                    self.obj_poses[start:end, 0:time_steps],
                )
            )
        return batches

    def get_start_pose_and_motion_for_batch(
        self, batch_idx: int, time_steps: int
    ) -> Dict[str, torch.Tensor]:
        pose_and_motion_batch = {}
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.dataset_size)
        pose_and_motion_batch["obj_start_pose"] = self.obj_poses[start:end, 0]
        for i in range(time_steps):
            pose_and_motion_batch[f"motion_capture_{i}"] = self.eff_poses[start:end, i]
        return pose_and_motion_batch

    def get_gt_data_for_batch(
        self, batch_idx: int, time_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.dataset_size)
        obj_poses_gt = self.obj_poses[start:end, :time_steps, :].clone()
        eff_poses_gt = self.eff_poses[start:end, :time_steps, :].clone()
        return obj_poses_gt, eff_poses_gt


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
