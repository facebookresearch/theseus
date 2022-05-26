import os
import pathlib
import random
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import yaml

import theseus as th

# ----------------------------------------------------------------------------------- #
# ------------------------------------ Data loading --------------------------------- #
# ----------------------------------------------------------------------------------- #
FileInfo = Tuple[pathlib.Path, pathlib.Path, pathlib.Path, str]


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool,
        num_images: int,
        dataset_dir: str,
        map_type: str,
        val_ratio: float = 0,
        filter_collision_maps: bool = True,
    ):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.map_type = map_type

        with open(self.dataset_dir / "meta.yaml") as f:
            self.cfg = yaml.safe_load(f)

        self.collision_maps = set()
        collision_fname = self.dataset_dir / "collision_maps.txt"
        if collision_fname.is_file() and filter_collision_maps:
            with open(collision_fname, "r") as f:
                self.collision_maps.update(f.read().splitlines())

        files_per_type = self.get_all_files()
        all_train_files: List[FileInfo] = []
        all_val_files: List[FileInfo] = []
        num_train = int((1 - val_ratio) * self.cfg["num_envs"])
        for type_ in files_per_type:
            if map_type == "mixed" or map_type == type_:
                all_train_files.extend(files_per_type[type_][:num_train])
                all_val_files.extend(files_per_type[type_][num_train:])

        random.shuffle(all_train_files)
        random.shuffle(all_val_files)
        self.files = (
            all_train_files[:num_images] if train else all_val_files[:num_images]
        )

    def get_all_files(self) -> Dict[str, List[FileInfo]]:
        files: Dict[str, List[FileInfo]] = dict((k, []) for k in self.cfg["map_types"])
        for map_type in self.cfg["map_types"]:
            for idx in range(self.cfg["num_envs"]):
                if f"{map_type}_{idx}" in self.collision_maps:
                    continue
                img_fname = self.dataset_dir / "im_sdf" / map_type / f"{idx}_im.png"
                sdf_fname = self.dataset_dir / "im_sdf" / map_type / f"{idx}_sdf.npy"
                traj_fname = (
                    self.dataset_dir
                    / "opt_trajs_gpmp2"
                    / map_type
                    / f"env_{idx}_prob_0.npz"
                )
                for f in [img_fname, sdf_fname, traj_fname]:
                    assert os.path.isfile(f)
                files[map_type].append(
                    (img_fname, sdf_fname, traj_fname, f"{map_type}_{idx}")
                )
        return files

    def __getitem__(self, idx: int):
        img_file, sdf_file, traj_file, file_id = self.files[idx]

        env_params = self.cfg["env_params"]

        # SDF Data
        cells_per_unit = self.cfg["im_size"] / (
            env_params["x_lims"][1] - env_params["x_lims"][0]
        )
        cell_size = torch.tensor(cells_per_unit).reciprocal().view(1)
        origin = torch.tensor([env_params["x_lims"][0], env_params["y_lims"][0]])
        sdf_data = torch.from_numpy(np.load(sdf_file))

        # Image
        tmp_map = plt.imread(img_file)
        if tmp_map.ndim == 3:
            the_map = tmp_map[..., 0]

        # Trajectory
        traj_data = np.load(traj_file)
        trajectory = torch.from_numpy(traj_data["th_opt"]).permute(1, 0)
        # next two lines re-orient the dgpmp2 trajectory to theseus coordinate system
        trajectory[1] *= -1.0
        trajectory[3] *= -1.0
        return {
            "map_tensor": the_map,
            "sdf_origin": origin.double(),
            "cell_size": cell_size.double(),
            "sdf_data": sdf_data.double(),
            "expert_trajectory": trajectory.double(),
            "file_id": file_id,
        }

    def __len__(self):
        return len(self.files)


# ----------------------------------------------------------------------------------- #
# ------------------------------- Plotting utilities -------------------------------- #
# ----------------------------------------------------------------------------------- #
def _create_line_from_trajectory(
    x_list, y_list, linestyle="-", linewidth=2, alpha=1.0, color="red"
):
    line = plt.Line2D(x_list, y_list)
    line.set_linestyle(linestyle)
    line.set_linewidth(linewidth)
    line.set_color(color)
    line.set_alpha(alpha)
    return line


def _add_robot_to_trajectory(x_list, y_list, radius, color="magenta", alpha=0.05):
    patches = []
    for i in range(x_list.shape[0]):
        patch = mpl.patches.Circle((x_list[i], y_list[i]), radius)
        patches.append(patch)
    patch_collection = mpl.collections.PatchCollection(
        patches, alpha=alpha, color=color
    )
    return patch_collection


def generate_trajectory_figs(
    map_tensor: torch.Tensor,
    sdf: th.eb.SignedDistanceField2D,
    trajectories: List[torch.Tensor],
    robot_radius: float,
    max_num_figures: int = 20,
    labels: Optional[List[str]] = None,
    fig_idx_robot: int = 1,
    figsize: Tuple[int, int] = (8, 8),
) -> List[plt.Figure]:
    # cell rows/cols for each batch of trajectories
    traj_rows = []
    traj_cols = []
    for trajectory in trajectories:
        row, col, _ = sdf.convert_points_to_cell(trajectory[:, :2, :])
        traj_rows.append(np.clip(row, 0, map_tensor.shape[1] - 1))
        traj_cols.append(np.clip(col, 0, map_tensor.shape[1] - 1))

    # Generate a separate figure for each batch index
    colors = ["green", "blue", "red"]
    if not labels:
        labels = ["initial_solution", "best_solution", "expert"]
    figures: List[plt.Figure] = []
    for map_idx in range(map_tensor.shape[0]):
        if map_idx >= max_num_figures:
            continue
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        map_data = map_tensor[map_idx].clone().cpu().numpy()
        map_data = np.tile(map_data, (3, 1, 1)).transpose((1, 2, 0))
        axs.imshow(map_data)
        cell_size = sdf.cell_size.data
        patches = []
        for t_idx, trajectory in enumerate(trajectories):
            row = traj_rows[t_idx][map_idx]
            col = traj_cols[t_idx][map_idx]
            line = _create_line_from_trajectory(col, row, color=colors[t_idx])
            axs.add_line(line)
            if t_idx == fig_idx_robot:  # solution trajectory
                radius = robot_radius / cell_size[map_idx][0]
                patch_coll = _add_robot_to_trajectory(col, row, radius)
                axs.add_collection(patch_coll)
            patches.append(mpl.patches.Patch(color=colors[t_idx], label=labels[t_idx]))
        patches.append(mpl.patches.Patch(color="magenta"))
        axs.legend(handles=patches[:-1], fontsize=16)
        axs.set_yticks([])
        axs.set_xticks([])
        fig.tight_layout()
        figures.append(fig)
    return figures
