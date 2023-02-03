# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, cast

import numpy as np
import torch

import theseus as th


class PoseGraphEdge:
    def __init__(
        self,
        i: int,
        j: int,
        relative_pose: Union[th.SE2, th.SE3],
        weight: Optional[th.DiagonalCostWeight] = None,
    ):
        self.i = i
        self.j = j
        self.relative_pose = relative_pose
        self.weight = weight

    def to(self, *args, **kwargs):
        self.weight.to(*args, **kwargs)
        self.relative_pose.to(*args, **kwargs)


# This function reads a file in g2o formate and returns the number of of poses, initial
# values and edges.
# g2O format: https://github.com/RainerKuemmerle/g2o/wiki/File-format-slam-3d
def read_3D_g2o_file(
    path: str, dtype: Optional[torch.dtype] = None
) -> Tuple[int, List[th.SE3], List[PoseGraphEdge]]:
    with open(path, "r") as file:
        lines = file.readlines()

        num_vertices = 0
        verts = dict()
        edges: List[PoseGraphEdge] = []

        for line in lines:
            tokens = line.split()

            if tokens[0] == "EDGE_SE3:QUAT":
                i = int(tokens[1])
                j = int(tokens[2])

                n = len(edges)

                x_y_z_quat = torch.from_numpy(
                    np.array([tokens[3:10]], dtype=np.float64)
                ).to(dtype)
                x_y_z_quat[:, 3:] /= torch.norm(x_y_z_quat[:, 3:], dim=1)
                x_y_z_quat[:, 3:] = x_y_z_quat[:, [6, 3, 4, 5]]
                relative_pose = th.SE3(
                    x_y_z_quaternion=x_y_z_quat, name="EDGE_SE3__{}".format(n)
                )

                sel = [0, 6, 11, 15, 18, 20]
                weight = th.Variable(
                    torch.from_numpy(np.array(tokens[10:], dtype=np.float64)[sel])
                    .to(dtype)
                    .sqrt()
                    .view(1, -1)
                )

                edges.append(
                    PoseGraphEdge(
                        i,
                        j,
                        relative_pose,
                        th.DiagonalCostWeight(weight, name="EDGE_WEIGHT__{}".format(n)),
                    )
                )

                num_vertices = max(num_vertices, i)
                num_vertices = max(num_vertices, j)
            elif tokens[0] == "VERTEX_SE3:QUAT":
                i = int(tokens[1])

                x_y_z_quat = torch.from_numpy(
                    np.array([tokens[2:]], dtype=np.float64)
                ).to(dtype)
                x_y_z_quat[:, 3:] /= torch.norm(x_y_z_quat[:, 3:], dim=1)
                x_y_z_quat[:, 3:] = x_y_z_quat[:, [6, 3, 4, 5]]
                verts[i] = x_y_z_quat

                num_vertices = max(num_vertices, i)

        num_vertices += 1

        if len(verts) > 0:
            vertices = [
                th.SE3(x_y_z_quaternion=x_y_z_quat, name="VERTEX_SE3__{}".format(i))
                for i, x_y_z_quat in sorted(verts.items())
            ]
        else:
            vertices = []

        return (num_vertices, vertices, edges)


# This function reads a file in g2o formate and returns the number of of poses, initial
# values and edges
# g2o format: https://github.com/RainerKuemmerle/g2o/wiki/File-Format-SLAM-2D
def read_2D_g2o_file(
    path: str, dtype: Optional[torch.dtype] = None
) -> Tuple[int, List[th.SE2], List[PoseGraphEdge]]:
    with open(path, "r") as file:
        lines = file.readlines()

        num_vertices = 0
        verts = dict()
        edges: List[PoseGraphEdge] = []

        for line in lines:
            tokens = line.split()

            if tokens[0] == "EDGE_SE2":
                i = int(tokens[1])
                j = int(tokens[2])

                n = len(edges)

                x_y_theta = torch.from_numpy(
                    np.array([tokens[3:6]], dtype=np.float64)
                ).to(dtype)
                relative_pose = th.SE2(
                    x_y_theta=x_y_theta, name="EDGE_SE2__{}".format(n)
                )

                sel = [0, 3, 5]
                weight = th.Variable(
                    torch.from_numpy(np.array(1, tokens[6:], dtype=np.float64)[sel])
                    .to(dtype)
                    .sqrt()
                    .view(1, -1)
                )

                edges.append(
                    PoseGraphEdge(
                        i,
                        j,
                        relative_pose,
                        th.DiagonalCostWeight(weight, name="EDGE_WEIGHT__{}".format(n)),
                    )
                )

                num_vertices = max(num_vertices, i)
                num_vertices = max(num_vertices, j)
            elif tokens[0] == "VERTEX_SE2":
                i = int(tokens[1])

                x_y_theta = torch.from_numpy(
                    np.array([tokens[2:]], dtype=np.float64)
                ).to(dtype)
                verts[i] = x_y_theta

                num_vertices = max(num_vertices, i)

        num_vertices += 1

        if len(verts) > 0:
            vertices = [
                th.SE2(x_y_theta=x_y_theta, name="VERTEX_SE2__{}".format(i))
                for i, x_y_theta in sorted(verts.items())
            ]
        else:
            vertices = []

        return (num_vertices, vertices, edges)


class PoseGraphDataset:
    def __init__(
        self,
        poses: Union[List[th.SE2], List[th.SE3]],
        edges: List[PoseGraphEdge],
        gt_poses: Optional[Union[List[th.SE2], List[th.SE3]]] = None,
        batch_size: int = 1,
        device: th.DeviceType = None,
    ):
        dataset_sizes: List[int] = [pose.shape[0] for pose in poses]
        if gt_poses is not None:
            dataset_sizes.extend([gt_pose.shape[0] for gt_pose in gt_poses])
        dataset_sizes.extend([edge.relative_pose.shape[0] for edge in edges])
        uniqe_batch_sizes = set(dataset_sizes)

        if len(uniqe_batch_sizes) != 1:
            raise ValueError("Provided data has muliple batches.")

        self.poses = poses
        self.edges = edges
        self.gt_poses = gt_poses
        self.batch_size = batch_size
        self.dataset_size = dataset_sizes[0]
        self.num_batches = (self.dataset_size - 1) // self.batch_size + 1

        self.to(device=device)

    def load_3D_g2o_file(
        self, path: str, dtype: Optional[torch.dtype] = None
    ) -> "PoseGraphDataset":
        _, poses, edges = read_3D_g2o_file(path, dtype)
        return PoseGraphDataset(poses, edges)

    def load_2D_g2o_file(
        self, path: str, dtype: Optional[torch.dtype] = None
    ) -> "PoseGraphDataset":
        _, poses, edges = read_2D_g2o_file(path, dtype)
        return PoseGraphDataset(poses, edges)

    def histogram(self) -> str:
        buckets = np.zeros(11)
        for edge in self.edges:
            error = self.poses[edge.j].local(
                self.poses[edge.i].compose(edge.relative_pose)
            )
            error_norm = error.norm(dim=1)
            idx = (10 * error_norm).to(dtype=int)
            idx = torch.where(idx > len(buckets) - 1, len(buckets) - 1, idx)
            for i in idx:
                buckets[i] += 1
        max_buckets = max(buckets)
        hist_str = ""
        for i in range(len(buckets)):
            bi = buckets[i]
            label = f"{i}-{i+1}" if i + 1 < len(buckets) else f"{i}+"
            barlen = round(bi * 80 / max_buckets)
            hist_str += f"{label}: {'#' * barlen} {bi}\n"
        return hist_str

    @staticmethod
    def generate_synthetic_3D(
        num_poses: int,
        rotation_noise: float = 0.05,
        translation_noise: float = 0.1,
        loop_closure_ratio: float = 0.2,
        loop_closure_outlier_ratio: float = 0.05,
        max_num_loop_closures: int = 10,
        dataset_size: int = 1,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple["PoseGraphDataset", List[bool]]:
        poses: List[th.SE3] = list()
        gt_poses: List[th.SE3] = list()
        edges = list()
        inliers = list()

        poses.append(
            th.SE3(
                tensor=torch.tile(torch.eye(3, 4, dtype=dtype), [dataset_size, 1, 1]),
                name="VERTEX_SE3__0",
            )
        )
        gt_poses.append(
            th.SE3(
                tensor=torch.tile(torch.eye(3, 4, dtype=dtype), [dataset_size, 1, 1]),
                name="VERTEX_SE3_GT__0",
            )
        )

        info = torch.tensor(
            [1 / translation_noise] * 3 + [1 / rotation_noise] * 3, dtype=dtype
        ).view(1, -1)

        for n in range(1, num_poses):
            gt_relative_pose = th.SE3.exp_map(
                torch.cat(
                    [
                        torch.rand(dataset_size, 3, dtype=dtype) - 0.5,
                        2.0 * torch.rand(dataset_size, 3, dtype=dtype) - 1,
                    ],
                    dim=1,
                )
            )
            noise_relative_pose = th.SE3.exp_map(
                torch.cat(
                    [
                        rotation_noise
                        * (2 * torch.rand(dataset_size, 3, dtype=dtype) - 1),
                        translation_noise
                        * (2.0 * torch.rand(dataset_size, 3, dtype=dtype) - 1),
                    ],
                    dim=1,
                )
            )
            relative_pose = cast(th.SE3, gt_relative_pose.compose(noise_relative_pose))
            relative_pose.name = "EDGE_SE3__{}_{}".format(n - 1, n)
            weight = th.DiagonalCostWeight(
                th.Variable(info),
                name="EDGE_WEIGHT__{}_{}".format(n - 1, n),
            )

            gt_pose = cast(th.SE3, gt_poses[-1].compose(gt_relative_pose))
            gt_pose.name = "VERTEX_SE3_GT__{}".format(n)
            pose = cast(th.SE3, poses[-1].compose(relative_pose))
            pose.name = "VERTEX_SE3__{}".format(n)

            gt_poses.append(gt_pose)
            poses.append(pose)
            edges.append(
                PoseGraphEdge(n - 1, n, relative_pose=relative_pose, weight=weight)
            )
            inliers.append(True)

            if np.random.rand(1) <= loop_closure_ratio and n - 1 > 0:
                num_loop_closures = np.random.randint(max_num_loop_closures) + 1
                indices = set(np.random.randint(0, n - 1, num_loop_closures))
                j = n

                for i in indices:
                    gt_relative_pose = cast(
                        th.SE3, gt_poses[i].inverse().compose(gt_poses[j])
                    )
                    if np.random.rand(1) > loop_closure_outlier_ratio:
                        noise_relative_pose = th.SE3.exp_map(
                            torch.cat(
                                [
                                    rotation_noise
                                    * (2 * torch.rand(1, 3, dtype=dtype) - 1),
                                    translation_noise
                                    * (2.0 * torch.rand(1, 3, dtype=dtype) - 1),
                                ],
                                dim=1,
                            )
                        )
                        inliers.append(True)
                    else:
                        noise_relative_pose = th.SE3.rand(
                            dataset_size, generator=generator, dtype=dtype
                        )
                        inliers.append(False)

                    relative_pose = cast(
                        th.SE3, gt_relative_pose.compose(noise_relative_pose)
                    )
                    relative_pose.name = "EDGE_SE3__{}_{}".format(i, j)

                    weight = th.DiagonalCostWeight(
                        th.Variable(info),
                        name="EDGE_WEIGHT__{}_{}".format(i, j),
                    )
                    edges.append(
                        PoseGraphEdge(i, j, relative_pose=relative_pose, weight=weight)
                    )

        for i in range(len(poses)):
            noise_pose = th.SE3.exp_map(
                torch.cat(
                    [
                        rotation_noise * (2 * torch.rand(1, 3, dtype=dtype) - 1),
                        translation_noise * (2.0 * torch.rand(1, 3, dtype=dtype) - 1),
                    ],
                    dim=1,
                )
            )
            poses[i].tensor = gt_poses[i].compose(noise_pose).tensor

        return PoseGraphDataset(poses, edges, gt_poses, batch_size=batch_size), inliers

    def write_3D_g2o(self, filename: str):
        for n in range(self.dataset_size):
            with open(filename + f"_{n}.g2o", "w") as file:
                for edge in self.edges:
                    measurement = edge.relative_pose.tensor[n : n + 1]
                    quat = th.SO3(
                        tensor=measurement[:, :, :3], strict=False
                    ).to_quaternion()
                    tran = measurement[:, :, 3]
                    measurement = torch.cat([tran, quat], dim=1).view(-1).numpy()
                    weight = edge.weight.diagonal.tensor**2
                    line = (
                        f"EDGE_SE3:QUAT {edge.i} {edge.j} {measurement[0]} {measurement[1]} "
                        f"{measurement[2]} "
                        f"{measurement[4]} {measurement[5]} "
                        f"{measurement[6]} {measurement[3]} "
                        f"{weight[0,0]} 0 0 0 0 0 {weight[0,1]} 0 0 0 0 {weight[0,2]} 0 0 0 "
                        f"{weight[0,3]} 0 0 {weight[0,4]} 0 {weight[0,5]}\n"
                    )
                    file.write(line)
                for i, pose in enumerate(self.poses):
                    pose_n = pose[n : n + 1]
                    quat = th.SO3(tensor=pose_n[:, :, :3], strict=False).to_quaternion()
                    tran = pose_n[:, :, 3]
                    pose_data = torch.cat([tran, quat], dim=1).view(-1).numpy()
                    line = (
                        f"VERTEX_SE3:QUAT {i} {pose_data[0]} {pose_data[1]} {pose_data[2]} "
                        f"{pose_data[4]} {pose_data[5]} {pose_data[6]} {pose_data[3]}\n"
                    )
                    file.write(line)
                file.close()

    def get_batch_dataset(self, batch_idx: int = 0) -> "PoseGraphDataset":
        assert batch_idx < self.num_batches
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.dataset_size)
        group_cls = self.poses[0].__class__

        poses = cast(
            Union[List[th.SE2], List[th.SE3]],
            [
                group_cls(tensor=pose[start:end].clone(), name=pose.name + "__batch")
                for pose in self.poses
            ],
        )
        if self.gt_poses is not None:
            gt_poses = cast(
                Union[List[th.SE2], List[th.SE3]],
                [
                    group_cls(
                        tensor=gt_pose[start:end].clone(), name=gt_pose.name + "__batch"
                    )
                    for gt_pose in self.gt_poses
                ],
            )
        else:
            gt_poses = None
        edges = [
            PoseGraphEdge(
                edge.i,
                edge.j,
                relative_pose=group_cls(
                    tensor=edge.relative_pose[start:end].clone(),
                    name=edge.relative_pose.name + "__batch",
                ),
                weight=edge.weight,
            )
            for edge in self.edges
        ]

        return PoseGraphDataset(poses, edges, gt_poses, batch_size=self.batch_size)

    def to(self, *args, **kwargs):
        if self.gt_poses is not None:
            for gt_pose in self.gt_poses:
                gt_pose.to(*args, **kwargs)

        if self.poses is not None:
            for pose in self.poses:
                pose.to(*args, **kwargs)

        if self.edges is not None:
            for edge in self.edges:
                edge.to(*args, **kwargs)


def pg_histogram(
    poses: Union[List[th.SE2], List[th.SE3]], edges: List[PoseGraphEdge]
) -> str:
    pg = PoseGraphDataset(poses=poses, edges=edges)
    return pg.histogram()
