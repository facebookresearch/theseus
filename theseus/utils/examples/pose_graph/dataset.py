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


# This function reads a file in g2o formate and returns the number of of poses, initial
# values and edges.
# g2O format: https://github.com/RainerKuemmerle/g2o/wiki/File-format-slam-3d
def read_3D_g2o_file(path: str) -> Tuple[int, List[th.SE3], List[PoseGraphEdge]]:
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
                relative_pose = th.SE3(
                    x_y_z_quaternion=x_y_z_quat, name="EDGE_SE3__{}".format(n)
                )

                sel = [0, 6, 11, 15, 18, 20]
                weight = th.Variable(
                    torch.from_numpy(np.array(tokens[10:], dtype=np.float64)[sel])
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
def read_2D_g2o_file(path: str) -> Tuple[int, List[th.SE2], List[PoseGraphEdge]]:
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

                x_y_theta = torch.from_numpy(np.array([tokens[3:6]], dtype=np.float64))
                relative_pose = th.SE2(
                    x_y_theta=x_y_theta, name="EDGE_SE2__{}".format(n)
                )

                sel = [0, 3, 5]
                weight = th.Variable(
                    torch.from_numpy(np.array(tokens[6:], dtype=np.float64)[sel])
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
            elif tokens[0] == "VERTEX_SE2:QUAT":
                i = int(tokens[1])

                x_y_theta = torch.from_numpy(np.array([tokens[2:]], dtype=np.float64))
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
    ):
        self.poses = poses
        self.edges = edges
        self.gt_poses = gt_poses

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

    @staticmethod
    def generate_synthetic_3D(
        num_poses: int,
        rotation_noise: float = 0.05,
        translation_noise: float = 0.1,
        loop_closure_ratio: float = 0.2,
        loop_closure_outlier_ratio: float = 0.05,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple["PoseGraphDataset", List[bool]]:
        poses = list()
        gt_poses = list()
        edges = list()
        inliers = list()

        poses.append(
            th.SE3(
                data=torch.eye(3, 4, dtype=dtype).reshape(1, 3, 4), name="VERTEX_SE3__0"
            )
        )
        gt_poses.append(th.SE3(data=torch.eye(3, 4, dtype=dtype).reshape(1, 3, 4)))

        for n in range(1, num_poses):
            gt_relative_pose = th.SE3.exp_map(
                torch.cat(
                    [
                        torch.rand(1, 3, dtype=dtype) - 0.5,
                        2.0 * torch.rand(1, 3, dtype=dtype) - 1,
                    ],
                    dim=1,
                )
            )
            noise_relative_pose = th.SE3.exp_map(
                torch.cat(
                    [
                        rotation_noise * (2 * torch.rand(1, 3, dtype=dtype) - 1),
                        translation_noise * (2.0 * torch.rand(1, 3, dtype=dtype) - 1),
                    ],
                    dim=1,
                )
            )
            relative_pose = cast(th.SE3, gt_relative_pose.compose(noise_relative_pose))
            relative_pose.name = "EDGE_SE3__{}_{}".format(n - 1, n)
            weight = th.DiagonalCostWeight(
                th.Variable(torch.ones(1, 6, dtype=dtype)),
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
                i = np.random.randint(n - 1)
                j = n - 1

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
                        1, generator=generator, dtype=dtype
                    )
                    inliers.append(False)

                relative_pose = cast(
                    th.SE3, gt_relative_pose.compose(noise_relative_pose)
                )
                relative_pose.name = "EDGE_SE3__{}_{}".format(i, j)

                weight = th.DiagonalCostWeight(
                    th.Variable(10 * torch.ones(1, 6, dtype=dtype)),
                    name="EDGE_WEIGHT__{}_{}".format(i, j),
                )
                edges.append(
                    PoseGraphEdge(i, j, relative_pose=relative_pose, weight=weight)
                )

        return PoseGraphDataset(poses, edges, gt_poses), inliers
