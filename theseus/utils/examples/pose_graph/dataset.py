# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

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
                )
                x_y_z_quat[3:] /= torch.norm(x_y_z_quat[3:])
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

                x_y_z_quat = torch.from_numpy(np.array([tokens[2:]], dtype=np.float64))
                x_y_z_quat[3:] /= torch.norm(x_y_z_quat[3:])
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
