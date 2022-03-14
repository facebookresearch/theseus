# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import numpy as np
import torch

import theseus as th


class PoseGraph3DEdge:
    def __init__(self, i: int, j: int, relative_pose: th.SE3):
        self.i = i
        self.j = j
        self.measurement = relative_pose


class PoseGraph2DEdge:
    def __init__(self, i: int, j: int, relative_pose: th.SE2):
        self.i = i
        self.j = j
        self.measurement = relative_pose


def read_3D_g2o_file(path: str) -> Tuple[int, List[th.SE3], List[PoseGraph3DEdge]]:
    with open(path, "r") as file:
        lines = file.readlines()

        num_vertices = 0
        verts = dict()
        edges = []

        for line in lines:
            tokens = line.split()

            if tokens[0] == "EDGE_SE3:QUAT":
                i = int(tokens[1])
                j = int(tokens[2])

                x_y_z_quat = torch.from_numpy(
                    np.array([tokens[3:10]], dtype=np.float64)
                )
                x_y_z_quat[3:] /= torch.norm(x_y_z_quat[3:])
                relative_pose = th.SE3(x_y_z_quaternion=x_y_z_quat)
                edges.append(PoseGraph3DEdge(i, j, relative_pose))

                num_vertices = max(num_vertices, i)
                num_vertices = max(num_vertices, j)

            if tokens[0] == "VERTEX_SE3:QUAT":
                i = int(tokens[1])

                x_y_z_quat = torch.from_numpy(np.array([tokens[2:]], dtype=np.float64))
                x_y_z_quat[3:] /= torch.norm(x_y_z_quat[3:])
                verts[i] = x_y_z_quat

                num_vertices = max(num_vertices, i)

        num_vertices += 1

        if len(verts) > 0:
            vertices = [
                th.SE3(x_y_z_quaternion=x_y_z_quat, name="VERTEX_SE3_{}".format(i))
                for i, x_y_z_quat in sorted(verts.items())
            ]
        else:
            vertices = []

        return (num_vertices, vertices, edges)


def read_2D_g2o_file(path: str) -> Tuple[int, List[th.SE2], List[PoseGraph2DEdge]]:
    with open(path, "r") as file:
        lines = file.readlines()

        num_vertices = 0
        verts = dict()
        edges = []

        for line in lines:
            tokens = line.split()

            if tokens[0] == "EDGE_SE2":
                i = int(tokens[1])
                j = int(tokens[2])

                x_y_theta = torch.from_numpy(np.array([tokens[3:6]], dtype=np.float64))
                relative_pose = th.SE2(x_y_theta=x_y_theta)
                edges.append(PoseGraph2DEdge(i, j, relative_pose))

                num_vertices = max(num_vertices, i)
                num_vertices = max(num_vertices, j)

            if tokens[0] == "VERTEX_SE3:QUAT":
                i = int(tokens[1])

                x_y_theta = torch.from_numpy(np.array([tokens[2:]], dtype=np.float64))
                verts[i] = x_y_theta

                num_vertices = max(num_vertices, i)

        num_vertices += 1

        if len(verts) > 0:
            vertices = [
                th.SE2(x_y_theta=x_y_theta, name="VERTEX_SE2_{}".format(i))
                for i, x_y_theta in sorted(verts.items())
            ]
        else:
            vertices = []

        return (num_vertices, vertices, edges)
