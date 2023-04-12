# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import torch
from scipy import ndimage

from theseus.core import Variable, as_variable
from theseus.geometry import Point2
from theseus.utils import gather_from_rows_cols


class SignedDistanceField2D:
    # origin shape is:  batch_size x 2
    # sdf_data shape is: batch_size x field_height x field_width
    def __init__(
        self,
        origin: Union[Point2, torch.Tensor],
        cell_size: Union[float, torch.Tensor, Variable],
        sdf_data: Optional[Union[torch.Tensor, Variable]] = None,
        occupancy_map: Optional[Union[torch.Tensor, Variable]] = None,
        occupancy_threshold: float = 0.75,
        sdf_boundary_value: float = 0.0,
    ):
        if occupancy_map is not None:
            if sdf_data is not None:
                raise ValueError(
                    "Only one of sdf_data and occupancy_map should be provided."
                )
            sdf_data = self._compute_sdf_data_from_map(
                occupancy_map,
                SignedDistanceField2D.convert_cell_size(cell_size).tensor,
                threshold=occupancy_threshold,
            )
        else:
            if sdf_data is None:
                raise ValueError(
                    "Either sdf_data or argument occupancy_map should be provided."
                )
        self.origin: Point2
        self.cell_size: Variable
        self.sdf_data: Variable
        self.update_data(origin, sdf_data, cell_size)
        self._num_rows = sdf_data.shape[1]
        self._num_cols = sdf_data.shape[2]
        self.sdf_boundary_value = sdf_boundary_value

    def _compute_sdf_data_from_map(
        self,
        occupancy_map_batch: Union[Variable, torch.Tensor],
        cell_size: torch.Tensor,
        threshold: float = 0.75,
    ) -> Variable:
        if isinstance(occupancy_map_batch, Variable):
            occupancy_map_batch = occupancy_map_batch.tensor
        if cell_size.shape[0] != occupancy_map_batch.shape[0]:
            cell_size = cell_size.expand(occupancy_map_batch.shape[0], 1)

        # Code from https://github.com/gtrll/gpmp2/
        if occupancy_map_batch.ndim != 3:
            raise ValueError(
                "Argument occupancy_map to SignedDistanceField2D must be "
                "a batch of matrices."
            )
        num_maps = occupancy_map_batch.shape[0]
        all_sdf_data = []

        for i in range(num_maps):
            occupancy_map = occupancy_map_batch[i]

            cur_map = occupancy_map > threshold
            cur_map = cur_map.int()

            if torch.max(cur_map) == 0:
                map_x, map_y = occupancy_map.size(0), occupancy_map.size(1)
                max_map_size = 2 * cell_size[i].item() * max(map_x, map_y)
                sdf_data = (
                    torch.ones(occupancy_map.shape, dtype=occupancy_map.dtype)
                    * max_map_size
                )
            else:
                # inverse map
                inv_map = 1 - cur_map
                # get signed distance from map and inverse map
                # since bwdist(foo) = ndimage.distance_transform_edt(1-foo)
                map_dist = ndimage.distance_transform_edt(inv_map.numpy())
                inv_map_dist = ndimage.distance_transform_edt(cur_map.numpy())

                sdf_data = map_dist - inv_map_dist
                # metric
                sdf_data = (
                    torch.tensor(sdf_data, dtype=occupancy_map.dtype) * cell_size[i]
                )

            all_sdf_data.append(sdf_data)

        sdf_data_var = Variable(torch.stack(all_sdf_data))

        return sdf_data_var

    @staticmethod
    def convert_origin(origin: Union[torch.Tensor, Point2]) -> Point2:
        if not isinstance(origin, Point2) and not isinstance(origin, torch.Tensor):
            raise ValueError(
                "Argument origin to SignedDistanceField2D must be either "
                "a tensor or a Point2 variable."
            )
        if not isinstance(origin, Point2):
            try:
                return Point2(tensor=origin)
            except ValueError:
                raise ValueError(
                    "Argument origin to SignedDistanceField2D must be a batch of "
                    "2D tensors."
                )
        return origin

    @staticmethod
    def convert_cell_size(cell_size: Union[float, torch.Tensor, Variable]) -> Variable:
        if not isinstance(cell_size, Variable):
            if not isinstance(cell_size, torch.Tensor):
                if not isinstance(cell_size, float):
                    raise ValueError(
                        "Argument cell_size must be either a Variable, "
                        "tensor, or float."
                    )
                cell_size = torch.tensor(cell_size).view(-1, 1)
            return Variable(cell_size)
        if not (
            cell_size.ndim == 1 or (cell_size.ndim == 2 and cell_size.shape[1] == 1)
        ):
            raise ValueError("Argument cell_size must be a batch of 0D or 1D tensors.")
        return cell_size

    @staticmethod
    def convert_sdf_data(sdf_data: Union[torch.Tensor, Variable]) -> Variable:
        sdf_data = as_variable(sdf_data)
        if sdf_data.ndim != 3:
            raise ValueError(
                "Argument sdf_data to SignedDistanceField2D must be a "
                "batch of matrices."
            )
        return sdf_data

    def update_data(
        self,
        origin: Union[torch.Tensor, Point2],
        sdf_data: Union[torch.Tensor, Variable],
        cell_size: Union[float, torch.Tensor, Variable],
    ):
        # Update origin
        self.origin = SignedDistanceField2D.convert_origin(origin)

        # Update cell size
        self.cell_size = SignedDistanceField2D.convert_cell_size(cell_size)

        # Update sdf_data
        self.sdf_data = SignedDistanceField2D.convert_sdf_data(sdf_data)

    def convert_points_to_cell(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin = (
            self.origin.tensor.unsqueeze(-1)
            if self.origin.ndim == 2
            else self.origin.tensor
        )
        cell_size = (
            self.cell_size.tensor
            if self.cell_size.ndim == 2
            else self.cell_size.tensor.unsqueeze(-1)
        )
        px = points[:, 0]
        py = points[:, 1]

        out_of_bounds_idx = (
            (px < origin[:, 0])
            .logical_or(px > (origin[:, 0] + (self._num_cols - 1.0) * cell_size))
            .logical_or(py < origin[:, 1])
            .logical_or(py > (origin[:, 1] + (self._num_rows - 1.0) * cell_size))
        )

        col = (px - origin[:, 0]) / cell_size
        row = (py - origin[:, 1]) / cell_size
        return row, col, out_of_bounds_idx

    # Points shape must be (batch_size, 2, num_points)
    #   Computed distances for each point, shape (batch_size, num_points)
    #   Jacobian tensor, shape (batch_size, num_points, 2)
    def signed_distance(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, cols, out_of_bounds_idx = self.convert_points_to_cell(points)

        lr = torch.floor(rows)
        lc = torch.floor(cols)
        hr = lr + 1.0
        hc = lc + 1.0
        lri = lr.long().clamp(0, self._num_rows - 1)
        lci = lc.long().clamp(0, self._num_cols - 1)
        hri = hr.long().clamp(0, self._num_rows - 1)
        hci = hc.long().clamp(0, self._num_cols - 1)

        def gather_sdf(r_, c_):
            return gather_from_rows_cols(self.sdf_data.tensor, r_, c_)

        # Compute the distance
        hrdiff = hr - rows
        hcdiff = hc - cols
        lrdiff = rows - lr
        lcdiff = cols - lc
        dist = (
            hrdiff * hcdiff * gather_sdf(lri, lci)
            + lrdiff * hcdiff * gather_sdf(hri, lci)
            + hrdiff * lcdiff * gather_sdf(lri, hci)
            + lrdiff * lcdiff * gather_sdf(hri, hci)
        )
        dist[out_of_bounds_idx] = self.sdf_boundary_value

        # Compute the jacobians

        cell_size = (
            self.cell_size.tensor
            if self.cell_size.ndim == 2
            else self.cell_size.tensor.unsqueeze(-1)
        )

        jac1 = (
            hrdiff * (gather_sdf(lri, hci) - gather_sdf(lri, lci))
            + lrdiff * (gather_sdf(hri, hci) - gather_sdf(hri, lci))
        ) / cell_size
        jac2 = (
            hcdiff * (gather_sdf(hri, lci) - gather_sdf(lri, lci))
            + lcdiff * (gather_sdf(hri, hci) - gather_sdf(lri, hci))
        ) / cell_size
        jac1[out_of_bounds_idx] = 0
        jac2[out_of_bounds_idx] = 0
        return dist, torch.stack([jac1, jac2], dim=2)

    def to(self, *args, **kwargs):
        self.cell_size.to(*args, **kwargs)
        self.origin.to(*args, **kwargs)
        self.sdf_data.to(*args, **kwargs)
