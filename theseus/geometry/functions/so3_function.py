# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from theseus.geometry.lie_group_check import _LieGroupCheckContext
from theseus.geometry.functions.utils import check_jacobians_list
import theseus.constants
from theseus.geometry.functions.lie_group_function import LieGroupFunction

from typing import List, Optional, cast

import torch


class SO3Function(LieGroupFunction):
    @staticmethod
    def dim() -> int:
        return 3

    @staticmethod
    def check_group_tensor(tensor: torch.Tensor) -> bool:
        with torch.no_grad():
            if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
                raise ValueError("SO3 data tensors can only be 3x3 matrices.")

            MATRIX_EPS = theseus.constants._SO3_MATRIX_EPS[tensor.dtype]
            if tensor.dtype != torch.float64:
                tensor = tensor.double()

            _check = (
                torch.matmul(tensor, tensor.transpose(1, 2))
                - torch.eye(3, 3, dtype=tensor.dtype, device=tensor.device)
            ).abs().max().item() < MATRIX_EPS
            _check &= (torch.linalg.det(tensor) - 1).abs().max().item() < MATRIX_EPS

        return _check

    @staticmethod
    def check_tangent_vector(tangent_vector: torch.Tensor) -> bool:
        _check = tangent_vector.ndim == 3 and tangent_vector.shape[1:] == (3, 1)
        _check |= tangent_vector.ndim == 2 and tangent_vector.shape[1] == 3
        return _check

    @staticmethod
    def check_hat_matrix(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
            raise ValueError("Hat matrices of SO(3) can only be 3x3 matrices")

        checks_enabled, silent_unchecks = _LieGroupCheckContext.get_context()
        if checks_enabled:
            if (
                matrix.transpose(1, 2) + matrix
            ).abs().max().item() > theseus.constants._SO3_HAT_EPS[matrix.dtype]:
                raise ValueError("Hat matrices of SO(3) can only be skew-symmetric.")
        elif not silent_unchecks:
            warnings.warn(
                "Lie group checks are disabled, so the skew-symmetry of hat matrices is "
                "not checked for SO3.",
                RuntimeWarning,
            )

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: theseus.constants.DeviceType = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return SO3Function.exp_map.call(
            theseus.constants.PI
            * torch.randn(
                size[0],
                3,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        )

    class project(LieGroupFunction.project):
        @staticmethod
        def call(matrix) -> torch.Tensor:
            if matrix.shape[-2:] != (3, 3):
                raise ValueError("Inconsistent shape for matrix.")

            return torch.stack(
                (
                    matrix[..., 2, 1] - matrix[..., 1, 2],
                    matrix[..., 0, 2] - matrix[..., 2, 0],
                    matrix[..., 1, 0] - matrix[..., 0, 1],
                ),
                dim=1,
            )

        @staticmethod
        def backward(ctx, grad_output):
            return SO3Function.hat.call(grad_output)

    class left_project(LieGroupFunction.left_project):
        @staticmethod
        def manifold() -> type:
            return SO3Function

    class left_apply(LieGroupFunction.left_apply):
        @staticmethod
        def call(
            group: torch.Tensor,
            matrix: torch.Tensor,
            jacobians: Optional[List[torch.Tensor]] = None,
        ) -> torch.Tensor:
            if not SO3Function.check_group_tensor(group):
                raise ValueError("Invalid SO3 data tensors.")

            if matrix.ndim != 3 or matrix.shape[1] != 3:
                raise ValueError("The matrix tensor must have 3 rows.")

            if jacobians is not None:
                check_jacobians_list(jacobians)
                check_jacobians_list(jacobians)
                jacobians.append(SO3Function.left_apply.call(group, matrix))
                jacobians.append(
                    group.view(group.shape + (1,)).expand(group.shape + (3,))
                )

            return group @ matrix

        @staticmethod
        def forward(ctx, group, matrix, jacobians):
            ctx.save_for_backward(group, matrix)
            return SO3Function.left_apply.call(group, matrix, jacobians)

        @staticmethod
        def backward(ctx, grad_output):
            group = ctx.saved_tensors[0]
            matrix = ctx.saved_tensors[1]
            return (
                grad_output @ matrix.transpose(1, 2),
                group.transpose(1, 2) @ grad_output,
            )

    class hat(LieGroupFunction.hat):
        @staticmethod
        def call(tangent_vector: torch.Tensor) -> torch.Tensor:
            if not SO3Function.check_tangent_vector(tangent_vector):
                raise ValueError("Tangent vectors of SO3 should be 3-D vectors.")
            matrix = tangent_vector.new_zeros(tangent_vector.shape[0], 3, 3)

            matrix[:, 0, 1] = -tangent_vector[:, 2].view(-1)
            matrix[:, 0, 2] = tangent_vector[:, 1].view(-1)
            matrix[:, 1, 2] = -tangent_vector[:, 0].view(-1)
            matrix[:, 1, 0] = tangent_vector[:, 2].view(-1)
            matrix[:, 2, 0] = -tangent_vector[:, 1].view(-1)
            matrix[:, 2, 1] = tangent_vector[:, 0].view(-1)

            return matrix

        @staticmethod
        def forward(ctx, tangent_vector):
            return SO3Function.hat.call(tangent_vector)

        @staticmethod
        def backward(ctx, grad_output):
            grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
            return torch.stack(
                (
                    grad_output[:, 2, 1] - grad_output[:, 1, 2],
                    grad_output[:, 0, 2] - grad_output[:, 2, 0],
                    grad_output[:, 1, 0] - grad_output[:, 0, 1],
                ),
                dim=1,
            )

    class vee(LieGroupFunction.vee):
        @staticmethod
        def call(matrix: torch.Tensor) -> torch.Tensor:
            SO3Function.check_hat_matrix(matrix)
            return 0.5 * torch.stack(
                (
                    matrix[:, 2, 1] - matrix[:, 1, 2],
                    matrix[:, 0, 2] - matrix[:, 2, 0],
                    matrix[:, 1, 0] - matrix[:, 0, 1],
                ),
                dim=1,
            )

        @staticmethod
        def forward(ctx, tangent_vector):
            return SO3Function.vee.call(tangent_vector)

        @staticmethod
        def backward(ctx, grad_output):
            grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
            return 0.5 * SO3Function.hat.call(grad_output)

    class exp_map(LieGroupFunction.exp_map):
        @staticmethod
        def call(
            tangent_vector: torch.Tensor,
            jacobians: Optional[List[torch.Tensor]] = None,
        ) -> torch.Tensor:
            if not SO3Function.check_tangent_vector(tangent_vector):
                raise ValueError("Tangent vectors of SO3 should be 3-D vectors.")
            tangent_vector = tangent_vector.view(-1, 3)
            ret = tangent_vector.new_zeros(tangent_vector.shape[0], 3, 3)
            theta = torch.linalg.norm(tangent_vector, dim=1, keepdim=True).unsqueeze(1)
            theta2 = theta**2
            # Compute the approximations when theta ~ 0
            near_zero = (
                theta < theseus.constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
            )
            non_zero = torch.ones(
                1, dtype=tangent_vector.dtype, device=tangent_vector.device
            )
            theta_nz = torch.where(near_zero, non_zero, theta)
            theta2_nz = torch.where(near_zero, non_zero, theta2)

            cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
            sine = theta.sin()
            sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
            one_minus_cosie_by_theta2 = torch.where(
                near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
            )
            ret = (
                one_minus_cosie_by_theta2
                * tangent_vector.view(-1, 3, 1)
                @ tangent_vector.view(-1, 1, 3)
            )

            ret[:, 0, 0] += cosine.view(-1)
            ret[:, 1, 1] += cosine.view(-1)
            ret[:, 2, 2] += cosine.view(-1)
            sine_axis = sine_by_theta.view(-1, 1) * tangent_vector
            ret[:, 0, 1] -= sine_axis[:, 2]
            ret[:, 1, 0] += sine_axis[:, 2]
            ret[:, 0, 2] += sine_axis[:, 1]
            ret[:, 2, 0] -= sine_axis[:, 1]
            ret[:, 1, 2] -= sine_axis[:, 0]
            ret[:, 2, 1] += sine_axis[:, 0]

            if jacobians is not None:
                check_jacobians_list(jacobians)
                theta3_nz = theta_nz * theta2_nz
                theta_minus_sine_by_theta3 = torch.where(
                    near_zero, torch.zeros_like(theta), (theta - sine) / theta3_nz
                )
                jac = (
                    theta_minus_sine_by_theta3
                    * tangent_vector.view(-1, 3, 1)
                    @ tangent_vector.view(-1, 1, 3)
                )
                diag_jac = jac.diagonal(dim1=1, dim2=2)
                diag_jac += sine_by_theta.view(-1, 1)

                jac_temp = one_minus_cosie_by_theta2.view(-1, 1) * tangent_vector

                jac[:, 0, 1] += jac_temp[:, 2]
                jac[:, 1, 0] -= jac_temp[:, 2]
                jac[:, 0, 2] -= jac_temp[:, 1]
                jac[:, 2, 0] += jac_temp[:, 1]
                jac[:, 1, 2] += jac_temp[:, 0]
                jac[:, 2, 1] -= jac_temp[:, 0]

                jacobians.append(jac)

            return ret

        @staticmethod
        def jacobian(tangent_vector: torch.Tensor) -> torch.Tensor:
            if not SO3Function.check_tangent_vector(tangent_vector):
                raise ValueError("Tangent vectors of SO3 should be 3-D vectors.")
            tangent_vector = tangent_vector.view(-1, 3)
            theta = torch.linalg.norm(tangent_vector, dim=1, keepdim=True).unsqueeze(1)
            theta2 = theta**2
            # Compute the approximations when theta ~ 0
            near_zero = (
                theta < theseus.constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
            )
            non_zero = torch.ones(
                1, dtype=tangent_vector.dtype, device=tangent_vector.device
            )
            theta_nz = torch.where(near_zero, non_zero, theta)
            theta2_nz = torch.where(near_zero, non_zero, theta2)
            theta3_nz = theta_nz * theta2_nz
            sine = theta.sin()
            theta_minus_sine_by_theta3 = torch.where(
                near_zero, torch.zeros_like(theta), (theta - sine) / theta3_nz
            )
            cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
            sine = theta.sin()
            sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
            one_minus_cosie_by_theta2 = torch.where(
                near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
            )

            jac = (
                theta_minus_sine_by_theta3
                * tangent_vector.view(-1, 3, 1)
                @ tangent_vector.view(-1, 1, 3)
            )
            diag_jac = jac.diagonal(dim1=1, dim2=2)
            diag_jac += sine_by_theta.view(-1, 1)

            jac_temp = one_minus_cosie_by_theta2.view(-1, 1) * tangent_vector

            jac[:, 0, 1] += jac_temp[:, 2]
            jac[:, 1, 0] -= jac_temp[:, 2]
            jac[:, 0, 2] -= jac_temp[:, 1]
            jac[:, 2, 0] += jac_temp[:, 1]
            jac[:, 1, 2] += jac_temp[:, 0]
            jac[:, 2, 1] -= jac_temp[:, 0]

            return jac

        @staticmethod
        def forward(
            ctx,
            tangent_vector,
            jacobians=None,
        ):
            tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
            ret = SO3Function.exp_map.call(tangent_vector, jacobians)
            ctx.save_for_backward(tangent_vector, ret)
            ctx.jacobians = jacobians

            return ret

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.jacobians is None:
                tangent_vector: torch.Tensor = ctx.saved_tensors[0]
                ctx.jacobians = SO3Function.exp_map.jacobian(tangent_vector)

            R: torch.Tensor = ctx.saved_tensors[1]
            jacs: torch.Tensor = ctx.jacobians
            dR = R.transpose(1, 2) @ grad_output
            grad = jacs.transpose(1, 2) @ torch.stack(
                (
                    dR[:, 2, 1] - dR[:, 1, 2],
                    dR[:, 0, 2] - dR[:, 2, 0],
                    dR[:, 1, 0] - dR[:, 0, 1],
                ),
                dim=1,
            ).view(-1, 3, 1)
            return grad.view(-1, 3)

    class adjoint(LieGroupFunction.adjoint):
        @staticmethod
        def call(
            g: torch.Tensor,
        ) -> torch.Tensor:
            if not SO3Function.check_group_tensor(g):
                raise ValueError("Invalid data tensor for SO3.")
            return g

        @staticmethod
        def forward(ctx, g):
            g: torch.Tensor = cast(torch.Tensor, g)
            return SO3Function.adjoint.call(g)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    class inverse(LieGroupFunction.inverse):
        @staticmethod
        def call(
            g: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
        ) -> torch.Tensor:
            if not SO3Function.check_group_tensor(g):
                raise ValueError("Invalid data tensor for SO3.")
            if jacobians is not None:
                check_jacobians_list(jacobians)
                jacobians.append(-SO3Function.adjoint.call(g))
            return g.transpose(1, 2)

        @staticmethod
        def jacobian(g: torch.Tensor) -> torch.Tensor:
            if not SO3Function.check_group_tensor(g):
                raise ValueError("Invalid data tensor for SO3.")
            return -SO3Function.adjoint.call(g)

        @staticmethod
        def forward(ctx, g, jacobians=None):
            g: torch.Tensor = cast(torch.Tensor, g)
            return SO3Function.inverse.call(g, jacobians)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.transpose(1, 2)

    class compose(LieGroupFunction.compose):
        @staticmethod
        def call(
            g0: torch.Tensor,
            g1: torch.Tensor,
            jacobians: Optional[List[torch.Tensor]] = None,
        ) -> torch.Tensor:
            if not SO3Function.check_group_tensor(
                g0
            ) or not SO3Function.check_group_tensor(g1):
                raise ValueError("Invalid data tensor for SO3.")
            if jacobians is not None:
                check_jacobians_list(jacobians)
                jacobians.append(-SO3Function.inverse.jacobian(g1))
                jacobians.append(g1.new_zeros(g0.shape[0], 3, 3))
                jacobians[1][:, 0, 0] = 1
                jacobians[2][:, 1, 1] = 1
                jacobians[3][:, 2, 2] = 1
            return g0 @ g1

        @staticmethod
        def forward(ctx, g0, g1, jacobians=None):
            g0: torch.Tensor = cast(torch.Tensor, g0)
            g1: torch.Tensor = cast(torch.Tensor, g1)
            ctx.save_for_backward(g0, g1)
            return SO3Function.compose.call(g0, g1, jacobians)

        @staticmethod
        def backward(ctx, grad_output):
            g0 = ctx.saved_tensors[0]
            g1 = ctx.saved_tensors[1]
            return grad_output @ g1.transpose(1, 2), g0.transpose(1, 2) @ grad_output
