# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, cast

import torch

from theseus.core import CostFunction, CostWeight, Variable, as_variable
from theseus.geometry import SE2, OptionalJacobians
from theseus.geometry.so2 import SO2


# Based on the model described in
# Zhou et al. A Fast Stochastic Contact Model for Planar Pushing and Grasping:
# Theory and Experimental Validation, 2017
# https://arxiv.org/abs/1705.10664
class QuasiStaticPushingPlanar(CostFunction):
    def __init__(
        self,
        obj1: SE2,
        obj2: SE2,
        eff1: SE2,
        eff2: SE2,
        c_square: Union[Variable, torch.Tensor, float],
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        self.obj1 = obj1
        self.obj2 = obj2
        self.eff1 = eff1
        self.eff2 = eff2
        self.register_optim_vars(["obj1", "obj2", "eff1", "eff2"])

        c_square = as_variable(c_square, dtype=self.obj1.dtype, name=f"csquare_{name}")
        if c_square.tensor.squeeze().ndim > 1:
            raise ValueError("dt must be a 0-D or 1-D tensor.")
        self.c_square = c_square
        self.register_aux_vars(["c_square"])

    def _get_tensor_info(self) -> Tuple[int, torch.device, torch.dtype]:
        return self.obj1.shape[0], self.obj1.device, self.obj1.dtype

    # See Eqs. 3-7 in Zhou et al.
    def _compute_D(
        self,
        contact_point_2: torch.Tensor,
        J_cp2_e2: OptionalJacobians,
        get_jacobians: bool = False,
    ) -> Tuple[torch.Tensor, OptionalJacobians]:
        batch_size, device, dtype = self._get_tensor_info()

        # Setting up intermediate jacobians, if needed
        J_cp2o: OptionalJacobians = [] if get_jacobians else None

        # Current contact point in object frame
        contact_point_2__obj = self.obj2.transform_to(contact_point_2, jacobians=J_cp2o)
        px = contact_point_2__obj.tensor[:, 0]
        py = contact_point_2__obj.tensor[:, 1]

        # Putting D together
        D = (
            torch.eye(3, 3, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        D[:, 0, 2] = D[:, 2, 0] = -py
        D[:, 1, 2] = D[:, 2, 1] = px
        D[:, 2, 2] = -self.c_square.tensor.view(-1)

        if not get_jacobians:
            return D, None

        # Computing jacobians
        J_cp2o_o2, J_cp2o_cp2 = J_cp2o
        J_cp2o_e2 = J_cp2o_cp2.matmul(J_cp2_e2[0])

        def _get_J_D_var(J_cp2o_var_):
            # For matmul batching, this is set up with dimensions:
            #   batch_size x se2_dim x 3 x 3,
            # so that J_D_var_[b, d] is the (3, 3) matrix derivative
            # of the b-th batch D wrt to the d-th dimention of var
            J_D_var_ = torch.zeros(batch_size, 3, 3, 3, device=device, dtype=dtype)
            J_D_var_[..., 0, 2] = J_D_var_[..., 2, 0] = -J_cp2o_var_[:, 1]
            J_D_var_[..., 1, 2] = J_D_var_[..., 2, 1] = J_cp2o_var_[:, 0]
            return J_D_var_

        J_D_o2 = _get_J_D_var(J_cp2o_o2)
        J_D_e2 = _get_J_D_var(J_cp2o_e2)

        return D, [J_D_o2, J_D_e2]

    # See Eqs. 3-7 in Zhou et al.
    def _compute_V(
        self,
        obj2_angle: SO2,
        J_o2angle_o2: OptionalJacobians,
        get_jacobians: bool = False,
    ) -> Tuple[torch.Tensor, OptionalJacobians]:
        batch_size, device, dtype = self._get_tensor_info()

        # Setting up intermediate jacobians, if needed
        J_o2xy_o2: OptionalJacobians = [] if get_jacobians else None
        J_o1xy_o1: OptionalJacobians = [] if get_jacobians else None
        J_vxyoo: OptionalJacobians = [] if get_jacobians else None
        J_odw: OptionalJacobians = [] if get_jacobians else None
        J_omega_odw: OptionalJacobians = [] if get_jacobians else None

        # Compute object velocities using consecutive object poses
        vel_xy_obj__world = self.obj2.xy(jacobians=J_o2xy_o2) - self.obj1.xy(
            jacobians=J_o1xy_o1
        )
        vel_xy_obj__obj = obj2_angle.unrotate(
            vel_xy_obj__world.tensor, jacobians=J_vxyoo
        )

        # Putting V together
        obj_diff__world = cast(SE2, self.obj1.between(self.obj2, jacobians=J_odw))
        vx = vel_xy_obj__obj.tensor[:, 0]
        vy = vel_xy_obj__obj.tensor[:, 1]
        omega = obj_diff__world.theta(jacobians=J_omega_odw)
        V = torch.stack([vx, vy, omega.squeeze(1)], dim=1)

        if not get_jacobians:
            return V, None

        # Computing jacobians
        def _get_J_V_var(J_vxyoo_var_, J_omega_var_):
            # For matmul batching, this is set up with dimensions:
            #   batch_size x se2_dim x 3 x 1,
            # so that J_V_var_[b, d] is the (3, 1) vector derivative
            # of the b-th batch V wrt to the d-th dimention of var
            J_V_var_ = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
            J_V_var_[..., 0] = J_vxyoo_var_[:, 0]
            J_V_var_[..., 1] = J_vxyoo_var_[:, 1]
            J_V_var_[..., 2] = J_omega_var_[:, 0]
            return J_V_var_.unsqueeze(3)

        # Computing velocity derivates wrt to objects 1, 2 variables
        J_vxyow_o2 = J_o2xy_o2[0]
        J_vxyow_o1 = -J_o1xy_o1[0]
        J_vxyoo_o2angle, J_vxyoo_vxyow = J_vxyoo
        J_vxyoo_o1 = J_vxyoo_vxyow.matmul(J_vxyow_o1)
        J_vxyoo_o2 = J_vxyoo_o2angle.matmul(J_o2angle_o2[0]) + J_vxyoo_vxyow.matmul(
            J_vxyow_o2
        )

        J_odw_o1, J_odw_o2 = J_odw
        J_omega_o1 = J_omega_odw[0].matmul(J_odw_o1)
        J_omega_o2 = J_omega_odw[0].matmul(J_odw_o2)

        J_V_o1 = _get_J_V_var(J_vxyoo_o1, J_omega_o1)
        J_V_o2 = _get_J_V_var(J_vxyoo_o2, J_omega_o2)

        return V, [J_V_o1, J_V_o2]

    # See Eqs. 3-7 in Zhou et al.
    def _compute_Vp(
        self,
        obj2_angle: SO2,
        J_o2angle_o2: OptionalJacobians,
        contact_point_2: torch.Tensor,
        J_cp2_e2: OptionalJacobians,
        get_jacobians: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, device, dtype = self._get_tensor_info()

        # Setting up intermediate jacobians, if needed
        J_cp1_e1: OptionalJacobians = [] if get_jacobians else None
        J_vxyco: OptionalJacobians = [] if get_jacobians else None

        # Compute contact point velocities using consecutive end effector poses
        contact_point_1 = self.eff1.xy(jacobians=J_cp1_e1).tensor
        vel_xy_contact__world = contact_point_2 - contact_point_1

        # Transform velocity to object's 2 frame
        vel_xy_contact__obj = obj2_angle.unrotate(
            vel_xy_contact__world, jacobians=J_vxyco
        )

        # Putting Vp together
        v_px = vel_xy_contact__obj.tensor[:, 0]
        v_py = vel_xy_contact__obj.tensor[:, 1]
        Vp = torch.stack([v_px, v_py, torch.zeros_like(v_px)], dim=1)

        if not get_jacobians:
            return Vp, None

        # Computing jacobians
        def _get_J_Vp_var(J_vxyco_var_):
            # For matmul batching, this is set up with dimensions:
            #   batch_size x se2_dim x 3 x 1,
            # so that J_Vp_var_[b, d] is the (3, 1) vector derivative
            # of the b-th batch Vp wrt to the d-th dimention of var
            J_Vp_var_ = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
            J_Vp_var_[..., 0] = J_vxyco_var_[:, 0]  # batch_size x se2_dim
            J_Vp_var_[..., 1] = J_vxyco_var_[:, 1]  # batch_size x se2_dim
            return J_Vp_var_.unsqueeze(3)

        J_vxyco_o2angle, J_vxyco_vxycw = J_vxyco
        J_vxyco_o2 = J_vxyco_o2angle.matmul(J_o2angle_o2[0])
        J_vxyco_e1 = J_vxyco_vxycw.matmul(-J_cp1_e1[0])
        J_vxyco_e2 = J_vxyco_vxycw.matmul(J_cp2_e2[0])

        J_Vp_o2 = _get_J_Vp_var(J_vxyco_o2)
        J_Vp_e1 = _get_J_Vp_var(J_vxyco_e1)
        J_Vp_e2 = _get_J_Vp_var(J_vxyco_e2)

        return Vp, [J_Vp_o2, J_Vp_e1, J_Vp_e2]

    def _error_and_jacobians_impl(
        self, get_jacobians: bool
    ) -> Tuple[torch.Tensor, OptionalJacobians]:
        J_o2angle_o2: OptionalJacobians = [] if get_jacobians else None
        J_cp2_e2: OptionalJacobians = [] if get_jacobians else None

        # These quantities are needed by two or more of D, V, Vp, so we compute once.
        obj2_angle = self.obj2.rotation
        self.obj2.theta(jacobians=J_o2angle_o2)
        contact_point_2 = self.eff2.xy(jacobians=J_cp2_e2).tensor

        # D * V = Vp, See Zhou et al.
        # A Fast Stochastic Contact Model for Planar Pushing and Grasping:
        # Theory and Experimental Validation, 2017
        # https://arxiv.org/abs/1705.10664
        D, J_D = self._compute_D(contact_point_2, J_cp2_e2, get_jacobians=get_jacobians)
        V, J_V = self._compute_V(obj2_angle, J_o2angle_o2, get_jacobians=get_jacobians)
        Vp, J_Vp = self._compute_Vp(
            obj2_angle,
            J_o2angle_o2,
            contact_point_2,
            J_cp2_e2,
            get_jacobians=get_jacobians,
        )

        error = torch.bmm(D, V.unsqueeze(2)).squeeze(2) - Vp

        if not get_jacobians:
            return error, None

        # Computing jacobians
        V_exp = V.view(-1, 1, 3, 1)
        D_exp = D.view(-1, 1, 3, 3)
        sum_str = "bdij,bdjk->bdik"

        J_D_o2, J_D_e2 = J_D
        J_V_o1, J_V_o2 = J_V
        J_Vp_o2, J_Vp_e1, J_Vp_e2 = J_Vp

        def _get_J_err_var_(J_D_var_, J_V_var_, J_Vp_var_):
            # After einsum, each result is a tensor with size:
            #   batch_size x se2_dim x 3 x 1
            # so that after all terms are summed,
            # J_err_var_[b, d] is a (3, 1) vector derivative of
            # the b-th batch element of the 3-D error wrt to the
            # d-th dimension of the optimization variable.
            # At return, we squeeze the trailing dimension, and permute, so that the
            # output has shape (batch_size, 3, se2_dim), which is the expected jacobian
            # shape.
            J_err_var_ = torch.zeros_like(J_V_o1)
            if J_D_var_ is not None:
                J_err_var_ += torch.einsum(sum_str, J_D_var_, V_exp)
            if J_V_var_ is not None:
                J_err_var_ += torch.einsum(sum_str, D_exp, J_V_var_)
            if J_Vp_var_ is not None:
                J_err_var_ += -J_Vp_var_
            return J_err_var_.squeeze(3).permute(0, 2, 1)

        J_err_o1 = _get_J_err_var_(None, J_V_o1, None)
        J_err_o2 = _get_J_err_var_(J_D_o2, J_V_o2, J_Vp_o2)
        J_err_e1 = _get_J_err_var_(None, None, J_Vp_e1)
        J_err_e2 = _get_J_err_var_(J_D_e2, None, J_Vp_e2)

        return error, [J_err_o1, J_err_o2, J_err_e1, J_err_e2]

    def error(self) -> torch.Tensor:
        return self._error_and_jacobians_impl(get_jacobians=False)[0]

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        error, jacobians = self._error_and_jacobians_impl(get_jacobians=True)
        return jacobians, error

    def dim(self) -> int:
        return 3

    def _copy_impl(self, new_name: Optional[str] = None) -> "QuasiStaticPushingPlanar":
        return QuasiStaticPushingPlanar(
            self.obj1.copy(),
            self.obj2.copy(),
            self.eff1.copy(),
            self.eff2.copy(),
            self.c_square.copy(),
            self.weight.copy(),
            name=new_name,
        )
