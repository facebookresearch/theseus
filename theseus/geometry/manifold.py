# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import Any, List, Optional

import torch

from theseus.constants import _CHECK_DTYPE_SUPPORTED
from theseus.core.variable import Variable

from .lie_group_check import _LieGroupCheckContext

OptionalJacobians = Optional[List[torch.Tensor]]


# Abstract class to represent Manifold-type variables in the objective.
# Concrete classes must implement the following methods:
#   - `_init_tensor`: initializes the underlying tensor given constructor arguments.
#       The provided tensor must have a batch dimension.
#   -`_local`: given two close Manifolds gives distance in tangent space
#   - `_retract`: returns Manifold close by delta to given Manifold
#   - `update`: replaces the Manifold's tensor with the provided one (and checks that
#       the provided has the right format).
#   - `dof`: # of degrees of freedom of the variable
#
# Constructor can optionally provide an initial tensor value as a keyword argument.
class Manifold(Variable, abc.ABC):
    def __init__(
        self,
        *args: Any,
        tensor: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        strict: bool = False,
    ):
        # If nothing specified, use torch's default dtype
        # else tensor.dtype takes precedence
        if tensor is None and dtype is None:
            dtype = torch.get_default_dtype()
        if tensor is not None:
            checks_enabled, silent_unchecks = _LieGroupCheckContext.get_context()
            if checks_enabled:
                tensor = self._check_tensor(tensor, strict)
            elif not silent_unchecks:
                warnings.warn(
                    f"Manifold consistency checks are disabled "
                    f"for {self.__class__.__name__}.",
                    RuntimeWarning,
                )
            if dtype is not None and tensor.dtype != dtype:
                warnings.warn(
                    f"tensor.dtype {tensor.dtype} does not match given dtype {dtype}, "
                    "tensor.dtype will take precendence."
                )
            dtype = tensor.dtype

        _CHECK_DTYPE_SUPPORTED(dtype)
        super().__init__(self.__class__._init_tensor(*args).to(dtype=dtype), name=name)
        if tensor is not None:
            self.update(tensor)

    # This method should return a tensor with the manifold's representation
    # as a function of the given args
    @staticmethod
    @abc.abstractmethod
    def _init_tensor(*args: Any) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def dof(self) -> int:
        pass

    def numel(self) -> int:
        return self.tensor[0].numel()

    @abc.abstractmethod
    def _local_impl(
        self, variable2: "Manifold", jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _retract_impl(self, delta: torch.Tensor) -> "Manifold":
        pass

    @abc.abstractmethod
    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        pass

    def project(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        return self._project_impl(euclidean_grad, is_sparse)

    @staticmethod
    @abc.abstractmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        pass

    @classmethod
    def _check_tensor(cls, tensor: torch.Tensor, strict: bool = True) -> torch.Tensor:
        check = cls._check_tensor_impl(tensor)

        if not check:
            if strict:
                raise ValueError(f"The input tensor is not valid for {cls.__name__}.")
            else:
                tensor = cls.normalize(tensor)
                warnings.warn(
                    f"The input tensor is not valid for {cls.__name__} "
                    f"and has been normalized."
                )

        return tensor

    def local(
        self,
        variable2: "Manifold",
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        local_out = self._local_impl(variable2, jacobians)
        return local_out

    def retract(self, delta: torch.Tensor) -> "Manifold":
        return self._retract_impl(delta)

    # Must copy everything
    @abc.abstractmethod
    def _copy_impl(self, new_name: Optional[str] = None) -> "Manifold":
        pass

    def copy(self, new_name: Optional[str] = None) -> "Manifold":
        if not new_name:
            new_name = f"{self.name}_copy"
        var_copy = self._copy_impl(new_name=new_name)
        return var_copy

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        the_copy = self.copy()
        memo[id(self)] = the_copy
        return the_copy

    # calls to() on the internal tensors
    def to(self, *args, **kwargs):
        _, dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            _CHECK_DTYPE_SUPPORTED(dtype)
        super().to(*args, **kwargs)


# Alias for Manifold.local()
def local(
    variable1: Manifold,
    variable2: Manifold,
    jacobians: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    return variable1.local(variable2, jacobians=jacobians)


# Alias for Manifold.retract()
def retract(variable: Manifold, delta: torch.Tensor) -> Manifold:
    return variable.retract(delta)
