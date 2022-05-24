import abc
import warnings
from typing import Any, List, Optional

import torch

from theseus.core.variable import Variable

OptionalJacobians = Optional[List[torch.Tensor]]


# Abstract class to represent Manifold-type variables in the objective.
# Concrete classes must implement the following methods:
#   - `_init_data`: initializes the underlying data tensor given constructor arguments.
#       The provided tensor must have a batch dimension.
#   -`_local`: given two close Manifolds gives distance in tangent space
#   - `_retract`: returns Manifold close by delta to given Manifold
#   - `update`: replaces the data tensor with the provided one (and checks that
#       the provided has the right format).
#   - `dof`: # of degrees of freedom of the variable
#
# Constructor can optionally provide an initial data value as a keyword argument.
class Manifold(Variable, abc.ABC):
    def __init__(
        self,
        *args: Any,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # If nothing specified, use torch's default dtype
        # else data.dtype takes precedence
        if data is None and dtype is None:
            dtype = torch.get_default_dtype()
        if data is not None:
            if dtype is not None and data.dtype != dtype:
                warnings.warn(
                    f"data.dtype {data.dtype} does not match given dtype {dtype}, "
                    "data.dtype will take precendence."
                )
            dtype = data.dtype

        super().__init__(self.__class__._init_data(*args).to(dtype=dtype), name=name)
        if data is not None:
            self.update(data)

    # This method should return a tensor with the manifold's data representation
    # as a function of the given args
    @staticmethod
    @abc.abstractmethod
    def _init_data(*args: Any) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def dof(self) -> int:
        pass

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
