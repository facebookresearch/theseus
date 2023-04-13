# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import threading
import warnings
from typing import Any, Callable, List, Optional, Tuple
from typing import cast as type_cast

import torch
from torch.utils._pytree import tree_flatten, tree_map_only
from torch.types import Number

from theseus.labs.lie.functional.constants import DeviceType
from theseus.labs.lie.functional.lie_group import UnaryOperatorOpFnType
from .types import (
    ltype as _ltype,
    Device,
    SE3,
    SO3,
    TensorType,
    _IdentityFnType,
    _JFnReturnType,
    _RandFnType,
)


class _LieAsEuclideanContext:
    contexts = threading.local()

    @classmethod
    def get_context(cls):
        if not hasattr(cls.contexts, "as_eucledian"):
            cls.contexts.as_eucledian = False
        return cls.contexts.as_eucledian

    @classmethod
    def set_context(cls, as_eucledian: bool):
        cls.contexts.as_eucledian = as_eucledian


class as_euclidean:
    def __init__(self) -> None:
        self.prev = _LieAsEuclideanContext.get_context()
        _LieAsEuclideanContext.set_context(True)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _LieAsEuclideanContext.set_context(self.prev)


def euclidean_enabled() -> bool:
    return _LieAsEuclideanContext.get_context()


class _LieTensorBase(torch.Tensor):
    def __new__(
        cls,
        data: torch.Tensor,
        ltype: _ltype,
        requires_grad: bool = False,
        _shared_memory: bool = False,
    ):
        return cls._make_subclass(
            cls, data if _shared_memory else data.clone(), requires_grad  # type: ignore
        )

    @property
    def _t(self) -> torch.Tensor:
        return self.as_subclass(torch.Tensor)

    def __init__(
        self,
        data: Any,
        ltype: _ltype,
        requires_grad: Optional[bool] = None,
        _shared_memory: bool = False,
    ):
        self.ltype = ltype
        self.ltype._fn_lib.check_group_tensor(self.as_subclass(torch.Tensor))

    def __repr__(self) -> str:  # type: ignore
        return f"LieTensor({self._t}, ltype=lie.{self.ltype})"

    @classmethod
    def to_torch(cls, args: Any):
        return tree_map_only(cls, lambda x: x._t, args)

    @staticmethod
    def resolve_ltype(t: Any):
        return t.ltype if hasattr(t, "ltype") else None

    @classmethod
    def get_ltype(cls, args):
        ltypes = [cls.resolve_ltype(a) for a in tree_flatten(args)[0]]
        ltypes = set(x for x in ltypes if x is not None)
        if len(ltypes) > 1:
            raise ValueError(
                f"All LieTensors must be of the same ltype. " f"But found {ltypes}"
            )
        return next(iter(ltypes)) if len(ltypes) > 0 else None

    @classmethod
    def _torch_func_impl_eucl(cls, func, types, args=(), kwargs=None, raw_tensor=False):
        kwargs = kwargs or {}
        torch_args = cls.to_torch(args)
        torch_kwargs = cls.to_torch(kwargs)
        ltype = cls.get_ltype(args)
        ret = func(*torch_args, **torch_kwargs)
        if raw_tensor:
            return ret
        return tree_map_only(torch.Tensor, lambda x: cls(x, ltype), ret)


_EUCLID_GRAD_MARKER = "_lie_euclidean_grad"


# Lightweight wrapper so that LieTensor.add_ knows that this is a gradient
# that should be left projected and retracted
class _EuclideanGrad(torch.Tensor):
    _lie_euclidean_grad = True

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # Need to do this marking thing because torch optimizers use
        # in-place operations, so in many cases the descent direction won't be
        # an object of type _EuclideanGrad. With this, anything that touches
        # an _EuclideanGrad becomes marked and can be used to update LieTensors.
        def _mark(x):
            if isinstance(x, torch.Tensor):
                setattr(x, _EUCLID_GRAD_MARKER, True)

        for a in tree_flatten(args)[0]:
            _mark(a)

        return super().__torch_function__(func, types, args, kwargs or {})


def _LIE_TENSOR_GRAD_ERROR_MSG(func):
    return (
        f"LieTensor.{func.__name__} is only supported for "
        "tensors resulting from euclidean gradient computations."
    )


class LieTensor(_LieTensorBase):
    # These are operations where calling super().__torch_function__() on the
    # LieTensor itself is safe
    _SAFE_SUPER_OPS: List[Callable] = [
        torch.Tensor.device.__get__,  # type: ignore
        torch.Tensor.dtype.__get__,  # type: ignore
        torch.Tensor.is_leaf.__get__,  # type: ignore
        torch.Tensor.is_mps.__get__,  # type: ignore
        torch.Tensor.is_sparse.__get__,  # type: ignore
        torch.Tensor.is_quantized.__get__,  # type: ignore
        torch.Tensor.grad.__get__,  # type: ignore
        torch.Tensor.layout.__get__,  # type: ignore
        torch.Tensor.requires_grad.__get__,  # type: ignore
        torch.Tensor.retains_grad.__get__,  # type: ignore
        torch.Tensor.storage,  # type: ignore
        torch.Tensor.__getitem__,  # type: ignore
        torch.Tensor.clone,
        torch.Tensor.__format__,
        torch.Tensor.new_tensor,
        torch.Tensor.to,
        torch.cat,
        torch.is_complex,
        # torch.stack  # requires arbitrary batch support
    ]

    # These are operations where calling super().__torch_function__() on the
    # lie_tensor._t view is safe, and that require a return value of
    # raw torch.Tensor, because their return values don't make sense for LieTensors.
    _SAFE_AS_EUCL_OPS: List[Callable] = [
        torch.zeros_like,
        torch.ones_like,
        torch.Tensor.new_zeros,
        torch.Tensor.new_ones,
        torch.allclose,
        torch.isclose,
        torch.Tensor.grad_fn.__get__,  # type: ignore
        torch.Tensor.shape.__get__,  # type: ignore
        torch.Tensor.ndim.__get__,  # type: ignore
    ]

    def __init__(
        self,
        tensor: torch.Tensor,
        ltype: _ltype,
        requires_grad: Optional[bool] = None,
        _shared_memory: bool = False,
    ):
        super().__init__(
            tensor,
            ltype,
            requires_grad=requires_grad,
            _shared_memory=_shared_memory,
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor, ltype: _ltype) -> "LieTensor":
        return _FromTensor.apply(tensor, ltype)

    @classmethod
    def _torch_function_impl_lie(cls, func, types, args=(), kwargs=None):
        if func in LieTensor._SAFE_AS_EUCL_OPS:
            return super()._torch_func_impl_eucl(
                func, types, args, kwargs, raw_tensor=True
            )
        if func in LieTensor._SAFE_SUPER_OPS:
            ltype = cls.get_ltype(args)
            ret = super().__torch_function__(func, types, args, kwargs or {})
            if func == torch.Tensor.grad.__get__:
                return _EuclideanGrad(ret) if ret is not None else ret
            # tree-map to set the ltypes correctly
            if func == torch.Tensor.__getitem__:
                if isinstance(args[1], tuple):
                    raise NotImplementedError(
                        "LieTensor currently only supports slicing the batch dimension."
                    )
                assert ret.ndim in [2, 3]
                if ret.ndim == 2:
                    ret = ret._t.unsqueeze(0)
            return tree_map_only(torch.Tensor, lambda x: from_tensor(x, ltype), ret)
        raise NotImplementedError(
            "Tried to call a torch function not supported by LieTensor. "
            "If trying to operate on the raw tensor data, please use group._t, "
            "or run inside the context lie.as_euclidean()."
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if euclidean_enabled():
            return super()._torch_func_impl_eucl(
                func, types, args, kwargs, raw_tensor=True
            )
        else:
            return cls._torch_function_impl_lie(func, types, args, kwargs)

    def _check_ltype(self, other: "LieTensor", op_name: str):
        if other.ltype != self.ltype:
            raise ValueError("f{op_name} requires both tensors to have same ltype.")

    # Returns a new LieTensor with the given data and the same ltype as self
    def new_lietensor(  # type: ignore
        self, args: Any, device: Device = None, dtype: Optional[torch.dtype] = None
    ) -> "LieTensor":
        if isinstance(args, LieTensor):
            warnings.warn(
                "To copy construct from a LieTensor, it is recommended to use "
                "lie_tensor.clone().detach() instead of lie_tensor.new_lietensor(t)",
                UserWarning,
            )
            tensor_arg = args._t
            dtype = dtype or self.dtype
        elif isinstance(args, torch.Tensor):
            tensor_arg = args
            dtype = dtype or args.dtype
        else:
            tensor_arg = torch.as_tensor(args)
            dtype = dtype or torch.get_default_dtype()
        return LieTensor(tensor_arg.to(device=device, dtype=dtype), ltype=self.ltype)

    # ------------------------------------------------------
    # ------ Operators
    # ------------------------------------------------------
    # For the next ones the output also depends on self's data
    def log(self) -> torch.Tensor:
        return self.ltype._fn_lib.log(self._t)

    def adj(self) -> torch.Tensor:
        return self.ltype._fn_lib.adj(self._t)

    def inv(self) -> "LieTensor":
        return self.from_tensor(self.ltype._fn_lib.inv(self._t), self.ltype)

    def compose(self, other: "LieTensor") -> "LieTensor":
        self._check_ltype(other, "compose")
        return self.from_tensor(
            self.ltype._fn_lib.compose(self._t, other._t), self.ltype
        )

    def transform_from(self, point: torch.Tensor) -> torch.Tensor:
        return self.ltype._fn_lib.transform_from(self._t, point)

    def left_act(self, matrix: torch.Tensor) -> torch.Tensor:
        return self.ltype._fn_lib.left_act(self._t, matrix)

    def left_project(self, matrix: torch.Tensor) -> torch.Tensor:
        return self.ltype._fn_lib.left_project(self._t, matrix)

    # ------------------------------------------------------
    # Operator Jacobians
    # ------------------------------------------------------
    def _unary_jop_base(
        self,
        input0: torch.Tensor,
        fn: UnaryOperatorOpFnType,
        out_is_group: bool = True,
    ) -> _JFnReturnType:
        jacs: List[torch.Tensor] = []
        op_res: TensorType = fn(input0, jacobians=jacs)
        if out_is_group:
            op_res = self.from_tensor(op_res, self.ltype)
        return jacs, op_res

    def jlog(self) -> _JFnReturnType:
        return self._unary_jop_base(self._t, self.ltype._fn_lib.log, out_is_group=False)

    def jinv(self) -> _JFnReturnType:
        return self._unary_jop_base(self._t, self.ltype._fn_lib.inv)

    def jcompose(self, other: "LieTensor") -> _JFnReturnType:
        self._check_ltype(other, "jcompose")
        jacs: List[torch.Tensor] = []
        op_res = self.from_tensor(
            self.ltype._fn_lib.compose(self._t, other._t, jacobians=jacs), self.ltype
        )
        return jacs, op_res

    def jtransform_from(self, point: torch.Tensor) -> _JFnReturnType:
        jacs: List[torch.Tensor] = []
        op_res = self.ltype._fn_lib.transform_from(self._t, point, jacobians=jacs)
        return jacs, op_res

    def _no_jop(self, input0: TensorType) -> _JFnReturnType:
        raise NotImplementedError

    jadjoint = _no_jop
    jhat = _no_jop
    jvee = _no_jop
    jlift = _no_jop
    jproject = _no_jop
    jleft_act = _no_jop
    jleft_project = _no_jop

    def retract(self, delta: TensorType) -> "LieTensor":
        if not isinstance(delta, torch.Tensor):
            raise TypeError(
                "LieTensor.retract() expects a single torch.Tensor argument."
            )
        return self.compose(self.ltype.exp(delta))

    def local(self, other: "LieTensor") -> torch.Tensor:
        if not isinstance(other, LieTensor):
            raise TypeError(
                f"Incorrect argument for LieTensor.local(). "
                f"Expected LieTensor, but got {type(other)}."
            )
        if not other.ltype == self.ltype:
            raise ValueError(
                f"Incorrect ltype for local. Expected {self.ltype}, "
                f"received {other.ltype}."
            )
        return self.inv().compose(other).log()

    # ------------------------------------------------------
    # Overloaded python and torch operators
    # ------------------------------------------------------
    def __mul__(self, other: TensorType) -> "LieTensor":
        if not isinstance(other, LieTensor):
            raise TypeError(
                f"Incorrect argument for '*' operator. "
                f"Expected LieTensor, but got {type(other)}"
            )
        return type_cast(LieTensor, other).compose(self)

    def __matmul__(self, point: TensorType) -> torch.Tensor:
        if isinstance(point, LieTensor):
            raise TypeError(
                "Incorrect argument for '@' operator. "
                "Expected a torch.Tensor (x, y, z), but got a LieTensor."
            )
        return self.transform_from(point)

    def set_(self, tensor: "LieTensor"):  # type: ignore
        if not isinstance(tensor, LieTensor):
            raise RuntimeError(
                "LieTensor.set_ is only supported for LieTensor arguments."
            )
        if not tensor.ltype == self.ltype:
            raise RuntimeError(
                f"Tried to set a tensor of type {self.ltype} with a "
                f"tensor of type {tensor.ltype}"
            )
        super().set_(tensor)  # type: ignore

    def add_(self, tensor: torch.Tensor, *, alpha: Number = 1.0):  # type: ignore
        if hasattr(tensor, _EUCLID_GRAD_MARKER):
            grad = self.ltype._fn_lib.left_project(self._t, tensor)
            res = self.retract(grad * alpha)
            self.set_(res)
        else:
            raise RuntimeError(_LIE_TENSOR_GRAD_ERROR_MSG(self.add_))

    def addcdiv_(
        self, tensor1: torch.Tensor, tensor2: torch.Tensor, value: Number = 1.0
    ):
        can_do = False
        for t in [tensor1, tensor2]:
            if hasattr(t, _EUCLID_GRAD_MARKER):
                can_do = True
        if can_do:
            self.add_(_EuclideanGrad(value * tensor1 / tensor2))
        else:
            raise RuntimeError(_LIE_TENSOR_GRAD_ERROR_MSG(self.addcdiv_))

    def addcmul_(
        self, tensor1: torch.Tensor, tensor2: torch.Tensor, value: Number = 1.0
    ):
        can_do = False
        for t in [tensor1, tensor2]:
            if hasattr(t, _EUCLID_GRAD_MARKER):
                can_do = True
        if can_do:
            self.add_(_EuclideanGrad(value * tensor1 * tensor2))
        else:
            raise RuntimeError(_LIE_TENSOR_GRAD_ERROR_MSG(self.addcmul_))


# ----------------------------
# Tensor creation functions
# ----------------------------
def as_lietensor(
    data: Any,
    ltype: Optional[_ltype] = None,
    dtype: torch.dtype = torch.float,
    device: DeviceType = None,
) -> _LieTensorBase:
    if isinstance(data, LieTensor):
        if data.dtype == dtype and data.device == torch.device(device or "cpu"):
            return data
        return type_cast(LieTensor, data.to(device=device, dtype=dtype))
    if ltype is None:
        raise ValueError("ltype must be provided.")
    return type_cast(
        LieTensor,
        LieTensor.from_tensor(data, ltype=ltype).to(device=device, dtype=dtype),
    )


def cast(
    data: Any,
    ltype: Optional[_ltype] = None,
    dtype: torch.dtype = torch.float,
    device: DeviceType = None,
) -> _LieTensorBase:
    return as_lietensor(data, ltype=ltype, dtype=dtype, device=device)


from_tensor: Callable[[torch.Tensor, _ltype], LieTensor] = LieTensor.from_tensor


class _FromTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, ltype: _ltype) -> LieTensor:  # type: ignore
        return LieTensor(
            tensor, ltype, requires_grad=tensor.requires_grad, _shared_memory=True
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore
        return grad_output, None


def _build_random_fn(op_name: str, ltype: _ltype) -> _RandFnType:
    assert op_name in ["rand", "randn"]

    def fn(
        *size: Any,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> LieTensor:
        good = all([isinstance(a, int) for a in size])
        if not good:
            arg_types = " ".join(type(a).__name__ for a in size)
            raise TypeError(
                f"{op_name}() received invalid combination of arguments - "
                f"got ({arg_types}), but expected (tuple of ints)."
            )
        fn = getattr(ltype._fn_lib, op_name)
        return LieTensor(
            fn(
                *size,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            ),
            ltype,
            requires_grad=requires_grad,
            _shared_memory=True,
        )

    return fn


def _build_identity_fn(ltype: _ltype) -> _IdentityFnType:
    def fn(
        *size: Any,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> LieTensor:
        good = all([isinstance(a, int) for a in size])
        if not good:
            arg_types = " ".join(type(a).__name__ for a in size)
            raise TypeError(
                f"identity() received invalid combination of arguments - "
                f"got ({arg_types}), but expected (tuple of ints)."
            )
        return LieTensor(
            ltype._fn_lib.identity(
                *size,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            ),
            ltype,
            requires_grad=requires_grad,
            _shared_memory=True,
        )

    return fn


def _build_call_impl(ltype: _ltype) -> Callable[[torch.Tensor], LieTensor]:
    def fn(tensor: torch.Tensor) -> LieTensor:
        return LieTensor(tensor, ltype)

    return fn


SE3.rand = _build_random_fn("rand", SE3)
SE3.randn = _build_random_fn("randn", SE3)
SE3.identity = _build_identity_fn(SE3)
SE3._call_impl = _build_call_impl(SE3)
SO3.rand = _build_random_fn("rand", SO3)
SO3.randn = _build_random_fn("randn", SO3)
SO3.identity = _build_identity_fn(SO3)
SO3._call_impl = _build_call_impl(SO3)
SE3._create_lie_tensor = SO3._create_lie_tensor = LieTensor


def log(group: LieTensor) -> torch.Tensor:
    return group.log()


def adj(group: LieTensor) -> torch.Tensor:
    return group.adj()


def inv(group: LieTensor) -> LieTensor:
    return group.inv()


def compose(group1: LieTensor, group2: LieTensor) -> LieTensor:
    return group1.compose(group2)


def transform_from(group1: LieTensor, tensor: torch.Tensor) -> torch.Tensor:
    return group1.transform_from(tensor)


def left_act(group: LieTensor, matrix: torch.Tensor) -> torch.Tensor:
    return group.left_act(matrix)


def left_project(group: LieTensor, matrix: torch.Tensor) -> torch.Tensor:
    return group.left_project(matrix)


def jlog(group: LieTensor) -> _JFnReturnType:
    return group.jlog()


def jinv(group: LieTensor) -> _JFnReturnType:
    return group.jinv()


def jcompose(group1: LieTensor, group2: LieTensor) -> _JFnReturnType:
    return group1.jcompose(group2)


def jtransform_from(group1: LieTensor, tensor: torch.Tensor) -> _JFnReturnType:
    return group1.jtransform_from(tensor)


def retract(group: LieTensor, delta: TensorType) -> LieTensor:
    return group.retract(delta)


def local(group1: LieTensor, group2: LieTensor) -> torch.Tensor:
    return group1.local(group2)
