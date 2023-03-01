# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import builtins
import threading
import warnings
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union
from typing import cast as type_cast

import torch
from torch.utils._pytree import tree_flatten, tree_map_only
from torch.types import Number

from theseus.labs.lie.functional.constants import DeviceType
from theseus.labs.lie.functional.lie_group import LieGroupFns, UnaryOperatorOpFnType
from theseus.labs.lie.functional import se3 as _se3_impl, so3 as _so3_impl
from .types import ltype as _ltype, SE3, SO3

Device = Union[torch.device, str, builtins.int, None]
TensorType = Union[torch.Tensor, "_LieTensorBase"]
_JFnReturnType = Tuple[List[torch.Tensor], TensorType]


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


def _get_fn_lib(ltype: _ltype):
    return {
        SE3: _se3_impl,
        SO3: _so3_impl,
    }[ltype]


class _LieTensorBase(torch.Tensor):
    def __new__(cls, data: torch.Tensor, ltype: _ltype, requires_grad=False):
        return cls._make_subclass(cls, data, requires_grad)  # type: ignore

    @property
    def _t(self) -> torch.Tensor:
        return self.as_subclass(torch.Tensor)

    def __init__(self, data: Any, ltype: _ltype, requires_grad=None):
        self.ltype = ltype

    def __repr__(self) -> str:  # type: ignore
        return f"LieTensor({self._t}, ltype=lie.{self.ltype})"

    @classmethod
    def to_torch(cls, args: Any):
        return tree_map_only(cls, lambda x: x._t, args)

    @classmethod
    def resolve_ltype(cls, t: Any):
        return t.ltype if isinstance(t, cls) else None

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
        return tree_map_only(torch.Tensor, lambda x: cls(ret, ltype), ret)


class _TangentTensor(_LieTensorBase):
    def __init__(self, data: Any, ltype: _ltype, requires_grad=None):
        super().__init__(data, _ltype.tgt, requires_grad=requires_grad)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return cls._torch_func_impl_eucl(func, types, args, kwargs)


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


def _eval_op(
    fn_lib: LieGroupFns,
    op_name: str,
    input0: torch.Tensor,
    jacobians: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    return getattr(fn_lib, op_name)(input0, jacobians=jacobians)


def _LIE_TENSOR_ERROR_MSG(func):
    return (
        f"LieTensor.{func.__name__} is only supported for "
        "tensors resulting from euclidean gradient computations."
    )


class LieTensor(_LieTensorBase):
    # These are operations where calling super().__torch_function__() on the
    # LieTensor itself is safe
    _SAFE_SUPER_OPS: List[Callable] = [
        torch.Tensor.shape.__get__,  # type: ignore
        torch.Tensor.is_leaf.__get__,  # type: ignore
        torch.Tensor.requires_grad.__get__,  # type: ignore
        torch.Tensor.retains_grad.__get__,  # type: ignore
        torch.Tensor.grad.__get__,  # type: ignore
        torch.Tensor.clone,
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
    ]

    def __init__(self, tensor: torch.Tensor, ltype: _ltype, requires_grad=None):
        super().__init__(tensor, ltype, requires_grad=requires_grad)
        self._fn_lib = _get_fn_lib(self.ltype)
        self._fn_lib.check_group_tensor(self.as_subclass(torch.Tensor))

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
            return tree_map_only(LieTensor, lambda x: LieTensor(ret, ltype), ret)
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
    def new(self, args: Any, device: Device = None) -> "LieTensor":  # type: ignore
        if isinstance(args, LieTensor):
            warnings.warn(
                "Calling new() on a LieTensor results in shared data storage. "
                "To copy construct from a LieTensor, it is recommended to use "
                "lie_tensor.clone().",
                UserWarning,
            )
            return LieTensor(args._t, ltype=self.ltype)
        if isinstance(args, torch.Tensor):
            return LieTensor.from_tensor(args, self.ltype)
        return LieTensor(torch.as_tensor(args, device=device), ltype=self.ltype)

    # ------------------------------------------------------
    # ------ Operators
    # ------------------------------------------------------
    # The following could be static methods, because self is used
    # only to infer the type. But for each of this, there are also
    # versions such as:
    #   - `lie.exp(ltype, tangent_vector)`
    #   - `lie.lift(ltype, matrix)`
    # that we expect to be more commonly used.
    def exp(self, tangent_vector: torch.Tensor) -> "LieTensor":  # type: ignore
        return self.new(_eval_op(self._fn_lib, "exp", tangent_vector))

    def hat(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        return _eval_op(self._fn_lib, "hat", tangent_vector)

    def vee(self, matrix: torch.Tensor) -> torch.Tensor:
        return _eval_op(self._fn_lib, "vee", matrix)

    def lift(self, matrix: torch.Tensor) -> torch.Tensor:
        return _eval_op(self._fn_lib, "lift", matrix)

    def project(self, matrix: torch.Tensor) -> torch.Tensor:
        return _eval_op(self._fn_lib, "project", matrix)

    # For the next ones the output also depends on self's data
    def log(self) -> torch.Tensor:
        return self._fn_lib.log(self._t)

    def adj(self) -> torch.Tensor:
        return self._fn_lib.adj(self._t)

    def inv(self) -> "LieTensor":
        return self.new(self._fn_lib.inv(self._t))

    def compose(self, other: "LieTensor") -> "LieTensor":
        self._check_ltype(other, "compose")
        return self.new(self._fn_lib.compose(self._t, other._t))

    def left_act(self, matrix: torch.Tensor) -> torch.Tensor:
        return self._fn_lib.left_act(self._t, matrix)

    def left_project(self, matrix: torch.Tensor) -> torch.Tensor:
        return self._fn_lib.left_project(self._t, matrix)

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
            op_res = self.new(op_res)
        return jacs, op_res

    def jexp(self, tangent_vector: torch.Tensor) -> _JFnReturnType:
        return self._unary_jop_base(tangent_vector, self._fn_lib.exp)

    def jlog(self) -> _JFnReturnType:
        return self._unary_jop_base(self._t, self._fn_lib.log, out_is_group=False)

    def jinv(self) -> _JFnReturnType:
        return self._unary_jop_base(self._t, self._fn_lib.inv)

    def jcompose(self, other: "LieTensor") -> _JFnReturnType:
        self._check_ltype(other, "jcompose")
        jacs: List[torch.Tensor] = []
        op_res = self.new(self._fn_lib.compose(self._t, other._t, jacobians=jacs))
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
        assert isinstance(delta, torch.Tensor)
        return self.compose(self.exp(delta))

    # ------------------------------------------------------
    # Overloaded python and torch operators
    # ------------------------------------------------------
    def __add__(self, other: TensorType) -> "LieTensor":
        if not isinstance(other, _TangentTensor):
            raise RuntimeError(
                "Operator + is only supported for tensors of ltype=tgt. "
                "If you intend to add the raw tensor data, please use group._t. "
                "If you intend to retract, then cast your tensor to ltype=tgt, or "
                "use group.retract(tensor)."
            )
        return self.retract(other._t)

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
            grad = self._fn_lib.left_project(self._t, tensor)
            res = self.retract(grad * alpha)
            self.set_(res)
        else:
            raise RuntimeError(_LIE_TENSOR_ERROR_MSG(self.add_))

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
            raise RuntimeError(_LIE_TENSOR_ERROR_MSG(self.addcdiv_))

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
            raise RuntimeError(_LIE_TENSOR_ERROR_MSG(self.addcmul_))


# ----------------------------
# Tensor creation functions
# ----------------------------
def as_lietensor(
    data: Any,
    ltype: Optional[_ltype] = None,
    dtype: torch.dtype = torch.float,
    device: Union[torch.device, str] = None,
) -> _LieTensorBase:
    if isinstance(data, LieTensor):
        return data
    if ltype is None:
        raise ValueError("ltype must be provided.")
    if ltype == _ltype.tgt:
        return _TangentTensor(data, None)
    return type_cast(
        LieTensor, LieTensor(data, ltype=ltype).to(device=device, dtype=dtype)
    )


# With this new version of the code I like @fantaosha's proposal of using the
# name cast() because it implies type conversion rather than a wrapper being created.
cast = as_lietensor

from_tensor = LieTensor.from_tensor


class _FromTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, ltype: _ltype) -> LieTensor:  # type: ignore
        return LieTensor(tensor, ltype, requires_grad=tensor.requires_grad)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore
        return grad_output, None


# Similar to the one in functional, except it returns a LieTensor
# and receives a ltype
class _RandFnType(Protocol):
    def __call__(
        self,
        *args: Any,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> LieTensor:
        pass


def _build_random_fn(op_name: str) -> _RandFnType:
    assert op_name in ["rand", "randn"]

    def fn(
        *args: Any,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> LieTensor:
        good = all([isinstance(a, int) for a in args[:-1]]) and isinstance(
            args[-1], _ltype
        )
        if not good:
            arg_types = " ".join(type(a).__name__ for a in args)
            raise TypeError(
                f"{op_name}() received invalid combination of arguments - "
                f"got ({arg_types}), but expected (tuple of ints size, ltype)."
            )
        size, ltype = args[:-1], args[-1]
        fn = {SE3: getattr(_se3_impl, op_name), SO3: getattr(_so3_impl, op_name)}[ltype]
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
        )

    return fn


rand: _RandFnType = _build_random_fn("rand")
randn: _RandFnType = _build_random_fn("randn")


def log(group: LieTensor) -> torch.Tensor:
    return group.log()


def adj(group: LieTensor) -> torch.Tensor:
    return group.adj()


def inv(group: LieTensor) -> LieTensor:
    return group.inv()


def exp(tangent_vector: torch.Tensor, ltype: _ltype) -> "LieTensor":
    return LieTensor(_eval_op(_get_fn_lib(ltype), "exp", tangent_vector), ltype)


def hat(tangent_vector: torch.Tensor, ltype: _ltype) -> torch.Tensor:
    return _eval_op(_get_fn_lib(ltype), "hat", tangent_vector)


def vee(matrix: torch.Tensor, ltype: _ltype) -> torch.Tensor:
    return _eval_op(_get_fn_lib(ltype), "vee", matrix)


def lift(matrix: torch.Tensor, ltype: _ltype) -> torch.Tensor:
    return _eval_op(_get_fn_lib(ltype), "lift", matrix)


def project(matrix: torch.Tensor, ltype: _ltype) -> torch.Tensor:
    return _eval_op(_get_fn_lib(ltype), "project", matrix)


def compose(group1: LieTensor, group2: LieTensor) -> LieTensor:
    return group1.compose(group2)


def left_act(group: LieTensor, matrix: torch.Tensor) -> torch.Tensor:
    return group.left_act(matrix)


def left_project(group: LieTensor, matrix: torch.Tensor) -> torch.Tensor:
    return group.left_project(matrix)


def jlog(group: LieTensor) -> _JFnReturnType:
    return group.jlog()


def jinv(group: LieTensor) -> _JFnReturnType:
    return group.jinv()


def jexp(tangent_vector: torch.Tensor, ltype: _ltype) -> _JFnReturnType:
    jacs: List[torch.Tensor] = []
    exp_tensor = _eval_op(_get_fn_lib(ltype), "exp", tangent_vector, jacobians=jacs)
    return jacs, LieTensor(exp_tensor, ltype)


def jcompose(group1: LieTensor, group2: LieTensor) -> _JFnReturnType:
    return group1.jcompose(group2)


def retract(group: LieTensor, delta: TensorType) -> LieTensor:
    return group.retract(delta)
