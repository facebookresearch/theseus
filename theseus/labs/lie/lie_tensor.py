# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import builtins
import threading
import warnings
from typing import Any, List, Optional, Protocol, Tuple, Union

import torch

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


def _eval_op(
    fn_lib: LieGroupFns,
    op_name: str,
    input0: torch.Tensor,
    jacobians: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    return getattr(fn_lib, op_name)(input0, jacobians=jacobians)


def _get_fn_lib(ltype: _ltype):
    return {
        SE3: _se3_impl,
        SO3: _so3_impl,
    }[ltype]


class _LieTensorBase(torch.Tensor):
    _t: torch.Tensor
    ltype: _ltype

    def __new__(
        cls, data: Any, ltype: _ltype = _ltype.tgt, requires_grad: bool = False
    ):
        t: torch.Tensor = super().__new__(torch.Tensor, data)
        r: _LieTensorBase = torch.Tensor._make_subclass(cls, t, requires_grad)  # type: ignore
        r._t = t
        r.ltype = ltype
        return r

    def __repr__(self) -> str:
        return f"LieTensor({self._t}, ltype=lie.{self.ltype})"

    @staticmethod
    def _get_ltype(x: Any) -> Optional[_ltype]:
        if isinstance(x, (list, tuple)):
            return _LieTensorBase._resolve_ltype(x)
        if isinstance(x, _LieTensorBase):
            return x.ltype
        return None

    @staticmethod
    def _resolve_ltype(args: Any) -> Optional[_ltype]:
        args_ltypes = [_LieTensorBase._get_ltype(a) for a in args]
        ltypes = set([t for t in args_ltypes if t is not None])
        print("al", args_ltypes)
        if len(ltypes) > 1:
            raise ValueError("All LieTensors must be of the same ltype.")
        return next(iter(ltypes)) if len(ltypes) > 0 else None

    @classmethod
    def _torch_func_impl_eucl(cls, func, types, args=(), kwargs=None):
        print("---------")
        kwargs = kwargs or {}
        torch_args = [a._t if isinstance(a, cls) else a for a in args]
        ltype = cls._resolve_ltype(args)
        ret = super().__torch_function__(func, types, torch_args, kwargs)
        return cls(ret, ltype) if isinstance(ret, torch.Tensor) else ret


class TangentTensor(_LieTensorBase):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return cls._torch_func_impl_eucl(func, types, args, kwargs)


class LieTensor(_LieTensorBase):
    SAFE_TORCH_OPS_FOR_NONEUCL = [
        torch.cat,
        torch.clone,
        # torch.stack  # requires arbitrary batch support
    ]

    _fn_lib: LieGroupFns

    def __init__(self, data: Any, ltype: _ltype, requires_grad: bool = False):
        self._fn_lib = _get_fn_lib(self.ltype)
        self._fn_lib.check_group_tensor(self._t)

    @classmethod
    def _torch_function_impl_lie(cls, func, types, args=(), kwargs=None):
        if func not in LieTensor.SAFE_TORCH_OPS_FOR_NONEUCL:
            raise NotImplementedError(
                "Tried to call a torch function not supported by LieTensor. "
                "If trying to operate on the raw tensor data, please use group._t, "
                "or run inside the context lie.as_eucledian()."
            )
        return super()._torch_func_impl_eucl(func, types, args, kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if euclidean_enabled():
            return super()._torch_func_impl_eucl(func, types, args, kwargs)
        else:
            return cls._torch_function_impl_lie(func, types, args, kwargs)

    def _check_ltype(self, other: "LieTensor", op_name: str):
        if other.ltype != self.ltype:
            raise ValueError("f{op_name} requires both tensors to have same ltype.")

    # Returns a new LieTensor with the given data and the same ltype as self
    def new(self, *args: Any, device: Device = None) -> "LieTensor":
        if isinstance(args[0], LieTensor):
            warnings.warn(
                "Calling new() on a LieTensor results in shared data storage. "
                "To copy construct from a LieTensor, it is recommended to use lie_tensor.clone().",
                UserWarning,
            )
            return LieTensor(args[0]._t, ltype=self.ltype)
        return LieTensor(torch.as_tensor(*args, device=device), ltype=self.ltype)

    # ------ Operators
    # The following could be static methods, because self is used
    # only to infer the type. For each of this, there are
    # versions such as:
    #   - `lie.exp(ltype, tangent_vector)`
    #   - `lie.lift(ltype, matrix)`
    # that we expect to be more commonly used.
    def exp(self, tangent_vector: torch.Tensor) -> "LieTensor":
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

    # Operator Jacobians
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

    def __add__(self, other: TensorType) -> "LieTensor":
        if not isinstance(other, TangentTensor):
            raise RuntimeError(
                "Operator + is only supported for tensors of ltype=tgt. "
                "If you intend to add the raw tensor data, please use group._t. "
                "If you intend to retract, then cast your tensor to ltype=tgt, or "
                "use group.retract(tensor)."
            )
        return self.retract(other._t)


# ----------------------------
# Tensor creation functions
# ----------------------------
def new(
    data: Any,
    ltype: Optional[_ltype] = None,
    dtype: torch.dtype = torch.float,
    device: Union[torch.device, str] = None,
) -> _LieTensorBase:
    if isinstance(data, LieTensor):
        return data.new(data)
    if ltype is None:
        raise ValueError("ltype must be provided.")
    if ltype == _ltype.tgt:
        return TangentTensor(data)
    return LieTensor(data, ltype=ltype)


as_lietensor = new

# With this new version of the code I like @fantaosha's proposal of using the name cast()
# because it implies type conversion rather than a wrapper being created.
cast = new


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


def exp(ltype: _ltype, tangent_vector: torch.Tensor) -> "LieTensor":
    return LieTensor(_eval_op(_get_fn_lib(ltype), "exp", tangent_vector), ltype)


def hat(ltype: _ltype, tangent_vector: torch.Tensor) -> torch.Tensor:
    return _eval_op(_get_fn_lib(ltype), "hat", tangent_vector)


def vee(ltype: _ltype, matrix: torch.Tensor) -> torch.Tensor:
    return _eval_op(_get_fn_lib(ltype), "vee", matrix)


def lift(ltype: _ltype, matrix: torch.Tensor) -> torch.Tensor:
    return _eval_op(_get_fn_lib(ltype), "lift", matrix)


def project(ltype: _ltype, matrix: torch.Tensor) -> torch.Tensor:
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


def jexp(ltype: _ltype, tangent_vector: torch.Tensor) -> _JFnReturnType:
    jacs: List[torch.Tensor] = []
    exp_tensor = _eval_op(_get_fn_lib(ltype), "exp", tangent_vector, jacobians=jacs)
    return jacs, LieTensor(exp_tensor, ltype)


def jcompose(group1: LieTensor, group2: LieTensor) -> _JFnReturnType:
    return group1.jcompose(group2)


def retract(group: LieTensor, delta: TensorType) -> LieTensor:
    return group.retract(delta)
