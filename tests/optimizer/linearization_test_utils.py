# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch  # noqa: F401

import theseus as th


class MockVector(th.Manifold):
    def __init__(self, value, length, name=None):
        super().__init__(value, length, name=name)

    def _init_tensor(value, length):
        return value * torch.ones(1, length)

    @staticmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        return True

    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def dof(self):
        return self.tensor.shape[1]

    def numel(self):
        return self.tensor.shape[1]

    def _local_impl(self, variable2):
        return torch.zeros(1)

    def _local_jacobian(self, vec2):
        pass

    def _retract_impl(self, delta):
        return self

    def _copy_impl(self):
        raise NotImplementedError

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        return euclidean_grad.clone()


class MockCostFunction(th.CostFunction):
    def __init__(self, optim_vars, cost_weight, dim, name=None):
        super().__init__(cost_weight, name=name)
        for i, var in enumerate(optim_vars):
            attr_name = f"optim_var_{i}"
            setattr(self, attr_name, var)
            self.register_optim_var(attr_name)
        self.the_dim = dim

    # Error function for this cost function is (for each batch element)
    #       ei = (i + 1) * sum_j (j + 1) * sum_k [var_j]_k
    def error(self):
        batch_size = self.optim_var_0.tensor.shape[0]
        error = torch.zeros(batch_size, self.the_dim)
        for batch_idx in range(batch_size):
            value = torch.sum(
                torch.tensor(
                    [
                        (j + 1) * v[batch_idx].sum()
                        for j, v in enumerate(self.optim_vars)
                    ]
                )
            )
            for i in range(self.the_dim):
                error[batch_idx, i] = (i + 1) * value
        return error

    # Jacobian_var_i = (i + 1) * jv
    # where jv is a matrix with self.the_dim rows such that:
    #   jv[j,k] = (j + 1) * [1, 1, 1, ..., 1] (#vars times, i.e., v.dim())
    def jacobians(self):
        batch_size = self.optim_var_0.tensor.shape[0]
        jacs = []
        for i, v in enumerate(self.optim_vars):
            jv = torch.ones(batch_size, self.the_dim, v.dof())
            jv = jv * torch.arange(1, self.the_dim + 1).view(1, -1, 1)
            jacs.append(jv * (i + 1))
        return jacs, self.error()

    def dim(self):
        return self.the_dim

    def _copy_impl(self):
        raise NotImplementedError


# This is just an identity matrix times some multiplier
class MockCostWeight(th.CostWeight):
    def __init__(self, dim, mult, name=None):
        super().__init__(name=name)
        self.sqrt = th.Variable(torch.eye(dim).unsqueeze(0) * mult)
        self.register_aux_var("sqrt")

    def weight_error(self, error):
        return NotImplemented

    def weight_jacobians_and_error(self, jacobians, error):
        batch_size = error.shape[0]
        sqrt = self.sqrt.tensor.repeat((batch_size, 1, 1))
        wjs = []
        for jac in jacobians:
            wjs.append(torch.matmul(self.sqrt.tensor, jac))
        werr = torch.matmul(sqrt, error.unsqueeze(2)).squeeze(2)
        return wjs, werr

    def _copy_impl(self, new_name=None):
        raise NotImplementedError


def build_test_objective_and_linear_system():
    # This function creates the an objective that results in the
    # following Ax = b, with A =
    #
    #    |   v4   |   v3  |  v2 | v1 |
    # f1 [- - - - | - - - | 2 2 |  1 ]
    # f2 [- - - - | 2 2 2 | - - |  1 ] * 2 * 1
    # f2 [- - - - | 2 2 2 | - - |  1 ] * 2 * 2
    # f3 [2 2 2 2 | - - - | 1 1 |  _ ] * 3 * 1
    # f3 [2 2 2 2 | - - - | 1 1 |  _ ] * 3 * 2
    # f3 [2 2 2 2 | - - - | 1 1 |  _ ] * 3 * 3
    #
    #   (the numbers at the right of the matrix are (cov) * cost_function_dim)
    #
    # and b = -[9, 38, 76, 108, 216, 324]
    #
    # Then checks that torch.linearization produces the same result
    var1 = MockVector(1, 1, name="v1")
    var2 = MockVector(2, 2, name="v2")
    var3 = MockVector(3, 3, name="v3")
    var4 = MockVector(4, 4, name="v4")

    cost_function1 = MockCostFunction(
        [var1, var2], MockCostWeight(1, 1, name="cov1"), 1, name="f1"
    )
    cost_function2 = MockCostFunction(
        [var1, var3], MockCostWeight(2, 2, name="cov2"), 2, name="f2"
    )
    cost_function3 = MockCostFunction(
        [var2, var4], MockCostWeight(3, 3, name="cov3"), 3, name="f3"
    )

    objective = th.Objective()
    objective.add(cost_function1)
    objective.add(cost_function2)
    objective.add(cost_function3)

    ordering = th.VariableOrdering(objective, default_order=False)
    ordering.extend([var4, var3, var2, var1])

    batch_size = 4
    objective.update(
        {
            "v1": torch.ones(batch_size, 1),
            "v2": 2 * torch.ones(batch_size, 2),
            "v3": 3 * torch.ones(batch_size, 3),
            "v4": 4 * torch.ones(batch_size, 4),
        }
    )

    # if cost_function_k(vi, vj) then error is i ** 2 + 2 * (j ** 2)
    # and weight is k
    wer1 = 1.0 * (1 + 2 * 4)
    wer2 = 2.0 * (1 + 2 * 9)
    wer3 = 3.0 * (4 + 2 * 16)

    b = -torch.tensor([wer1, wer2, 2 * wer2, wer3, 2 * wer3, 3 * wer3])
    b = b.unsqueeze(0).repeat(batch_size, 1)

    # --- The following puts together the matrix shown above
    # weighted jacobians for all errors
    var1j = torch.ones(1)
    var2j = torch.ones(2)
    var3j = torch.ones(3)
    var4j = torch.ones(4)

    A1 = torch.cat([torch.zeros(7), 2 * var2j, var1j]).view(1, -1)
    A2p = torch.cat([torch.zeros(4), 2 * var3j, torch.zeros(2), var1j])
    A2 = 2 * torch.stack([A2p, 2 * A2p])
    A3p = torch.cat([2 * var4j, torch.zeros(3), var2j, torch.zeros(1)])
    A3 = 3 * torch.stack([A3p, 2 * A3p, 3 * A3p])
    A = torch.cat([A1, A2, A3])
    A = A.unsqueeze(0).repeat(batch_size, 1, 1)

    return objective, ordering, A, b
