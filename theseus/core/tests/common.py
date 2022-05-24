import copy

import torch

import theseus as th


class MockVar(th.Manifold):
    def __init__(self, length, data=None, name=None):
        super().__init__(length, data=data, name=name)

    @staticmethod
    def _init_data(length):
        return torch.empty(1, length)

    def dof(self):
        return 0

    def _local_impl(self, variable2):
        pass

    def _local_jacobian(self, var2):
        pass

    def _retract_impl(delta):
        pass

    def _copy_impl(self, new_name=None):
        return MockVar(self.data.shape[1], data=self.data.clone(), name=new_name)

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        return euclidean_grad.clone()


class MockCostWeight(th.CostWeight):
    def __init__(
        self, the_data, name=None, add_dummy_var_with_name=None, add_optim_var=None
    ):
        super().__init__(name=name)
        self.the_data = the_data
        self.register_aux_var("the_data")
        if add_dummy_var_with_name:
            var = MockVar(1, name=add_dummy_var_with_name)
            setattr(self, add_dummy_var_with_name, var)
            self.register_optim_var(add_dummy_var_with_name)
        if add_optim_var:
            setattr(self, add_optim_var.name, add_optim_var)
            self.register_optim_var(add_optim_var.name)

    def weight_error(self, error):
        return self.the_data * error

    def weight_jacobians_and_error(self, jacobians, error):
        raise NotImplementedError(
            "weight_jacobians_and_error is not implemented for MockCostWeight."
        )

    def _copy_impl(self, new_name=None):
        return MockCostWeight(self.the_data.copy(), name=new_name)


class NullCostWeight(th.CostWeight):
    def __init__(self):
        super().__init__(name="null_cost_weight")

    def _init_data(self):
        pass

    def weight_error(self, error):
        return error

    def weight_jacobians_and_error(self, jacobians, error):
        return jacobians, error

    def _copy_impl(self, new_name=None):
        return NullCostWeight()


class MockCostFunction(th.CostFunction):
    def __init__(self, optim_vars, aux_vars, cost_weight, name=None):
        super().__init__(cost_weight, name=name)
        for i, var in enumerate(optim_vars):
            attr_name = f"optim_var_{i}"
            setattr(self, attr_name, var)
            self.register_optim_var(attr_name)
        for i, aux in enumerate(aux_vars):
            attr_name = f"aux_var_{i}"
            setattr(self, attr_name, aux)
            self.register_aux_var(attr_name)
        self._dim = 2

    def error(self):
        mu = torch.stack([v.data for v in self.optim_vars]).sum()
        return mu * torch.ones(self._dim)

    def jacobians(self):
        return [self.error()] * len(self._optim_vars_attr_names)

    def dim(self) -> int:
        return self._dim

    def _copy_impl(self, new_name=None):
        return MockCostFunction(
            [v.copy() for v in self.optim_vars],
            [aux.copy() for aux in self.aux_vars],
            self.weight.copy(),
            name=new_name,
        )


def create_mock_cost_functions(data=None, cost_weight=NullCostWeight()):
    len_data = 1 if data is None else data.shape[1]
    var1 = MockVar(len_data, data=data, name="var1")
    var2 = MockVar(len_data, data=data, name="var2")
    var3 = MockVar(len_data, data=data, name="var3")
    aux1 = MockVar(len_data, data=data, name="aux1")
    aux2 = MockVar(len_data, data=data, name="aux2")
    names = [
        "MockCostFunction.var1.var2",
        "MockCostFunction.var1.var3",
        "MockCostFunction.var2.var3",
    ]
    cost_function_1_2 = MockCostFunction(
        [var1, var2], [aux1, aux2], cost_weight, name=names[0]
    )
    cost_function_1_3 = MockCostFunction(
        [var1, var3], [aux1], cost_weight, name=names[1]
    )
    cost_function_2_3 = MockCostFunction(
        [var2, var3], [aux2], cost_weight, name=names[2]
    )

    var_to_cost_functions = {
        var1: [cost_function_1_2, cost_function_1_3],
        var2: [cost_function_1_2, cost_function_2_3],
        var3: [cost_function_2_3, cost_function_1_3],
    }
    aux_to_cost_functions = {
        aux1: [cost_function_1_2, cost_function_1_3],
        aux2: [cost_function_1_2, cost_function_2_3],
    }
    return (
        [cost_function_1_2, cost_function_1_3, cost_function_2_3],
        names,
        var_to_cost_functions,
        aux_to_cost_functions,
    )


def create_objective_with_mock_cost_functions(data=None, cost_weight=NullCostWeight()):
    (
        cost_functions,
        names,
        var_to_cost_functions,
        aux_to_cost_functions,
    ) = create_mock_cost_functions(data=data, cost_weight=cost_weight)

    objective = th.Objective()
    for cost_function in cost_functions:
        objective.add(cost_function)

    return (
        objective,
        cost_functions,
        names,
        var_to_cost_functions,
        aux_to_cost_functions,
    )


def check_copy_var(var):
    var.name = "old"
    new_var = var.copy(new_name="new")
    assert var is not new_var
    assert var.data is not new_var.data
    assert torch.allclose(var.data, new_var.data)
    assert new_var.name == "new"
    new_var_no_name = copy.deepcopy(var)
    assert new_var_no_name.name == f"{var.name}_copy"


def check_another_theseus_tensor_is_copy(var, other_var):
    assert isinstance(var, other_var.__class__)
    assert var is not other_var
    check_another_torch_tensor_is_copy(var.data, other_var.data)


def check_another_torch_tensor_is_copy(tensor, other_tensor):
    assert tensor is not other_tensor
    assert torch.allclose(tensor, other_tensor)


def check_another_theseus_function_is_copy(fn, other_fn, new_name):
    assert fn is not other_fn
    assert other_fn.name == new_name
    for var, new_var in zip(fn.optim_vars, other_fn.optim_vars):
        check_another_theseus_tensor_is_copy(var, new_var)
    for aux, new_aux in zip(fn.aux_vars, other_fn.aux_vars):
        check_another_theseus_tensor_is_copy(aux, new_aux)
