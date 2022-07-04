import torch

import theseus as th
from theseus.optimizer.gbp import GaussianBeliefPropagation, synchronous_schedule

torch.manual_seed(0)


def generate_data(num_points=100, a=1, b=0.5, noise_factor=0.01):
    # Generate data: 100 points sampled from the quadratic curve listed above
    data_x = torch.rand((1, num_points))
    noise = torch.randn((1, num_points)) * noise_factor
    data_y = a * data_x.square() + b + noise
    return data_x, data_y


def generate_learning_data(num_points, num_models):
    a, b = 3, 1
    data_batches = []
    for i in range(num_models):
        b = b + 2
        data = generate_data(num_points, a, b)
        data_batches.append(data)
    return data_batches


num_models = 10
data_batches = generate_learning_data(100, num_models)


# updated error function reflects change in 'a'
def quad_error_fn2(optim_vars, aux_vars):
    [a, b] = optim_vars
    x, y = aux_vars
    est = a.data * x.data.square() + b.data
    err = y.data - est
    return err


# The theseus_inputs dictionary is also constructed similarly to before,
# but with data matching the new shapes of the variables
def construct_theseus_layer_inputs():
    theseus_inputs = {}
    theseus_inputs.update(
        {
            "x": data_x,
            "y": data_y,
            "b": torch.ones((num_models, 1)),
            "a": a_tensor,
        }
    )
    return theseus_inputs


# convert data_x, data_y into torch.tensors of shape [num_models, 100]
data_x = torch.stack([data_x.squeeze() for data_x, _ in data_batches])
data_y = torch.stack([data_y.squeeze() for _, data_y in data_batches])

# construct one variable each of x, y of shape [num_models, 100]
x = th.Variable(data_x, name="x")
y = th.Variable(data_y, name="y")

# construct a as before
a = th.Vector(data=torch.rand(num_models, 1), name="a")

# construct one variable b, now of shape [num_models, 1]
b = th.Vector(data=torch.rand(num_models, 1), name="b")

# Again, 'b' is the only optim_var, and 'a' is part of aux_vars along with x, y
aux_vars = [x, y]

# cost function constructed as before
cost_function = th.AutoDiffCostFunction(
    [a, b], quad_error_fn2, 100, aux_vars=aux_vars, name="quadratic_cost_fn"
)

prior_weight = th.ScaleCostWeight(torch.ones(1))
prior_a = th.Difference(a, prior_weight, th.Vector(1))
prior_b = th.Difference(b, prior_weight, th.Vector(1))

# objective, optimizer and theseus layer constructed as before
objective = th.Objective()
objective.add(cost_function)
objective.add(prior_a)
objective.add(prior_b)

print([cf.name for cf in objective.cost_functions.values()])

optimizer = GaussianBeliefPropagation(
    objective,
    max_iterations=50,  # step_size=0.5,
)

theseus_optim = th.TheseusLayer(optimizer, vectorize=False)

a_tensor = torch.nn.Parameter(torch.rand(num_models, 1))


optim_arg = {
    "track_best_solution": True,
    "track_err_history": True,
    "verbose": True,
    "backward_mode": th.BackwardMode.FULL,
    "relin_threshold": 0.0000000001,
    "damping": 0.5,
    "dropout": 0.0,
    "schedule": synchronous_schedule(50, optimizer.n_edges),
    "lin_system_damping": 1e-5,
}


theseus_inputs = construct_theseus_layer_inputs()
print("inputs\n", theseus_inputs["a"], theseus_inputs["x"].shape)
updated_inputs, _ = theseus_optim.forward(theseus_inputs, optim_arg)

print(updated_inputs)
