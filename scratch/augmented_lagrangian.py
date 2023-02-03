# Implements the augmented Lagrangian method for constrained
# nonlinear least squares as described on page 11.21 of
# http://www.seas.ucla.edu/~vandenbe/133B/lectures/nllseq.pdf

import copy
import torch
import theseus as th


def solve_augmented_lagrangian(
    err_fn,
    optim_vars,
    aux_vars,
    dim,
    equality_constraint_fn,
    optimizer_cls,
    optimizer_kwargs,
    verbose,
    initial_state_dict,
    num_augmented_lagrangian_iterations=10,
    callback=None,
):
    def err_fn_augmented(optim_vars, aux_vars):
        # TODO: Dangerous to override optim_vars,aux_vars here
        original_aux_vars = aux_vars[:-2]
        original_err = err_fn(optim_vars, original_aux_vars)
        g = equality_constraint_fn(optim_vars, original_aux_vars)

        mu, z = aux_vars[-2:]
        sqrt_mu = torch.sqrt(mu.tensor)
        err_augmented = sqrt_mu * g + z.tensor / (2.0 * sqrt_mu)
        combined_err = torch.cat([original_err, err_augmented], axis=-1)
        return combined_err

    g = equality_constraint_fn(optim_vars, aux_vars)
    dim_constraints = g.shape[-1]
    dim_augmented = dim + dim_constraints

    num_batch = 1  # TODO: Infer this from somewhere else?
    mu = th.Variable(torch.ones(num_batch, 1), name="mu")
    z = th.Variable(torch.zeros(num_batch, dim_constraints), name="z")
    aux_vars_augmented = aux_vars + [mu, z]

    cost_fn_augmented = th.AutoDiffCostFunction(
        optim_vars=optim_vars,
        err_fn=err_fn_augmented,
        dim=dim_augmented,
        aux_vars=aux_vars_augmented,
        name="cost_fn_augmented",
    )

    objective = th.Objective()
    objective.add(cost_fn_augmented)

    optimizer = optimizer_cls(objective, **optimizer_kwargs)
    theseus_optim = th.TheseusLayer(optimizer)
    state_dict = copy.copy(initial_state_dict)

    prev_g_norm = 0
    for i in range(num_augmented_lagrangian_iterations):
        state_dict, info = theseus_optim.forward(state_dict)
        g_x = equality_constraint_fn(optim_vars, aux_vars)
        g_norm = g_x.norm()
        if verbose:
            print(f"=== [{i}] mu: {mu.tensor.item():.2e}, ||g||: {g_norm:.2e}")
            print(state_dict)
        if callback is not None:
            callback(state_dict)
        z.tensor = z.tensor + 2 * mu.tensor * g_x
        if i > 0 and g_norm > 0.25 * prev_g_norm:
            mu.tensor = 2 * mu.tensor
        prev_g_norm = g_norm

    return state_dict
