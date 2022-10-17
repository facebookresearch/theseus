import torch

import theseus as th
from theseus import DCem


def test_dcem_optim():
    # Step 1: Construct optimization and auxiliary variables.
    # Construct variables of the function: these the optimization variables of the cost functions.
    x = th.Vector(1, name="x")
    y = th.Vector(1, name="y")

    # Construct auxiliary variables for the constants of the function.
    a = th.Vector(tensor=torch.randn(1, 1), name="a")
    b = th.Vector(tensor=torch.randn(1, 1), name="b")

    # Step 2: Construct cost weights
    # For w1, let's use a named variable
    w1 = th.ScaleCostWeight(th.Variable(tensor=torch.ones(1, 1), name="w1_sqrt"))
    w2 = th.ScaleCostWeight(2.0)  # we provide 2, as sqrt of 4 for the (y-b)^2 term

    # Step 3: Construct cost functions representing each error term
    # First term
    cf1 = th.Difference(x, a, w1, name="term_1")
    # Second term
    cf2 = th.Difference(y, b, w2, name="term_2")

    objective = th.Objective()
    objective.add(cf1)
    objective.add(cf2)

    # Step 5: Evaluate objective under current values
    # Note this needs to be preceded by a call to `objective.update`
    # Here we use the update function to set values of all variables
    objective.update(
        {
            "a": torch.ones(1, 1),
            "b": 2 * torch.ones(1, 1),
            "x": 0.5 * torch.ones(1, 1),
            "y": 3 * torch.ones(1, 1),
        }
    )

    # Recall that our objective is (x - a)^2 + 4 (y - b)^2
    # which is minimized at x = a and y = b
    # Let's start by assigning random values to them
    objective.update({"x": torch.randn(1, 1), "y": torch.randn(1, 1)})

    optimizer = DCem(objective)

    info = optimizer.optimize()

    print(info)
    assert torch.tensor(
        [info.best_solution["x"] - 1.0 < 1e-3, info.best_solution["y"] - 2.0 < 1e-3]
    ).all()


def test_dcem_theseus_layer():
    # Step 1: Construct optimization and auxiliary variables.
    # Construct variables of the function: these the optimization variables of the cost functions.
    x = th.Vector(1, name="x")
    y = th.Vector(1, name="y")

    # Construct auxiliary variables for the constants of the function.
    a = th.Vector(tensor=torch.randn(1, 1), name="a")
    b = th.Vector(tensor=torch.randn(1, 1), name="b")

    # Step 2: Construct cost weights
    # For w1, let's use a named variable
    w1 = th.ScaleCostWeight(th.Variable(tensor=torch.ones(1, 1), name="w1_sqrt"))
    w2 = th.ScaleCostWeight(2.0)  # we provide 2, as sqrt of 4 for the (y-b)^2 term

    # Step 3: Construct cost functions representing each error term
    # First term
    cf1 = th.Difference(x, a, w1, name="term_1")
    # Second term
    cf2 = th.Difference(y, b, w2, name="term_2")

    objective = th.Objective()
    objective.add(cf1)
    objective.add(cf2)

    # Step 5: Evaluate objective under current values
    # Note this needs to be preceded by a call to `objective.update`
    # Here we use the update function to set values of all variables
    objective.update(
        {
            "a": torch.ones(1, 1),
            "b": 2 * torch.ones(1, 1),
            "x": 0.5 * torch.ones(1, 1),
            "y": 3 * torch.ones(1, 1),
        }
    )

    # Recall that our objective is (x - a)^2 + 4 (y - b)^2
    # which is minimized at x = a and y = b
    # Let's start by assigning random values to them
    objective.update({"x": torch.randn(1, 1), "y": torch.randn(1, 1)})

    optimizer = DCem(objective)

    layer = th.TheseusLayer(optimizer)
    values, info = layer.forward(
        {
            "x": torch.randn(1, 1),
            "y": torch.randn(1, 1),
            "a": torch.ones(1, 1),
            "b": 2 * torch.ones(1, 1),
            "w1_sqrt": torch.ones(1, 1),
        }
    )

    print(info)
    assert torch.tensor(
        [info.best_solution["x"] - 1.0 < 1e-3, info.best_solution["y"] - 2.0 < 1e-3]
    ).all()


test_dcem_optim()
