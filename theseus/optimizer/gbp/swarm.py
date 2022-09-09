import numpy as np
import random
import omegaconf
import time
from typing import Optional, Tuple, List

import torch

import theseus as th
from theseus.optimizer.gbp import GaussianBeliefPropagation, GBPSchedule

# from theseus.optimizer.gbp import SwarmViewer


OPTIMIZER_CLASS = {
    "gbp": GaussianBeliefPropagation,
    "gauss_newton": th.GaussNewton,
    "levenberg_marquardt": th.LevenbergMarquardt,
}

OUTER_OPTIMIZER_CLASS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

GBP_SCHEDULE = {
    "synchronous": GBPSchedule.SYNCHRONOUS,
}


def fc_block(in_f, out_f):
    return torch.nn.Sequential(torch.nn.Linear(in_f, out_f), torch.nn.ReLU())


class TargetMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=2,
        hidden_dim=8,
        hidden_layers=0,
    ):
        super(TargetMLP, self).__init__()
        # input is agent index
        self.relu = torch.nn.ReLU()
        self.in_layer = torch.nn.Linear(1, hidden_dim)
        hidden = [fc_block(hidden_dim, hidden_dim) for _ in range(hidden_layers)]
        self.mid = torch.nn.Sequential(*hidden)
        self.out_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.in_layer(x))
        x = self.mid(x)
        out = self.out_layer(x)
        return out


# custom factor for two agents collision
class TwoAgentsCollision(th.CostFunction):
    def __init__(
        self,
        weight: th.CostWeight,
        var1: th.Point2,
        var2: th.Point2,
        radius: th.Vector,
        name: Optional[str] = None,
    ):
        super().__init__(weight, name=name)
        self.var1 = var1
        self.var2 = var2
        self.radius = radius
        # skips data checks
        self.register_optim_vars(["var1", "var2"])
        self.register_aux_vars(["radius"])

    # no error when distance exceeds radius
    def error(self) -> torch.Tensor:
        dist = torch.norm(self.var1.tensor - self.var2.tensor, dim=1, keepdim=True)
        return torch.relu(1 - dist / self.radius.tensor)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        dist = torch.norm(self.var1.tensor - self.var2.tensor, dim=1, keepdim=True)
        denom = dist * self.radius.tensor
        jac = (self.var1.tensor - self.var2.tensor) / denom
        jac = jac[:, None, :]
        jac[dist > self.radius.tensor] = 0.0
        return [-jac, jac], self.error()

    def dim(self) -> int:
        return 1

    def _copy_impl(self, new_name: Optional[str] = None) -> "TwoAgentsCollision":
        return TwoAgentsCollision(
            self.weight.copy(),
            self.var1.copy(),
            self.var2.copy(),
            self.radius.copy(),
            name=new_name,
        )


# all agents should be in square of side length 1 centered at the origin
def square_loss_fn(outputs, side_len):
    positions = torch.cat(list(outputs.values()))
    loss = torch.relu(torch.abs(positions) - side_len / 2)
    return loss.sum()


def setup_problem(cfg: omegaconf.OmegaConf):
    dtype = torch.float32
    n_agents = cfg["setup"]["num_agents"]

    # create variables, one per agent
    positions = []
    for i in range(n_agents):
        init = torch.normal(torch.zeros(2), cfg["setup"]["init_std"])
        position = th.Point2(tensor=init, name=f"agent_{i}")
        positions.append(position)

    objective = th.Objective(dtype=dtype)

    # prior factor drawing each robot to the origin
    # origin = th.Point2(name="origin")
    # origin_weight = th.ScaleCostWeight(
    #     torch.tensor([cfg["setup"]["origin_weight"]], dtype=dtype)
    # )
    # for i in range(n_agents):
    #     origin_cf = th.Difference(
    #         positions[i],
    #         origin,
    #         origin_weight,
    #         name=f"origin_pull_{i}",
    #     )
    #     objective.add(origin_cf)

    # create collision factors, fully connected
    radius = th.Vector(
        tensor=torch.tensor([cfg["setup"]["collision_radius"]]), name="radius"
    )
    collision_weight = th.ScaleCostWeight(
        torch.tensor([cfg["setup"]["collision_weight"]], dtype=dtype)
    )
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            collision_cf = TwoAgentsCollision(
                weight=collision_weight,
                var1=positions[i],
                var2=positions[j],
                radius=radius,
                name=f"collision_{i}_{j}",
            )
            objective.add(collision_cf)

    # learned factors, encouraging a square formation
    target_weight = th.ScaleCostWeight(
        torch.tensor([cfg["setup"]["origin_weight"] * 10], dtype=dtype)
    )
    for i in range(n_agents):
        target = th.Point2(
            tensor=torch.normal(torch.zeros(2), cfg["setup"]["init_std"]),
            name=f"target_{i}",
        )
        target_cf = th.Difference(
            positions[i],
            target,
            target_weight,
            name=f"formation_target_{i}",
        )
        objective.add(target_cf)

    return objective


def main(cfg: omegaconf.OmegaConf):

    objective = setup_problem(cfg)

    #  setup optimizer and theseus layer
    vectorize = cfg["optim"]["vectorize"]
    optimizer = OPTIMIZER_CLASS[cfg["optim"]["optimizer_cls"]](
        objective,
        max_iterations=cfg["optim"]["max_iters"],
        vectorize=vectorize,
        # linearization_cls=th.SparseLinearization,
        # linear_solver_cls=th.LUCudaSparseSolver,
    )
    theseus_optim = th.TheseusLayer(optimizer, vectorize=vectorize)

    if cfg["device"] == "cuda":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    theseus_optim.to(cfg["device"])

    optim_arg = {
        "track_best_solution": False,
        "track_err_history": True,
        "track_state_history": True,
        "verbose": True,
        "backward_mode": th.BackwardMode.FULL,
    }
    if isinstance(optimizer, GaussianBeliefPropagation):
        gbp_args = cfg["optim"]["gbp_settings"].copy()
        lin_system_damping = torch.nn.Parameter(
            torch.tensor(
                [cfg["optim"]["gbp_settings"]["lin_system_damping"]],
                dtype=torch.float32,
            )
        )
        lin_system_damping.to(device=cfg["device"])
        gbp_args["lin_system_damping"] = lin_system_damping
        gbp_args["schedule"] = GBP_SCHEDULE[gbp_args["schedule"]]
        optim_arg = {**optim_arg, **gbp_args}

    # theseus inputs
    theseus_inputs = {}
    for agent in objective.optim_vars.values():
        theseus_inputs[agent.name] = agent.tensor.clone()

    # setup outer optimizer
    targets = {}
    for name, aux_var in objective.aux_vars.items():
        if "target" in name:
            targets[name] = torch.nn.Parameter(aux_var.tensor.clone())
    outer_optimizer = OUTER_OPTIMIZER_CLASS[cfg["outer_optim"]["optimizer"]](
        targets.values(), lr=cfg["outer_optim"]["lr"]
    )

    losses = []
    targets_history = {}
    for k, target in targets.items():
        targets_history[k] = target.detach().clone().cpu().unsqueeze(-1)

    for epoch in range(cfg["outer_optim"]["num_epochs"]):
        print(f" ******************* EPOCH {epoch} ******************* ")
        start_time = time.time_ns()
        outer_optimizer.zero_grad()

        for k, target in targets.items():
            theseus_inputs[k] = target.clone()

        theseus_outputs, info = theseus_optim.forward(
            input_tensors=theseus_inputs,
            optimizer_kwargs=optim_arg,
        )

        if epoch < cfg["outer"]["num_epochs"] - 1:
            loss = square_loss_fn(
                theseus_outputs, cfg["outer_optim"]["square_side_len"]
            )
            loss.backward()
            outer_optimizer.step()
            losses.append(loss.detach().item())
            end_time = time.time_ns()

            for k, target in targets.items():
                targets_history[k] = torch.cat(
                    (targets_history[k], target.detach().clone().cpu().unsqueeze(-1)),
                    dim=-1,
                )

            print(f"Loss {losses[-1]}")
            print(f"Epoch took {(end_time - start_time) / 1e9: .3f} seconds")

    print("Loss values:", losses)

    # visualisation
    # viewer = SwarmViewer(
    #     cfg["setup"]["agent_radius"],
    #     cfg["setup"]["collision_radius"],
    # )

    # viewer.vis_outer_targets_optim(
    #     targets_history,
    #     square_side=cfg["outer_optim"]["square_side_len"],
    #     video_file=cfg["outer_optim_video_file"],
    # )

    # viewer.vis_inner_optim(
    #     info.state_history,
    #     targets=targets,  # make sure targets are from correct innner optim
    #     show_edges=False,
    #     video_file=cfg["out_video_file"],
    # )


if __name__ == "__main__":

    cfg = {
        "seed": 0,
        "device": "cpu",
        "out_video_file": "outputs/swarm/inner.gif",
        "outer_optim_video_file": "outputs/swarm/outer_targets.gif",
        "setup": {
            "num_agents": 80,
            "init_std": 1.0,
            "agent_radius": 0.1,
            "collision_radius": 1.0,
            "origin_weight": 0.3,
            "collision_weight": 1.0,
        },
        "optim": {
            "max_iters": 20,
            "vectorize": True,
            "optimizer_cls": "gbp",
            # "optimizer_cls": "gauss_newton",
            # "optimizer_cls": "levenberg_marquardt",
            "gbp_settings": {
                "relin_threshold": 1e-8,
                "ftov_msg_damping": 0.0,
                "dropout": 0.0,
                "schedule": "synchronous",
                "lin_system_damping": 1.0e-2,
                "nesterov": False,
            },
        },
        "outer_optim": {
            "num_epochs": 25,
            "lr": 4e-1,
            "optimizer": "sgd",
            "square_side_len": 2.0,
        },
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    main(cfg)
