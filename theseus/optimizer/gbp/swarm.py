import numpy as np
import random
import omegaconf
import torch
from typing import Optional, Tuple, List

import pygame

import theseus as th
from theseus.optimizer.gbp import GaussianBeliefPropagation, GBPSchedule


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


class SwarmViewer:
    def __init__(
        self,
        state_history,
        agent_radius,
        collision_radius,
        show_edges=True,
    ):
        self.state_history = state_history
        self.t = 0
        self.num_iters = (~list(state_history.values())[0].isinf()[0, 0]).sum()

        self.agent_cols = None
        self.scale = 100
        self.show_edges = show_edges
        self.agent_r_pix = agent_radius * self.scale
        self.collision_radius = collision_radius
        self.range = np.array([[-3, -3], [3, 3]])
        self.h = (self.range[1, 1] - self.range[0, 1]) * self.scale
        self.w = (self.range[1, 0] - self.range[0, 0]) * self.scale

        pygame.init()
        pygame.display.set_caption("Swarm")
        self.myfont = pygame.font.SysFont("Jokerman", 40)
        self.screen = pygame.display.set_mode([self.h, self.w])

        self.draw_next()

        running = True
        while running:

            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.draw_next()

    def draw_next(self):
        if self.agent_cols is None:
            self.agent_cols = [
                tuple(np.random.choice(range(256), size=3))
                for i in range(len(self.state_history))
            ]

        if self.t < self.num_iters:
            self.screen.fill((255, 255, 255))

            # draw agents
            for i, state in enumerate(self.state_history.values()):
                pos = state[0, :, self.t].cpu().numpy()
                centre = self.pos_to_canvas(pos)
                pygame.draw.circle(
                    self.screen, self.agent_cols[i], centre, self.agent_r_pix
                )

            # draw edges between agents
            if self.show_edges:
                for i, state1 in enumerate(self.state_history.values()):
                    pos1 = state1[0, :, self.t].cpu().numpy()
                    j = 0
                    for state2 in self.state_history.values():
                        if j <= i:
                            j += 1
                            continue
                        pos2 = state2[0, :, self.t].cpu().numpy()
                        dist = np.linalg.norm(pos1 - pos2)
                        if dist < self.collision_radius:
                            start = self.pos_to_canvas(pos1)
                            end = self.pos_to_canvas(pos2)
                            pygame.draw.line(self.screen, (0, 0, 0), start, end)

            # draw text
            ssshow = self.myfont.render(
                f"t = {self.t} / {self.num_iters - 1}", True, (0, 0, 0)
            )
            self.screen.blit(ssshow, (10, 10))  # choose location of text

            pygame.display.flip()

            self.t += 1

    def pos_to_canvas(self, pos):
        return (
            (pos - self.range[0])
            / (self.range[1] - self.range[0])
            * np.array([self.h, self.w])
        )


def error_fn(optim_vars, aux_vars):
    var1, var2 = optim_vars
    radius = aux_vars[0]
    return torch.relu(
        1 - torch.norm(var1.tensor - var2.tensor, dim=1, keepdim=True) / radius.tensor
    )


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
        # to improve readability, we have skipped the data checks from code block above
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
        return [
            -jac,
            jac,
        ], self.error()

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
    origin = th.Point2()
    origin_weight = th.ScaleCostWeight(
        torch.tensor([cfg["setup"]["origin_weight"]], dtype=dtype)
    )
    for i in range(n_agents):
        origin_cf = th.Difference(
            positions[i],
            origin,
            origin_weight,
            name=f"origin_pull_{i}",
        )
        objective.add(origin_cf)

    # create collision factors, fully connected
    radius = th.Vector(tensor=torch.tensor([cfg["setup"]["collision_radius"]]))
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

    # print("initial states\n", theseus_inputs)

    with torch.no_grad():
        theseus_outputs, info = theseus_optim.forward(
            input_tensors=theseus_inputs,
            optimizer_kwargs=optim_arg,
        )

    # print("final states\n", theseus_outputs)

    # visualisation
    # SwarmViewer(
    #     info.state_history,
    #     cfg["setup"]["agent_radius"],
    #     cfg["setup"]["collision_radius"],
    #     show_edges=False,
    # )


if __name__ == "__main__":

    cfg = {
        "seed": 0,
        "device": "cpu",
        "setup": {
            "num_agents": 100,
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
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    main(cfg)
