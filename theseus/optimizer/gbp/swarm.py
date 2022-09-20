import numpy as np
import random
import omegaconf
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Optional, Tuple, List, Callable

import torch
import torch.nn as nn

import theseus as th
from theseus.optimizer.gbp import GaussianBeliefPropagation, GBPSchedule
from theseus.optimizer.gbp import SwarmViewer


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


# create image from a character and font
def gen_char_img(
    char, dilate=True, fontname="LiberationSerif-Bold.ttf", size=(200, 200)
):
    img = Image.new("L", size, "white")
    draw = ImageDraw.Draw(img)
    fontsize = int(size[0] * 0.5)
    font = ImageFont.truetype(fontname, fontsize)
    char_displaysize = font.getsize(char)
    offset = tuple((si - sc) // 2 for si, sc in zip(size, char_displaysize))
    draw.text((offset[0], offset[1] * 3 // 4), char, font=font, fill="#000")

    if dilate:
        img = img.filter(ImageFilter.MinFilter(3))

    return img


# all agents should be inside object (negative SDF values)
def target_char_loss(outputs, sdf):
    positions = torch.cat(list(outputs.values()))
    dists = sdf.signed_distance(positions)[0]
    if torch.sum(dists == 0).item() != 0:
        print("\n\nNumber of agents out of bounds: ", torch.sum(dists == 0).item())
    loss = torch.relu(dists)
    return loss.sum()


def gen_target_sdf(cfg):
    # setup target shape for outer loop loss fn
    vis_limits = np.array(cfg["setup"]["vis_limits"])
    cell_size = 0.05
    img_size = tuple(np.rint((vis_limits[1] - vis_limits[0]) / cell_size).astype(int))
    img = gen_char_img(
        cfg["outer_optim"]["target_char"],
        dilate=True,
        fontname="DejaVuSans-Bold.ttf",
        size=img_size,
    )
    occ_map = torch.Tensor(np.array(img) < 255)
    occ_map = torch.flip(
        occ_map, [0]
    )  # flip vertically so y axis is upwards wrt character
    # pad to expand area
    area_limits = np.array(cfg["setup"]["area_limits"])
    padded_size = tuple(
        np.rint((area_limits[1] - area_limits[0]) / cell_size).astype(int)
    )
    pad = int((padded_size[0] - img_size[0]) / 2)
    larger_occ_map = torch.zeros(padded_size)
    larger_occ_map[pad:-pad, pad:-pad] = occ_map
    sdf = th.eb.SignedDistanceField2D(
        th.Variable(torch.Tensor(area_limits[0][None, :])),
        th.Variable(torch.Tensor([cell_size])),
        occupancy_map=th.Variable(larger_occ_map[None, :]),
    )
    return sdf


def fc_block(in_f, out_f):
    return nn.Sequential(nn.Linear(in_f, out_f), nn.ReLU())


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim=2,
        output_dim=2,
        hidden_dim=8,
        hidden_layers=0,
        scale_output=1.0,
    ):
        super(SimpleMLP, self).__init__()
        # input is agent index
        self.scale_output = scale_output
        self.relu = nn.ReLU()
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        hidden = [fc_block(hidden_dim, hidden_dim) for _ in range(hidden_layers)]
        self.mid = nn.Sequential(*hidden)
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        y = self.relu(self.in_layer(x))
        y = self.mid(y)
        out = self.out_layer(y) * self.scale_output
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


# custom factor for GNN
class GNNTargets(th.CostFunction):
    def __init__(
        self,
        weight: th.CostWeight,
        agents: List[th.Point2],
        gnn_err_fn: Callable,
        name: Optional[str] = None,
    ):
        super().__init__(weight, name=name)
        self.agents = agents
        self.n_agents = len(agents)
        self._gnn_err_fn = gnn_err_fn
        # skips data checks
        for agent in self.agents:
            setattr(self, agent.name, agent)
        self.register_optim_vars([v.name for v in agents])

    # no error when distance exceeds radius
    def error(self) -> torch.Tensor:
        return self._gnn_err_fn(self.agents)

    # Cannot use autodiff for jacobians as we want the factor to be
    # independent for each agent. i.e. GNN is implemented as many prior factors
    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        batch_size = self.agents[0].shape[0]
        jacs = torch.zeros(
            batch_size,
            self.n_agents,
            self.dim(),
            2,
            dtype=self.agents[0].dtype,
            device=self.agents[0].device,
        )
        jacs[:, torch.arange(self.n_agents), 2 * torch.arange(self.n_agents), 0] = 1.0
        jacs[
            :, torch.arange(self.n_agents), 2 * torch.arange(self.n_agents) + 1, 1
        ] = 1.0
        jac_list = [jacs[:, i] for i in range(self.n_agents)]
        return jac_list, self.error()

    def dim(self) -> int:
        return self.n_agents * 2

    def _copy_impl(self, new_name: Optional[str] = None) -> "GNNTargets":
        return GNNTargets(
            self.weight.copy(),
            [agent.copy() for agent in self.agents],
            self._gnn_err_fn,
            name=new_name,
        )


def setup_problem(cfg: omegaconf.OmegaConf, gnn_err_fn):
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
    origin = th.Point2(name="origin")
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

    # GNN factor - GNN takes in all current belief means and outputs all targets
    target_weight = th.ScaleCostWeight(
        torch.tensor([cfg["setup"]["gnn_target_weight"]], dtype=dtype)
    )
    gnn_cf = GNNTargets(
        weight=target_weight,
        agents=positions,
        gnn_err_fn=gnn_err_fn,
        name="gnn_factor",
    )
    objective.add(gnn_cf)

    return objective


class SwarmGBPAndGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        n_agents = cfg["setup"]["num_agents"]
        self.gnn = SimpleMLP(
            input_dim=2 * n_agents,
            output_dim=2 * n_agents,
            hidden_dim=64,
            hidden_layers=2,
            scale_output=1.0,
        )

        # setup objective, optimizer and theseus layer
        objective = setup_problem(cfg, self._gnn_err_fn)
        vectorize = cfg["optim"]["vectorize"]
        optimizer = OPTIMIZER_CLASS[cfg["optim"]["optimizer_cls"]](
            objective,
            max_iterations=cfg["optim"]["max_iters"],
            vectorize=vectorize,
        )
        self.layer = th.TheseusLayer(optimizer, vectorize=vectorize)

        # put on device
        if cfg["device"] == "cuda":
            cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.gnn.to(cfg["device"])
        self.layer.to(cfg["device"])

        # optimizer arguments
        optim_arg = {
            "track_best_solution": False,
            "track_err_history": True,
            "track_state_history": True,
            "verbose": True,
            "backward_mode": th.BackwardMode.FULL,
        }
        if isinstance(optimizer, GaussianBeliefPropagation):
            gbp_args = cfg["optim"]["gbp_settings"].copy()
            lin_system_damping = torch.tensor(
                [cfg["optim"]["gbp_settings"]["lin_system_damping"]],
                dtype=torch.float32,
            )
            lin_system_damping.to(device=cfg["device"])
            gbp_args["lin_system_damping"] = lin_system_damping
            gbp_args["schedule"] = GBP_SCHEDULE[gbp_args["schedule"]]
            optim_arg = {**optim_arg, **gbp_args}
        self.optim_arg = optim_arg

        # fixed inputs to theseus layer
        self.inputs = {}
        for agent in objective.optim_vars.values():
            self.inputs[agent.name] = agent.tensor.clone()

    # network outputs offset for target from agent position
    # cost is zero when offset is zero, i.e. agent is at the target
    def _gnn_err_fn(self, positions: List[th.Manifold]):
        flattened_pos = torch.cat(
            [pos.tensor.unsqueeze(1) for pos in positions], dim=1
        ).flatten(1, 2)
        offsets = self.gnn(flattened_pos)
        return offsets

    def forward(self, track_history=False):

        optim_arg = self.optim_arg.copy()
        optim_arg["track_state_history"] = track_history

        outputs, info = self.layer.forward(
            input_tensors=self.inputs,
            optimizer_kwargs=optim_arg,
        )

        history = None
        if track_history:

            history = info.state_history

            # recover target history
            agent_histories = torch.cat(
                [state_hist.unsqueeze(1) for state_hist in history.values()], dim=1
            )
            history["agent_0"]

            batch_size = agent_histories.shape[0]
            ts = agent_histories.shape[-1]
            agent_histories = agent_histories.permute(
                0, 3, 1, 2
            )  # time dim is second dim
            agent_histories = agent_histories.flatten(-2, -1)
            target_hist = self.gnn(agent_histories)
            target_hist = target_hist.reshape(batch_size, ts, -1, 2)
            target_hist = target_hist.permute(0, 2, 3, 1)  # time back to last dim

            for i in range(target_hist.shape[1]):
                history[f"target_{i}"] = -target_hist[:, i] + history[f"agent_{i}"]

        return outputs, history


def main(cfg: omegaconf.OmegaConf):

    sdf = gen_target_sdf(cfg)

    model = SwarmGBPAndGNN(cfg)

    outer_optimizer = OUTER_OPTIMIZER_CLASS[cfg["outer_optim"]["optimizer"]](
        model.gnn.parameters(), lr=cfg["outer_optim"]["lr"]
    )

    viewer = SwarmViewer(cfg["setup"]["collision_radius"], cfg["setup"]["vis_limits"])

    losses = []
    for epoch in range(cfg["outer_optim"]["num_epochs"]):
        print(f" ******************* EPOCH {epoch} ******************* ")
        start_time = time.time_ns()
        outer_optimizer.zero_grad()

        track_history = False  # epoch % 20 == 0
        outputs, history = model.forward(track_history=track_history)

        loss = target_char_loss(outputs, sdf)

        loss.backward()
        outer_optimizer.step()
        losses.append(loss.detach().item())
        end_time = time.time_ns()

        print(f"Loss {losses[-1]}")
        print(f"Epoch took {(end_time - start_time) / 1e9: .3f} seconds")

        if track_history:
            viewer.vis_inner_optim(history, target_sdf=sdf, show_edges=False)

    print("Loss values:", losses)

    import ipdb

    ipdb.set_trace()

    # outputs visualisations
    # viewer.vis_outer_targets_optim(
    #     targets_history,
    #     target_sdf=sdf,
    #     video_file=cfg["outer_optim_video_file"],
    # )


if __name__ == "__main__":

    cfg = {
        "seed": 0,
        "device": "cpu",
        "out_video_file": "outputs/swarm/inner_mlp.gif",
        "outer_optim_video_file": "outputs/swarm/outer_targets_mlp.gif",
        "setup": {
            "num_agents": 50,
            "init_std": 1.0,
            "agent_radius": 0.1,
            "collision_radius": 1.0,
            "origin_weight": 0.1,
            "collision_weight": 1.0,
            "gnn_target_weight": 10.0,
            "area_limits": [[-20, -20], [20, 20]],
            "vis_limits": [[-3, -3], [3, 3]],
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
            "num_epochs": 100,
            "lr": 2e-2,
            "optimizer": "adam",
            "target_char": "A",
        },
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    main(cfg)
