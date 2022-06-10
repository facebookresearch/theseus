# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import theseus as th
import theseus.utils.examples as theg
import hydra
from scipy.io import savemat


@hydra.main(config_path="./configs/", config_name="pose_graph_benchmark")
def main(cfg):
    dataset_name = cfg.dataset
    file_path = (
        f"/private/home/taoshaf/Documents/theseus/datasets/{dataset_name}_init.g2o"
    )
    dtype = torch.float64

    th.SO3.SO3_EPS = 1e-6

    _, verts, edges = theg.pose_graph.read_3D_g2o_file(file_path, dtype=dtype)
    d = 3

    objective = th.Objective(dtype)

    for edge in edges:
        cost_func = th.eb.Between(
            verts[edge.i],
            verts[edge.j],
            edge.weight,
            edge.relative_pose,
        )
        objective.add(cost_func)

    pose_prior = th.Difference(
        var=verts[0],
        cost_weight=th.ScaleCostWeight(torch.tensor(0 * 1e-6, dtype=dtype)),
        target=verts[0].copy(new_name=verts[0].name + "PRIOR"),
    )
    objective.add(pose_prior)

    optimizer = th.GaussNewton(  # GaussNewton(
        objective,
        max_iterations=10,
        step_size=1.0,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
    )

    inputs = {var.name: var.data for var in verts}
    optimizer.objective.update(inputs)
    optimizer.optimize(verbose=True)

    results = {}
    results["objective"] = (
        objective.error_squared_norm().detach().cpu().numpy().sum() / 2
    )
    results["R"] = torch.cat(
        [pose.data[:, :, :d].detach().cpu() for pose in verts]
    ).numpy()
    results["t"] = torch.cat(
        [pose.data[:, :, d].detach().cpu() for pose in verts]
    ).numpy()

    savemat(dataset_name + ".mat", results)


if __name__ == "__main__":
    main()
