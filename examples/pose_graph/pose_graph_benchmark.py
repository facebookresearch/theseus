# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import theseus as th
import theseus.utils.examples as theg
import logging
import hydra
from scipy.io import savemat

# Logger
log = logging.getLogger(__name__)


@hydra.main(config_path="../configs/pose_graph", config_name="pose_graph_benchmark")
def main(cfg):
    dataset_name = cfg.dataset
    file_path = (
        f"/private/home/taoshaf/Documents/theseus/datasets/{dataset_name}_init.g2o"
    )
    dtype = torch.float32

    _, verts, edges = theg.pose_graph.read_3D_g2o_file(file_path, dtype=torch.float64)
    d = 3

    # _, verts_d, edges_d = theg.pose_graph.read_3D_g2o_file(file_path, dtype=torch.float64)
    # d = 3
    # verts_err = torch.cat([verts[i].data - verts_d[i].data for i in range(len(verts))], dim=0)

    objective = th.Objective(torch.float64)

    for edge in edges:
        cost_func = th.Between(
            verts[edge.i],
            verts[edge.j],
            edge.relative_pose,
            edge.weight,
        )
        objective.add(cost_func)

    pose_prior = th.Difference(
        var=verts[0],
        target=verts[0].copy(new_name=verts[0].name + "PRIOR"),
        cost_weight=th.ScaleCostWeight(torch.tensor(0 * 1e-6, dtype=torch.float64)),
    )
    objective.add(pose_prior)

    objective.to(dtype)
    optimizer = th.GaussNewton(
        objective,
        max_iterations=10,
        step_size=1,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
        vectorize=True,
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    inputs = {var.name: var.tensor for var in verts}
    optimizer.objective.update(inputs)

    start_event.record()
    torch.cuda.reset_peak_memory_stats()
    optimizer.optimize(verbose=True)
    end_event.record()

    torch.cuda.synchronize()
    forward_time = start_event.elapsed_time(end_event)
    forward_mem = torch.cuda.max_memory_allocated() / 1048576
    log.info(f"Forward pass took {forward_time} ms.")
    log.info(f"Forward pass used {forward_mem} MBs.")

    results = {}
    results["objective"] = (
        objective.error_squared_norm().detach().cpu().numpy().sum() / 2
    )
    results["R"] = torch.cat(
        [pose.tensor[:, :, :d].detach().cpu() for pose in verts]
    ).numpy()
    results["t"] = torch.cat(
        [pose.tensor[:, :, d].detach().cpu() for pose in verts]
    ).numpy()

    savemat(dataset_name + ".mat", results)


if __name__ == "__main__":
    main()
