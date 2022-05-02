import theseus as th
import torch
import theseus.utils.examples as theg
from typing import List

from theseus.utils.examples.pose_graph.dataset import PoseGraphEdge

use_batches = True
device = "cuda"
dataset_size = 128
num_poses = 1024

poses: List[th.SE3] = []
edges: List[PoseGraphEdge] = []

for n in range(16):
    num_poses, poses_n, edges_n = theg.pose_graph.read_3D_g2o_file(
        f"/private/home/taoshaf/Documents/theseus/datasets/cube/{1024}_poses_0.2_cube_{0}.g2o"
    )
    if len(poses) == 0:
        poses = poses_n
        edges = edges_n
    else:
        for pose, pose_n in zip(poses, poses_n):
            pose.data = torch.cat((pose.data, pose_n.data))

        for edge, edge_n in zip(edges, edges_n):
            edge.relative_pose.data = torch.cat(
                (edge.relative_pose.data, edge_n.relative_pose.data)
            )

pg_dataset = theg.PoseGraphDataset(poses=poses, edges=edges, batch_size=16)


robust_loss = th.TrivialLoss()
objective = th.Objective(dtype=torch.float64)
pose_indices: List[int] = [index for index, _ in enumerate(poses)]

for pose in poses:
    pose.to(device)

for edge in edges:
    relative_pose_cost = th.eb.Between(
        poses[edge.i],
        poses[edge.j],
        edge.weight,
        edge.relative_pose,
        loss_function=robust_loss,
    )
    objective.add(relative_pose_cost, use_batches=use_batches)

pose_prior_cost = th.eb.VariableDifference(
    var=poses[0],
    cost_weight=th.ScaleCostWeight(
        torch.tensor(1e-4, dtype=torch.float64, device=device)
    ),
    target=poses[0].copy(new_name=poses[0].name + "__PRIOR"),
)

objective.to(device)

objective.add(pose_prior_cost, use_batches=use_batches)
optimizer = th.GaussNewton(
    objective,
    max_iterations=10,
    step_size=1,
    abs_err_tolerance=0,
    rel_err_tolerance=0,
    linearization_cls=th.SparseLinearization,
    linear_solver_cls=th.LUCudaSparseSolver,
)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
optimizer.optimize(verbose=True)
end_event.record()

torch.cuda.synchronize()
forward_time = start_event.elapsed_time(end_event)

print(forward_time)
forward_time = start_event.elapsed_time(end_event)
