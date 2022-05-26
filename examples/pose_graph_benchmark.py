import torch
import theseus as th
import theseus.utils.examples as theg
import hydra
import pathlib
from scipy.io import savemat


# To run this example, you will need the cube datasets available at
# https://drive.google.com/file/d/1rDWMh9PP94D1aYy0ueahBQE_JVD_e88c/view?usp=sharing
#
# The steps below should let you run the example.
# From the root project folder do:
#   mkdir data
#   cd data
#   cp your/path/pose_graph_data.tar.gz .
#   tar -xzvf pose_graph_data.tar.gz
#   cd ..
#   python examples/pose_graph_benchmark.py

DATASET_DIR = pathlib.Path.cwd() / "data" / "pose_graph"

@hydra.main(config_path="./configs/", config_name="pose_graph_benchmark")
def main(cfg):
    dataset_name = cfg.dataset
    file_path = f"{DATASET_DIR}/{dataset_name}_init.g2o"
    dtype = torch.float64

    th.SO3.SO3_EPS = 1e-6

    num_verts, verts, edges = theg.pose_graph.read_3D_g2o_file(file_path, dtype=dtype)
    d = 3

    objective = th.Objective(dtype)

    loss_function = th.TrivialLoss()

    for edge in edges:
        cost_func = th.eb.Between(
            verts[edge.i],
            verts[edge.j],
            edge.weight,
            edge.relative_pose,
            loss_function=loss_function,
        )
        objective.add(cost_func, use_batches=True)

    pose_prior = th.eb.VariableDifference(
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
    results["objective"] = objective.function_value().detach().cpu().numpy().sum() / 2
    results["R"] = torch.cat(
        [pose.data[:, :, :d].detach().cpu() for pose in verts]
    ).numpy()
    results["t"] = torch.cat(
        [pose.data[:, :, d].detach().cpu() for pose in verts]
    ).numpy()

    savemat(dataset_name + ".mat", results)


if __name__ == "__main__":
    main()
