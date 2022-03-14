import torch
import theseus as th

import theseus.utils.examples as theg

torch.manual_seed(1)

file_path = "./datasets/tinyGrid3D.g2o"

num_verts, verts, edges = theg.pose_graph.read_3D_g2o_file(file_path)

objective = th.Objective(torch.float64)
