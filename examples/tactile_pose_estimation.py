# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pathlib
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

import theseus.utils.examples as theg

# Logger
logger = logging.getLogger(__name__)

# To run this example, you will need a tactile pushing dataset available at
# https://dl.fbaipublicfiles.com/theseus/tactile_pushing_data.tar.gz
#
# The steps below should let you run the example.
# From the root project folder do:
#   mkdir expts
#   cd expts
#   wget https://dl.fbaipublicfiles.com/theseus/tactile_pushing_data.tar.gz
#   tar -xzvf tactile_pushing_data.tar.gz
#   cd ..
#   python examples/tactile_pose_estimation.py
EXP_PATH = pathlib.Path.cwd() / "expts" / "tactile-pushing"
torch.set_default_dtype(torch.double)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

plt.ion()


# In this example, the goal is to track 2D object poses (x,y,theta) during planar
# pushing from tactile image measurements for single episode.
# This can solved as an optimization problem where the variables being estimated
# are object poses over time.
#
# We formulate the optimization using following cost terms:
#   * Quasi-static pushing planar: Penalizes deviation from quasi-static dynamics.
#     Uses velocity-only quasi-static model for sticking contact.
#   * Tactile measurements: Penalizes deviations from predicted relative pose
#     from tactile image feature pairs.
#   * Object-effector intersection: Penalizes intersections between object and
#     end-effector.
#   * End-effector priors: Penalizes deviation from end-effector poses captured from
#     motion capture.
#   * Boundary conditions: Penalizes deviation of first object pose from a global
#     pose prior.
#
# Based on the method described in,
# Sodhi et al. Learning Tactile Models for Factor Graph-based Estimation,
# 2021 (https://arxiv.org/abs/1705.10664)


def run_learning_loop(cfg):
    root_path = pathlib.Path(os.getcwd())
    logger.info(f"LOGGING TO {str(root_path)}")
    trainer = theg.TactilePushingTrainer(cfg, EXP_PATH, device)

    # -------------------------------------------------------------------- #
    # Main learning loop
    # -------------------------------------------------------------------- #
    # Use theseus_layer in an outer learning loop to learn different cost
    # function parameters:
    results_train = {}
    results_val = {}
    for epoch in range(cfg.train.num_epochs):
        logger.info(f" ********************* EPOCH {epoch} *********************")
        logger.info(" -------------- TRAINING --------------")
        train_losses, results_train[epoch], _ = trainer.compute_loss(epoch)
        logger.info(f"AVG. TRAIN LOSS: {np.mean(train_losses)}")
        torch.save(results_train, root_path / "results_train.pt")

        logger.info(" -------------- VALIDATION --------------")
        with torch.no_grad():
            val_losses, results_val[epoch], image_data = trainer.compute_loss(
                epoch, update=False
            )
        logger.info(f"AVG. VAL LOSS: {np.mean(val_losses)}")
        torch.save(results_val, root_path / "results_val.pt")

        if cfg.options.vis_traj:
            for i in range(len(image_data["obj_opt"])):
                save_dir = root_path / f"img_{i}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_fname = save_dir / f"epoch{epoch}.png"
                theg.visualize_tactile_push2d(
                    obj_poses=image_data["obj_opt"][i],
                    eff_poses=image_data["eff_opt"][i],
                    obj_poses_gt=image_data["obj_gt"][i],
                    eff_poses_gt=image_data["eff_gt"][i],
                    rect_len_x=cfg.shape.rect_len_x,
                    rect_len_y=cfg.shape.rect_len_y,
                    save_fname=save_fname,
                )


@hydra.main(config_path="./configs/", config_name="tactile_pose_estimation")
def main(cfg):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_learning_loop(cfg)


if __name__ == "__main__":
    main()
