import numpy as np
import shutil
import os
import torch

import pygame


class SwarmViewer:
    def __init__(
        self,
        collision_radius,
        area_limits,
    ):
        self.agent_cols = None
        self.scale = 100
        self.agent_r_pix = collision_radius / 20 * self.scale
        self.collision_radius = collision_radius
        self.target_sdf = None

        self.range = np.array(area_limits)
        self.h = (self.range[1, 1] - self.range[0, 1]) * self.scale
        self.w = (self.range[1, 0] - self.range[0, 0]) * self.scale

        self.video_file = None

        pygame.init()
        pygame.display.set_caption("Swarm")
        self.myfont = pygame.font.SysFont("Jokerman", 40)
        self.screen = pygame.display.set_mode([self.w, self.h])

    def vis_target_step(
        self,
        targets_history,
        target_sdf,
    ):
        self.state_history = targets_history
        self.t = (~list(targets_history.values())[0].isinf()[0, 0]).sum().item() - 1
        self.num_iters = self.t + 1

        self.targets = None
        self.show_edges = False
        self.width = 3
        self.target_sdf = target_sdf

        self.draw_next()

    def prepare_video(self, video_file):
        self.video_file = video_file
        if self.video_file is not None:
            self.tmp_dir = "/".join(self.video_file.split("/")[:-1]) + "/tmp"
            self.save_ix = 0
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.mkdir(self.tmp_dir)

    def vis_inner_optim(
        self,
        history,
        target_sdf=None,
        show_edges=True,
        video_file=None,
    ):
        self.prepare_video(video_file)

        self.state_history = {k: v for k, v in history.items() if "agent" in k}
        self.target_history = {k: v for k, v in history.items() if "target" in k}

        self.t = 0
        self.num_iters = (~list(self.state_history.values())[0].isinf()[0, 0]).sum()

        self.show_edges = show_edges
        self.width = 0
        self.target_sdf = target_sdf

        self.run()

    def vis_outer_targets_optim(
        self,
        targets_history,
        target_sdf=None,
        video_file=None,
    ):
        self.state_history = targets_history
        self.t = 0
        self.num_iters = list(targets_history.values())[0].shape[-1]

        self.video_file = video_file
        if self.video_file is not None:
            self.tmp_dir = "/".join(self.video_file.split("/")[:-1]) + "/tmp"
            self.save_ix = 0
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.mkdir(self.tmp_dir)

        self.targets = None
        self.show_edges = False
        self.width = 3
        self.target_sdf = target_sdf

        self.run()

    def run(self):
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

            # draw target shape as background
            if self.target_sdf is not None:
                sdf = self.target_sdf.sdf_data.tensor[0].transpose(0, 1)
                sdf_size = self.target_sdf.cell_size.tensor.item() * sdf.shape[0]
                area_size = self.range[1, 0] - self.range[0, 0]
                crop = np.round((1 - area_size / sdf_size) * sdf.shape[0] / 2).astype(
                    int
                )
                sdf = sdf[crop:-crop, crop:-crop]
                sdf = torch.flip(
                    sdf, [1]
                )  # flip vertically so y is increasing going up
                repeats = self.screen.get_width() // sdf.shape[0]
                sdf = torch.repeat_interleave(sdf, repeats, dim=0)
                sdf = torch.repeat_interleave(sdf, repeats, dim=1)
                sdf = sdf.detach().cpu().numpy()
                bg_img = np.zeros([*sdf.shape, 3])
                bg_img[sdf > 0] = 255
                bg_img[sdf <= 0] = [144, 238, 144]
                bg = pygame.surfarray.make_surface(bg_img)
                self.screen.blit(bg, (0, 0))

            # draw agents
            for i, state in enumerate(self.state_history.values()):
                pos = state[0, :, self.t].detach().cpu().numpy()
                centre = self.pos_to_canvas(pos)
                pygame.draw.circle(
                    self.screen,
                    self.agent_cols[i],
                    centre,
                    self.agent_r_pix,
                    self.width,
                )

            # draw edges between agents
            if self.show_edges:
                for i, state1 in enumerate(self.state_history.values()):
                    pos1 = state1[0, :, self.t].detach().cpu().numpy()
                    j = 0
                    for state2 in self.state_history.values():
                        if j <= i:
                            j += 1
                            continue
                        pos2 = state2[0, :, self.t].detach().cpu().numpy()
                        dist = np.linalg.norm(pos1 - pos2)
                        if dist < self.collision_radius:
                            start = self.pos_to_canvas(pos1)
                            end = self.pos_to_canvas(pos2)
                            pygame.draw.line(self.screen, (0, 0, 0), start, end)

            # draw agents
            for i, state in enumerate(self.target_history.values()):
                pos = state[0, :, self.t].detach().cpu().numpy()
                centre = self.pos_to_canvas(pos)
                pygame.draw.circle(
                    self.screen, self.agent_cols[i], centre, self.agent_r_pix, 3
                )

            # draw line between agent and target

            # draw text
            ssshow = self.myfont.render(
                f"t = {self.t} / {self.num_iters - 1}", True, (0, 0, 0)
            )
            self.screen.blit(ssshow, (10, 10))  # choose location of text

            pygame.display.flip()

            if self.video_file:
                self.save_image()

            self.t += 1

        elif self.t == self.num_iters and self.video_file:
            if self.video_file[-3:] == "mp4":
                os.system(
                    f"ffmpeg -r 4 -i {self.tmp_dir}/%06d.png -vcodec mpeg4 -y {self.video_file}"
                )
            elif self.video_file[-3:] == "gif":
                os.system(
                    f"ffmpeg -i {self.tmp_dir}/%06d.png -vf palettegen {self.tmp_dir}/palette.png"
                )
                os.system(
                    f"ffmpeg -r 4 -i {self.tmp_dir}/%06d.png -i {self.tmp_dir}/palette.png"
                    f" -lavfi paletteuse {self.video_file}"
                )
            else:
                raise ValueError("video file must be either mp4 or gif.")
            shutil.rmtree(self.tmp_dir)
            self.t += 1

    def pos_to_canvas(self, pos):
        x = (pos - self.range[0]) / (self.range[1] - self.range[0])
        x[1] = 1 - x[1]
        return x * np.array([self.h, self.w])

    def save_image(self):
        fname = self.tmp_dir + f"/{self.save_ix:06d}.png"
        pygame.image.save(self.screen, fname)
        self.save_ix += 1
