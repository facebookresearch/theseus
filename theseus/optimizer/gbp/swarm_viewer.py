import numpy as np
import shutil
import os
import pygame


class SwarmViewer:
    def __init__(
        self,
        agent_radius,
        collision_radius,
    ):
        self.agent_cols = None
        self.scale = 100
        self.agent_r_pix = agent_radius * self.scale
        self.collision_radius = collision_radius
        self.square_side = None

        self.range = np.array([[-3, -3], [3, 3]])
        self.h = (self.range[1, 1] - self.range[0, 1]) * self.scale
        self.w = (self.range[1, 0] - self.range[0, 0]) * self.scale

        pygame.init()
        pygame.display.set_caption("Swarm")
        self.myfont = pygame.font.SysFont("Jokerman", 40)
        self.screen = pygame.display.set_mode([self.w, self.h])

    def vis_inner_optim(
        self,
        state_history,
        targets=None,
        show_edges=True,
        video_file=None,
    ):
        self.state_history = state_history
        self.t = 0
        self.num_iters = (~list(state_history.values())[0].isinf()[0, 0]).sum()

        self.video_file = video_file
        if self.video_file is not None:
            self.tmp_dir = "/".join(self.video_file.split("/")[:-1]) + "/tmp"
            self.save_ix = 0
            os.mkdir(self.tmp_dir)

        self.targets = targets
        self.show_edges = show_edges
        self.width = 0

        self.run()

    def vis_outer_targets_optim(
        self,
        targets_history,
        square_side=None,
        video_file=None,
    ):
        self.state_history = targets_history
        self.t = 0
        self.num_iters = list(targets_history.values())[0].shape[-1]

        self.video_file = video_file
        if self.video_file is not None:
            self.tmp_dir = "/".join(self.video_file.split("/")[:-1]) + "/tmp"
            self.save_ix = 0
            os.mkdir(self.tmp_dir)

        self.targets = None
        self.show_edges = False
        self.width = 3
        self.square_side = square_side

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

            # draw agents
            for i, state in enumerate(self.state_history.values()):
                pos = state[0, :, self.t].cpu().numpy()
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

            # draw targets
            if self.targets is not None:
                for i, state in enumerate(self.targets.values()):
                    centre = self.pos_to_canvas(state[0].detach().cpu().numpy())
                    pygame.draw.circle(
                        self.screen, self.agent_cols[i], centre, self.agent_r_pix, 3
                    )

            # draw square
            if self.square_side is not None:
                side = self.square_side * self.scale
                left = (self.w - side) / 2
                top = (self.h - side) / 2
                pygame.draw.rect(self.screen, (0, 100, 255), (left, top, side, side), 3)

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
                    " -lavfi paletteuse {self.video_file}"
                )
            else:
                raise ValueError("video file must be either mp4 or gif.")
            shutil.rmtree(self.tmp_dir)
            self.t += 1

    def pos_to_canvas(self, pos):
        x = (pos - self.range[0]) / (self.range[1] - self.range[0])
        return x * np.array([self.h, self.w])

    def save_image(self):
        fname = self.tmp_dir + f"/{self.save_ix:06d}.png"
        pygame.image.save(self.screen, fname)
        self.save_ix += 1
