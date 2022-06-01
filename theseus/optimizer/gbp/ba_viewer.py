import threading

import numpy as np
import pyglet
import torch
import trimesh
import trimesh.viewer

import theseus as th


def draw_camera(
    transform, fov, resolution, color=(0.0, 1.0, 0.0, 0.8), marker_height=12.0
):
    camera = trimesh.scene.Camera(fov=fov, resolution=resolution)
    marker = trimesh.creation.camera_marker(camera, marker_height=marker_height)
    marker[0].apply_transform(transform)
    marker[1].apply_transform(transform)
    marker[1].colors = (color,) * len(marker[1].entities)

    return marker


class BAViewer(trimesh.viewer.SceneViewer):
    def __init__(
        self, belief_history, msg_history=None, cam_to_world=False, flip_z=True
    ):
        self._it = 0
        self.belief_history = belief_history
        self.msg_history = msg_history
        self.cam_to_world = cam_to_world
        self.flip_z = flip_z
        self.lock = threading.Lock()

        scene = trimesh.Scene()
        self.scene = scene
        self.next_iteration()
        scene.set_camera()
        super(BAViewer, self).__init__(scene=scene, resolution=(1080, 720))

    def on_key_press(self, symbol, modifiers):
        """
        Call appropriate functions given key presses.
        """
        magnitude = 10
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()
        elif symbol == pyglet.window.key.A:
            self.toggle_axis()
        elif symbol == pyglet.window.key.G:
            self.toggle_grid()
        elif symbol == pyglet.window.key.Q:
            self.on_close()
        elif symbol == pyglet.window.key.M:
            self.maximize()
        elif symbol == pyglet.window.key.F:
            self.toggle_fullscreen()
        elif symbol == pyglet.window.key.P:
            print(self.scene.camera_transform)
        elif symbol == pyglet.window.key.N:
            if self._it + 1 in self.belief_history:
                self._it += 1
                print("Iteration", self._it)
                self.next_iteration()
            else:
                print("No more iterations to view")

        if symbol in [
            pyglet.window.key.LEFT,
            pyglet.window.key.RIGHT,
            pyglet.window.key.DOWN,
            pyglet.window.key.UP,
        ]:
            self.view["ball"].down([0, 0])
            if symbol == pyglet.window.key.LEFT:
                self.view["ball"].drag([-magnitude, 0])
            elif symbol == pyglet.window.key.RIGHT:
                self.view["ball"].drag([magnitude, 0])
            elif symbol == pyglet.window.key.DOWN:
                self.view["ball"].drag([0, -magnitude])
            elif symbol == pyglet.window.key.UP:
                self.view["ball"].drag([0, magnitude])
            self.scene.camera_transform[...] = self.view["ball"].pose

    def next_iteration(self):
        with self.lock:
            points = []
            n_cams, n_pts = 0, 0
            for belief in self.belief_history[self._it]:
                if isinstance(belief.mean[0], th.SE3):
                    T = torch.vstack(
                        (
                            belief.mean[0].data[0],
                            torch.tensor(
                                [[0.0, 0.0, 0.0, 1.0]], dtype=belief.mean[0].dtype
                            ),
                        )
                    )
                    if not self.cam_to_world:
                        T = np.linalg.inv(T)
                    if self.flip_z:
                        T[:3, 2] *= -1.0
                    camera = draw_camera(
                        T, self.scene.camera.fov, self.scene.camera.resolution
                    )
                    self.scene.delete_geometry(f"cam_{n_cams}")
                    self.scene.add_geometry(camera, geom_name=f"cam_{n_cams}")
                    n_cams += 1
                elif isinstance(belief.mean[0], th.Point3):
                    point = belief.mean[0].data
                    points.append(point)

                    cov = torch.linalg.inv(belief.precision[0])
                    ellipse = make_ellipse(point[0], cov)
                    ellipse.visual.vertex_colors[:] = [255, 0, 0, 100]

                    self.scene.delete_geometry(f"ellipse_{n_pts}")
                    self.scene.add_geometry(ellipse, geom_name=f"ellipse_{n_pts}")

            points = torch.cat(points)
            points_tm = trimesh.PointCloud(points)
            self.scene.delete_geometry("points")
            self.scene.add_geometry(points_tm, geom_name="points")

            if self.msg_history:
                for msg in self.msg_history[self._it]:
                    if msg.precision.count_nonzero() != 0:
                        if msg.mean[0].dof() == 3 and "Reprojection" in msg.name:
                            ellipse = make_ellipse(
                                msg.mean[0][0], torch.linalg.inv(msg.precision[0])
                            )
                            if f"ellipse_{msg.name}" in self.scene.geometry:
                                self.scene.delete_geometry(f"ellipse_{msg.name}")
                            self.scene.add_geometry(
                                ellipse, geom_name=f"ellipse_{msg.name}"
                            )

            if self._it != 0:
                self._update_vertex_list()


def make_ellipse(mean, cov, do_lines=False):
    # eigvals_torch, eigvecs_torch = torch.linalg.eigh(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)  # eigenvecs are columns
    # print("eigvals", eigvals)  # , eigvals_torch.numpy())
    eigvals = eigvals / 10
    signs = np.sign(eigvals)
    eigvals = np.clip(np.abs(eigvals), 1.0, 100, eigvals) * signs

    if do_lines:
        points = []
        for i, eigvalue in enumerate(eigvals):
            disp = eigvalue * eigvecs[:, i]
            points.extend([mean + disp, mean - disp])

        paths = torch.cat(points).reshape(3, 2, 3)
        lines = trimesh.load_path(paths)

        return lines

    else:
        rotation = np.eye(4)
        rotation[:3, :3] = eigvecs

        ellipse = trimesh.creation.icosphere()
        ellipse.apply_scale(eigvals)
        ellipse.apply_transform(rotation)
        ellipse.apply_translation(mean)
        ellipse.visual.vertex_colors = trimesh.visual.random_color()
        ellipse.visual.vertex_colors[:, 3] = 100

        return ellipse
