import threading

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
    def __init__(self, belief_history):
        self._it = 0
        self.belief_history = belief_history
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
                    self.scene.delete_geometry(f"ellipse_{n_pts}")
                    self.scene.add_geometry(ellipse, geom_name=f"ellipse_{n_pts}")

            points = torch.cat(points)
            points_tm = trimesh.PointCloud(points)
            self.scene.delete_geometry("points")
            self.scene.add_geometry(points_tm, geom_name="points")
            if self._it != 0:
                self._update_vertex_list()


def make_ellipse(mean, cov):
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # rescale eigvals into range that fits in scene
    print(eigvals)
    eigvals = eigvals / 10
    eigvals = torch.maximum(torch.tensor(0.7), eigvals)
    eigvals = torch.minimum(torch.tensor(60.0), eigvals)

    rotation = torch.eye(4)
    rotation[:3, :3] = eigvecs

    ellipse = trimesh.creation.icosphere()
    ellipse.apply_scale(eigvals.numpy())
    ellipse.apply_transform(rotation)
    ellipse.apply_translation(mean)

    return ellipse
