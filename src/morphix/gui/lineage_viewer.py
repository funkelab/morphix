import numpy as np
import pygfx as gfx
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.auto import RenderCanvas

from ..cell import Cell
from .scene import Scene

import numpy as np
import pygfx as gfx
import pylinalg as la  # pylinalg is a pygfx dependency


class LineageViewer(QWidget):
    """The main widget of the lineage viewer.

    Args:
        lineage:

            The lineage to show.

        delta_t:

            The time difference between time steps. Used to show the effective
            forces per time step.

        inactive_cell_opacity:

            Set to a value >0 to show inactive cells.

        parent:

            QWidget parent.
    """

    def __init__(
        self,
        lineage: Cell,
        delta_t: float,
        inactive_cell_opacity: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.lineage = lineage
        self.n_timesteps = lineage.position.shape[0]
        self.current_t = 0
        self.playing = False
        self.dark_bg = True

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.canvas = RenderCanvas()
        layout.addWidget(self.canvas, stretch=1)
        self.canvas.add_event_handler(self.on_key_down, "key_down")
        self.canvas.add_event_handler(self.on_pointer_move, "pointer_move")

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_play)

        self.t_label = QLabel(f"t = 0 / {self.n_timesteps - 1}")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.n_timesteps - 1))
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.on_slider_changed)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.t_label)
        controls.addWidget(self.slider, stretch=1)
        layout.addLayout(controls)

        self.timer = QTimer()
        self.timer.setInterval(200)  # ms per frame
        self.timer.timeout.connect(self.advance_frame)

        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.camera = gfx.PerspectiveCamera(60, 16 / 9)
        self.camera.local.position = (0, 0, 10)
        self.camera.look_at((0, 0, 0))
        self.set_controller("fly")

        self.scene = Scene(
            lineage, delta_t=delta_t, inactive_cell_opacity=inactive_cell_opacity
        )
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def set_controller(self, mode: str):
        if mode == "fly":
            self.controller = gfx.FlyController(
                self.camera, register_events=self.renderer
            )
            self.controller.enable_keys = True
            self.controller.controls["mouse1"] = ("rotate", "drag", (0.001, 0.001))
            self.controller.controls["r"] = ("move", "repeat", (0.0, 1.0, 0.0))
            self.controller.controls["f"] = ("move", "repeat", (0.0, -1.0, 0.0))
            del self.controller.controls[" "]
            del self.controller.controls["q"]
            del self.controller.controls["e"]
            del self.controller.controls["shift"]
        elif mode == "orbit":
            self.controller = gfx.OrbitController(
                self.camera, target=(0, 0, 0), register_events=self.renderer
            )

    def on_key_down(self, event):
        if event["key"] == " ":
            self.play_button.click()
        elif event["key"] == "q":
            self.advance_frame(-1)
        elif event["key"] == "e":
            self.advance_frame(1)
        elif event["key"] == "b":
            self.toggle_bg()
        elif event["key"] == "Escape":
            self.set_highlight()

        # is a number?
        try:
            index = int(event["key"])
            if index >= 0 and index < 10:
                self.set_highlight(index)
        except ValueError:
            pass

    def on_pointer_move(self, event):
        origin, direction = self.get_mouse_ray(event["x"], event["y"])
        self.set_hover(origin, direction)

    def reset_camera(self, margin=1.2):
        """Position the camera such that all objects are visible."""
        bb_min, bb_max = self.scene.bounding_box
        center = (bb_min + bb_max) / 2.0
        extent = bb_max - bb_min
        radius = np.linalg.norm(extent) / 2.0 * margin

        if isinstance(self.camera, gfx.PerspectiveCamera):
            # Compute distance from center based on FOV
            fov = np.deg2rad(self.camera.fov)
            distance = radius / np.sin(fov / 2.0)
            self.camera.local.position = (center[0], center[1], center[2] + distance)
            self.camera.look_at(center)
        elif isinstance(self.camera, gfx.OrthographicCamera):
            self.camera.width = extent[0] * margin
            self.camera.height = extent[1] * margin
            self.camera.local.position = (center[0], center[1], center[2] + radius)
            self.camera.look_at(center)

    def toggle_play(self, checked):
        self.playing = bool(checked)
        if self.playing:
            self.play_button.setText("Pause")
            self.timer.start()
        else:
            self.play_button.setText("Play")
            self.timer.stop()

    def advance_frame(self, delta=1):
        t = (self.current_t + delta) % self.n_timesteps
        self.slider.setValue(t)

    def set_highlight(self, index=None):
        self.scene.set_highlight(index)
        self.canvas.update()

    def set_hover(self, origin, direction):
        self.scene.set_hover(origin, direction)
        self.canvas.update()

    def toggle_bg(self):
        self.scene.toggle_bg()
        self.canvas.update()

    def on_slider_changed(self, t):
        t = int(t)
        if t == self.current_t:
            return
        self.current_t = t
        self.t_label.setText(f"t = {self.current_t} / {self.n_timesteps - 1}")
        self.scene.set_timestep(self.current_t)
        self.canvas.update()

    def get_mouse_ray(self, x, y):
        width, height = self.renderer.logical_size
        ndc_x = (x / width) * 2.0 - 1.0
        ndc_y = -((y / height) * 2.0 - 1.0)

        projection = self.camera.projection_matrix
        view = self.camera.view_matrix
        inv_transform = la.mat_inverse(projection @ view)

        near_ndc_z = 0.0  # near plane
        far_ndc_z = 1.0  # far plane

        near_world = self.ndc_to_world(ndc_x, ndc_y, near_ndc_z, inv_transform)
        far_world = self.ndc_to_world(ndc_x, ndc_y, far_ndc_z, inv_transform)

        origin = near_world
        direction = far_world - near_world
        direction = direction / np.linalg.norm(direction)

        return origin, direction

    def ndc_to_world(self, ndc_x, ndc_y, ndc_z, inv_transform):
        world_h = inv_transform @ np.array([ndc_x, ndc_y, ndc_z, 1.0], dtype=np.float32)
        world = world_h[:3] / world_h[3]
        return world
