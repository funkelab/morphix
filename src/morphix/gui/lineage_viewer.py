from datetime import datetime

import numpy as np
import pygfx as gfx
import pylinalg as la  # pylinalg is a pygfx dependency
from PIL import Image
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
from tqdm import tqdm

from ..cell import Cell
from .scene import Scene


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
        lineages: Cell | list[Cell],
        delta_t: float,
        inactive_cell_opacity: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        if isinstance(lineages, Cell):
            lineages = [lineages]
        self.lineages = lineages
        self.n_timesteps = lineages[0].position.shape[0]
        self.current_t = 0
        self.current_lineage = 0
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
        self.t_slider = QSlider(Qt.Horizontal)
        self.t_slider.setMinimum(0)
        self.t_slider.setMaximum(max(0, self.n_timesteps - 1))
        self.t_slider.setValue(0)
        self.t_slider.setSingleStep(1)
        self.t_slider.valueChanged.connect(self.on_t_slider_changed)

        self.lineage_label = QLabel("lineage 0")
        self.lineage_slider = QSlider(Qt.Horizontal)
        self.lineage_slider.setMinimum(0)
        self.lineage_slider.setMaximum(len(self.lineages))
        self.lineage_slider.setValue(0)
        self.lineage_slider.setSingleStep(1)
        self.lineage_slider.valueChanged.connect(self.on_lineage_slider_changed)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.t_label)
        controls.addWidget(self.t_slider, stretch=1)
        layout.addLayout(controls)
        controls = QHBoxLayout()
        controls.addWidget(self.lineage_label)
        controls.addWidget(self.lineage_slider, stretch=1)
        layout.addLayout(controls)

        self.timer = QTimer()
        self.timer.setInterval(200)  # ms per frame
        self.timer.timeout.connect(self.advance_frame)

        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.camera = gfx.PerspectiveCamera(60, 16 / 9)
        self.camera.local.position = (0, 0, 10)
        self.camera.look_at((0, 0, 0))
        self.set_controller("fly")

        self.scenes = [
            Scene(
                lineage,
                delta_t=delta_t,
                inactive_cell_opacity=inactive_cell_opacity,
                grid_y=-5.0,
            )
            for lineage in self.lineages
        ]
        self.active_scene = self.scenes[0]
        self.canvas.request_draw(
            lambda: self.renderer.render(self.active_scene, self.camera)
        )

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
        elif event["key"] == "Q":
            self.advance_lineage(-1)
        elif event["key"] == "E":
            self.advance_lineage(1)
        elif event["key"] == "b":
            self.toggle_bg()
        elif event["key"] == "p":
            self.take_screenshot()
        elif event["key"] == "v":
            self.create_video()
        elif event["key"] == "c":
            self.reset_camera()
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
        bb_min, bb_max = self.active_scene.bounding_box
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
        self.t_slider.setValue(t)

    def advance_lineage(self, delta=1):
        lineage = (self.current_lineage + delta) % len(self.lineages)
        self.lineage_slider.setValue(lineage)

    def set_highlight(self, index=None):
        self.active_scene.set_highlight(index)
        self.canvas.update()

    def set_hover(self, origin, direction):
        self.active_scene.set_hover(origin, direction)
        self.canvas.update()

    def toggle_bg(self):
        self.active_scene.toggle_bg()
        self.canvas.update()

    def take_screenshot(self, filename="screenshot.png"):
        image = Image.fromarray(self.renderer.snapshot())
        image.save(f"screenshot_{self.timestamp()}.png")

    def create_video(self):
        frames = []
        filename = f"frames_{self.timestamp()}"

        print("capturing frames...")
        for lineage in tqdm(range(len(self.lineages))):
            self.lineage_slider.setValue(lineage)
            for t in tqdm(range(self.n_timesteps), leave=False):
                self.t_slider.setValue(t)
                self.renderer.render(self.active_scene, self.camera)
                frame = Image.fromarray(self.renderer.snapshot())
                frames.append(frame)

        print(f"saving frames to {filename}...")
        np.save(filename, np.array(frames))
        print("...done.")

    def on_t_slider_changed(self, t):
        t = int(t)
        if t == self.current_t:
            return
        self.current_t = t
        self.t_label.setText(f"t = {self.current_t} / {self.n_timesteps - 1}")
        self.active_scene.set_timestep(self.current_t)
        self.canvas.update()

    def on_lineage_slider_changed(self, lineage):
        lineage = int(lineage)
        if lineage == self.current_lineage:
            return
        if lineage >= len(self.lineages):
            return
        self.current_lineage = lineage
        self.lineage_label.setText(f"lineage {self.current_lineage}")
        self.active_scene = self.scenes[self.current_lineage]
        self.active_scene.set_timestep(self.current_t)
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

    def timestamp(self):
        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
