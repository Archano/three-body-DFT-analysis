import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button

from physicsEngine import simSetup, simulationStep


SYSTEM_PRESETS = {
    "Earth-Moon": {
        "body1_mass": 5.972e24,
        "body2_mass": 7.348e22,
    },
    "Sun-Earth": {
        "body1_mass": 1.989e30,
        "body2_mass": 5.972e24,
    },
    "Sun-Jupiter": {
        "body1_mass": 1.989e30,
        "body2_mass": 1.898e27,
    },
    "Jupiter-Europa": {
        "body1_mass": 1.898e27,
        "body2_mass": 4.799e22,
    },
    "Saturn-Titan": {
        "body1_mass": 5.683e26,
        "body2_mass": 1.345e23,
    },
    "Pluto-Charon": {
        "body1_mass": 1.303e22,
        "body2_mass": 1.586e21,
    },
}


def lagrange_points(mu):
    l4 = (0.5 - mu, math.sqrt(3) / 2)
    l5 = (0.5 - mu, -math.sqrt(3) / 2)
    return l4, l5


class CR3BPGUI:
    def __init__(self):
        self.current_system_name = "Earth-Moon"

        self.default_body1_mass = SYSTEM_PRESETS[self.current_system_name]["body1_mass"]
        self.default_body2_mass = SYSTEM_PRESETS[self.current_system_name]["body2_mass"]

        self.body1_mass = self.default_body1_mass
        self.body2_mass = self.default_body2_mass

        self.mu, self.body1_x, self.body2_x = simSetup(
            self.body1_mass,
            self.body2_mass
        )

        self.l4, self.l5 = lagrange_points(self.mu)

        self.state = [
            self.l4[0] + 0.01,
            self.l4[1],
            0.0,
            0.0
        ]

        self.timestep = 0.001
        self.steps_per_frame = 20
        self.time = 0.0
        self.running = False

        self.x_history = [self.state[0]]
        self.y_history = [self.state[1]]

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.subplots_adjust(left=0.34, right=0.97, bottom=0.08, top=0.93)

        self.setup_plot()
        self.setup_widgets()

        self.animation = FuncAnimation(
            self.fig,
            self.update_frame,
            interval=20,
            blit=False,
            cache_frame_data=False
        )

    def setup_plot(self):
        self.ax.set_title("Circular Restricted Three-Body Problem")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)
        self.ax.axis("equal")

        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-1.2, 1.2)

        self.body1_marker = self.ax.scatter(
            [self.body1_x], [0.0], s=160, label="Body 1"
        )
        self.body2_marker = self.ax.scatter(
            [self.body2_x], [0.0], s=80, label="Body 2"
        )

        self.l4_marker = self.ax.scatter(
            [self.l4[0]], [self.l4[1]], marker="x", s=100, label="L4"
        )
        self.l5_marker = self.ax.scatter(
            [self.l5[0]], [self.l5[1]], marker="x", s=100, label="L5"
        )

        self.trajectory_line, = self.ax.plot(
            self.x_history,
            self.y_history,
            linewidth=1.0,
            label="Trajectory"
        )

        self.particle_dot, = self.ax.plot(
            [self.state[0]],
            [self.state[1]],
            marker="o",
            markersize=6,
            linestyle="None",
            label="Particle"
        )

        self.status_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top"
        )

        self.ax.legend(loc="upper right")

    def setup_widgets(self):
        x0 = 0.03
        width = 0.24
        height = 0.032
        gap = 0.040

        y = 0.91

        self.body1_box = self.add_textbox(
            x0, y, width, height, "Body 1 mass", f"{self.body1_mass:.6e}"
        )
        y -= gap

        self.body2_box = self.add_textbox(
            x0, y, width, height, "Body 2 mass", f"{self.body2_mass:.6e}"
        )
        y -= gap

        self.x_box = self.add_textbox(x0, y, width, height, "x0", f"{self.state[0]:.8f}")
        y -= gap
        self.y_box = self.add_textbox(x0, y, width, height, "y0", f"{self.state[1]:.8f}")
        y -= gap
        self.vx_box = self.add_textbox(x0, y, width, height, "vx0", f"{self.state[2]:.8f}")
        y -= gap
        self.vy_box = self.add_textbox(x0, y, width, height, "vy0", f"{self.state[3]:.8f}")
        y -= gap

        self.timestep_box = self.add_textbox(x0, y, width, height, "timestep", f"{self.timestep}")
        y -= gap
        self.steps_box = self.add_textbox(x0, y, width, height, "steps/frame", f"{self.steps_per_frame}")
        y -= gap

        self.run_button = self.add_button(x0, y, width, height, "Run / Pause")
        self.run_button.on_clicked(self.toggle_running)
        y -= gap

        self.reset_button = self.add_button(x0, y, width, height, "Apply / Reset")
        self.reset_button.on_clicked(self.apply_reset)
        y -= gap

        self.l4_button = self.add_button(x0, y, width, height, "Set near L4")
        self.l4_button.on_clicked(self.set_near_l4)
        y -= gap

        self.l5_button = self.add_button(x0, y, width, height, "Set near L5")
        self.l5_button.on_clicked(self.set_near_l5)
        y -= gap

        self.clear_button = self.add_button(x0, y, width, height, "Clear trail")
        self.clear_button.on_clicked(self.clear_trail)
        y -= gap * 1.2

        self.system_buttons = {}
        for system_name in SYSTEM_PRESETS:
            button = self.add_button(x0, y, width, height, system_name)
            button.on_clicked(self.make_system_callback(system_name))
            self.system_buttons[system_name] = button
            y -= gap

    def add_textbox(self, x, y, width, height, label, initial):
        ax_box = self.fig.add_axes([x, y, width, height])
        return TextBox(ax_box, label, initial=initial)

    def add_button(self, x, y, width, height, label):
        ax_button = self.fig.add_axes([x, y, width, height])
        return Button(ax_button, label)

    def parse_float(self, box, fallback):
        try:
            return float(box.text)
        except ValueError:
            return fallback

    def parse_int(self, box, fallback):
        try:
            value = int(float(box.text))
            return max(1, value)
        except ValueError:
            return fallback

    def read_live_simulation_parameters(self):
        self.timestep = self.parse_float(self.timestep_box, self.timestep)
        self.steps_per_frame = self.parse_int(self.steps_box, self.steps_per_frame)

    def update_primary_positions(self):
        self.mu, self.body1_x, self.body2_x = simSetup(
            self.body1_mass,
            self.body2_mass
        )

        self.l4, self.l5 = lagrange_points(self.mu)

        self.body1_marker.set_offsets([[self.body1_x, 0.0]])
        self.body2_marker.set_offsets([[self.body2_x, 0.0]])
        self.l4_marker.set_offsets([[self.l4[0], self.l4[1]]])
        self.l5_marker.set_offsets([[self.l5[0], self.l5[1]]])

    def apply_reset(self, event=None):
        self.body1_mass = self.parse_float(self.body1_box, self.body1_mass)
        self.body2_mass = self.parse_float(self.body2_box, self.body2_mass)

        self.update_primary_positions()

        x0 = self.parse_float(self.x_box, self.state[0])
        y0 = self.parse_float(self.y_box, self.state[1])
        vx0 = self.parse_float(self.vx_box, self.state[2])
        vy0 = self.parse_float(self.vy_box, self.state[3])

        self.state = [x0, y0, vx0, vy0]
        self.time = 0.0

        self.x_history = [self.state[0]]
        self.y_history = [self.state[1]]

        self.update_artists()

    def set_near_l4(self, event=None):
        self.body1_mass = self.parse_float(self.body1_box, self.body1_mass)
        self.body2_mass = self.parse_float(self.body2_box, self.body2_mass)

        self.update_primary_positions()

        x0 = self.l4[0] + 0.01
        y0 = self.l4[1]
        vx0 = 0.0
        vy0 = 0.0

        self.set_initial_condition_boxes(x0, y0, vx0, vy0)
        self.reset_state(x0, y0, vx0, vy0)

    def set_near_l5(self, event=None):
        self.body1_mass = self.parse_float(self.body1_box, self.body1_mass)
        self.body2_mass = self.parse_float(self.body2_box, self.body2_mass)

        self.update_primary_positions()

        x0 = self.l5[0] + 0.01
        y0 = self.l5[1]
        vx0 = 0.0
        vy0 = 0.0

        self.set_initial_condition_boxes(x0, y0, vx0, vy0)
        self.reset_state(x0, y0, vx0, vy0)

    def set_initial_condition_boxes(self, x0, y0, vx0, vy0):
        self.x_box.set_val(f"{x0:.8f}")
        self.y_box.set_val(f"{y0:.8f}")
        self.vx_box.set_val(f"{vx0:.8f}")
        self.vy_box.set_val(f"{vy0:.8f}")

    def reset_state(self, x0, y0, vx0, vy0):
        self.state = [x0, y0, vx0, vy0]
        self.time = 0.0
        self.x_history = [x0]
        self.y_history = [y0]
        self.update_artists()

    def make_system_callback(self, system_name):
        def callback(event=None):
            self.apply_system_preset(system_name)
        return callback

    def apply_system_preset(self, system_name):
        preset = SYSTEM_PRESETS[system_name]

        self.current_system_name = system_name
        self.body1_mass = preset["body1_mass"]
        self.body2_mass = preset["body2_mass"]

        self.body1_box.set_val(f"{self.body1_mass:.6e}")
        self.body2_box.set_val(f"{self.body2_mass:.6e}")

        self.update_primary_positions()

        # After changing system, place particle near L4 by default.
        x0 = self.l4[0] + 0.01
        y0 = self.l4[1]
        vx0 = 0.0
        vy0 = 0.0

        self.set_initial_condition_boxes(x0, y0, vx0, vy0)
        self.reset_state(x0, y0, vx0, vy0)

    def clear_trail(self, event=None):
        self.x_history = [self.state[0]]
        self.y_history = [self.state[1]]
        self.update_artists()

    def toggle_running(self, event=None):
        self.running = not self.running

    def update_artists(self):
        self.trajectory_line.set_data(self.x_history, self.y_history)
        self.particle_dot.set_data([self.state[0]], [self.state[1]])

        self.status_text.set_text(
            f"system = {self.current_system_name}\n"
            f"t = {self.time:.3f}\n"
            f"mu = {self.mu:.8f}\n"
            f"x = {self.state[0]:.5f}\n"
            f"y = {self.state[1]:.5f}\n"
            f"vx = {self.state[2]:.5f}\n"
            f"vy = {self.state[3]:.5f}"
        )

        self.fig.canvas.draw_idle()

    def update_frame(self, frame):
        if not self.running:
            self.update_artists()
            return

        self.read_live_simulation_parameters()

        for _ in range(self.steps_per_frame):
            self.state, acc_x, acc_y, r1, r2 = simulationStep(
                self.state,
                self.timestep,
                self.mu
            )

            self.time += self.timestep
            self.x_history.append(self.state[0])
            self.y_history.append(self.state[1])

        self.update_artists()

    def show(self):
        plt.show()


if __name__ == "__main__":
    gui = CR3BPGUI()
    gui.show()