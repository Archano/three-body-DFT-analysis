import csv
import math
import sys
from datetime import datetime

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import (
        QApplication,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ModuleNotFoundError as exc:
    install_name = "PyQt6" if exc.name.startswith("PyQt6") else exc.name
    raise SystemExit(
        f"{exc.name} is required for the GUI. Install it with: pip install {install_name}"
    ) from exc

from physicsEngine import (
    accelerationX,
    accelerationY,
    particleToBody1And2Distance,
    simSetup,
    simulationStep,
)


SYSTEM_PRESETS = {
    "Earth-Moon": {
        "body1_mass": 5.972e24,
        "body2_mass": 7.348e22,
        "period_days": 27.321661,
    },
    "Sun-Earth": {
        "body1_mass": 1.989e30,
        "body2_mass": 5.972e24,
        "period_days": 365.256,
    },
    "Sun-Jupiter": {
        "body1_mass": 1.989e30,
        "body2_mass": 1.898e27,
        "period_days": 4332.589,
    },
    "Jupiter-Europa": {
        "body1_mass": 1.898e27,
        "body2_mass": 4.799e22,
        "period_days": 3.551181,
    },
    "Saturn-Titan": {
        "body1_mass": 5.683e26,
        "body2_mass": 1.345e23,
        "period_days": 15.945,
    },
    "Pluto-Charon": {
        "body1_mass": 1.303e22,
        "body2_mass": 1.586e21,
        "period_days": 6.387,
    },
}


def lagrange_points(mu):
    l4 = (0.5 - mu, math.sqrt(3) / 2)
    l5 = (0.5 - mu, -math.sqrt(3) / 2)
    return l4, l5


class CR3BPGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_system_name = "Earth-Moon"
        preset = SYSTEM_PRESETS[self.current_system_name]
        self.body1_mass = preset["body1_mass"]
        self.body2_mass = preset["body2_mass"]
        self.primary_period_days = preset["period_days"]

        self.mu, self.body1_x, self.body2_x = simSetup(
            self.body1_mass,
            self.body2_mass,
        )
        self.l4, self.l5 = lagrange_points(self.mu)

        self.state = [self.l4[0] + 0.01, self.l4[1], 0.0, 0.0]
        self.timestep = 0.001
        self.steps_per_update = 250
        self.max_visible_trail_points = 3000
        self.time = 0.0
        self.running = False

        self.batch_mode = False
        self.batch_total_steps = 0
        self.remaining_batch_steps = 0
        self.run_records = []
        self.last_export_filename = None

        self.x_history = [self.state[0]]
        self.y_history = [self.state[1]]

        self.setWindowTitle("CR3BP Simulator")
        self.resize(1450, 850)

        self.setup_ui()
        self.setup_plot()
        self.update_view()

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start()

    def setup_ui(self):
        central_widget = QWidget()
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        controls = self.build_controls()
        root_layout.addWidget(controls, 0)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout.addWidget(self.canvas)

        root_layout.addWidget(plot_panel, 1)
        self.setCentralWidget(central_widget)

    def build_controls(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(410)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.system_box = self.add_line_edit(f"{self.body1_mass:.6e}")
        self.secondary_box = self.add_line_edit(f"{self.body2_mass:.6e}")
        self.period_box = self.add_line_edit(f"{self.primary_period_days:.6f}")

        system_group = QGroupBox("System")
        system_layout = QFormLayout(system_group)
        system_layout.addRow("Primary mass", self.system_box)
        system_layout.addRow("Secondary mass", self.secondary_box)
        system_layout.addRow("Orbit period days", self.period_box)
        layout.addWidget(system_group)

        presets_group = QGroupBox("Presets")
        presets_layout = QGridLayout(presets_group)
        for index, system_name in enumerate(SYSTEM_PRESETS):
            button = QPushButton(system_name)
            button.clicked.connect(
                lambda checked=False, name=system_name: self.apply_system_preset(name)
            )
            presets_layout.addWidget(button, index // 2, index % 2)
        layout.addWidget(presets_group)

        self.x_box = self.add_line_edit(f"{self.state[0]:.8f}")
        self.y_box = self.add_line_edit(f"{self.state[1]:.8f}")
        self.vx_box = self.add_line_edit(f"{self.state[2]:.8f}")
        self.vy_box = self.add_line_edit(f"{self.state[3]:.8f}")

        initial_group = QGroupBox("Initial Particle State")
        initial_layout = QFormLayout(initial_group)
        initial_layout.addRow("x0", self.x_box)
        initial_layout.addRow("y0", self.y_box)
        initial_layout.addRow("vx0", self.vx_box)
        initial_layout.addRow("vy0", self.vy_box)
        layout.addWidget(initial_group)

        self.timestep_box = self.add_line_edit(f"{self.timestep}")
        self.steps_box = self.add_line_edit(f"{self.steps_per_update}")
        self.max_trail_box = self.add_line_edit(f"{self.max_visible_trail_points}")

        sim_group = QGroupBox("Simulation")
        sim_layout = QFormLayout(sim_group)
        sim_layout.addRow("dt per physics step", self.timestep_box)
        sim_layout.addRow("physics steps per redraw", self.steps_box)
        sim_layout.addRow("visible trail points", self.max_trail_box)

        sim_buttons = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.toggle_running)
        self.reset_button = QPushButton("Apply reset")
        self.reset_button.clicked.connect(self.apply_reset)
        sim_buttons.addWidget(self.run_button)
        sim_buttons.addWidget(self.reset_button)
        sim_layout.addRow(sim_buttons)

        point_buttons = QHBoxLayout()
        l4_button = QPushButton("Near L4")
        l4_button.clicked.connect(self.set_near_l4)
        l5_button = QPushButton("Near L5")
        l5_button.clicked.connect(self.set_near_l5)
        point_buttons.addWidget(l4_button)
        point_buttons.addWidget(l5_button)
        sim_layout.addRow(point_buttons)

        clear_button = QPushButton("Clear trail")
        clear_button.clicked.connect(self.clear_trail)
        sim_layout.addRow(clear_button)
        layout.addWidget(sim_group)

        self.batch_steps_box = self.add_line_edit("50000")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(True)

        batch_group = QGroupBox("Batch CSV")
        batch_layout = QFormLayout(batch_group)
        batch_layout.addRow("total physics steps", self.batch_steps_box)

        batch_buttons = QHBoxLayout()
        self.batch_button = QPushButton("Run CSV")
        self.batch_button.clicked.connect(self.start_batch_run)
        self.stop_export_button = QPushButton("Stop + export CSV")
        self.stop_export_button.clicked.connect(self.stop_and_export_run)
        batch_buttons.addWidget(self.batch_button)
        batch_buttons.addWidget(self.stop_export_button)
        batch_layout.addRow(batch_buttons)
        batch_layout.addRow("progress", self.progress_bar)
        layout.addWidget(batch_group)

        hint = QLabel(
            "dt controls simulation accuracy. Steps per redraw controls speed.\n"
            "Higher steps per redraw is faster but less visually smooth.\n"
            "Visible trail only affects plotting, not exported CSV data."
        )
        hint.setWordWrap(True)
        hint.setFrameShape(QFrame.Shape.StyledPanel)
        hint.setStyleSheet("color: #444; padding: 8px;")
        layout.addWidget(hint)

        self.status_label = QLabel()
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.status_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px; padding: 8px;"
        )
        self.status_label.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(self.status_label)

        layout.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def setup_plot(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Circular Restricted Three-Body Problem")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.35)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-1.2, 1.2)

        self.body1_marker = self.ax.scatter(
            [self.body1_x], [0.0], s=160, label="Body 1"
        )
        self.body2_marker = self.ax.scatter(
            [self.body2_x], [0.0], s=80, label="Body 2"
        )
        self.l4_marker = self.ax.scatter(
            [self.l4[0]], [self.l4[1]], marker="x", s=90, label="L4"
        )
        self.l5_marker = self.ax.scatter(
            [self.l5[0]], [self.l5[1]], marker="x", s=90, label="L5"
        )
        self.trajectory_line, = self.ax.plot(
            self.x_history,
            self.y_history,
            linewidth=1.0,
            label="Trajectory",
        )
        self.particle_dot, = self.ax.plot(
            [self.state[0]],
            [self.state[1]],
            marker="o",
            markersize=6,
            linestyle="None",
            label="Particle",
        )
        self.ax.legend(loc="upper right")

    def add_line_edit(self, initial_text):
        line_edit = QLineEdit(initial_text)
        line_edit.setMinimumWidth(145)
        return line_edit

    def parse_float(self, line_edit, fallback):
        try:
            return float(line_edit.text())
        except ValueError:
            return fallback

    def parse_int(self, line_edit, fallback):
        try:
            value = int(float(line_edit.text()))
            return max(1, value)
        except ValueError:
            return fallback

    def read_simulation_parameters(self):
        self.timestep = self.parse_float(self.timestep_box, self.timestep)
        self.steps_per_update = self.parse_int(self.steps_box, self.steps_per_update)
        self.max_visible_trail_points = self.parse_int(
            self.max_trail_box,
            self.max_visible_trail_points,
        )
        self.primary_period_days = self.parse_float(
            self.period_box,
            self.primary_period_days,
        )

    def sim_time_to_real_days(self, sim_time):
        if self.primary_period_days <= 0:
            return None
        return sim_time * self.primary_period_days / (2 * math.pi)

    def one_time_unit_in_days(self):
        if self.primary_period_days <= 0:
            return None
        return self.primary_period_days / (2 * math.pi)

    def update_primary_positions(self):
        self.mu, self.body1_x, self.body2_x = simSetup(
            self.body1_mass,
            self.body2_mass,
        )
        self.l4, self.l5 = lagrange_points(self.mu)

        self.body1_marker.set_offsets([[self.body1_x, 0.0]])
        self.body2_marker.set_offsets([[self.body2_x, 0.0]])
        self.l4_marker.set_offsets([[self.l4[0], self.l4[1]]])
        self.l5_marker.set_offsets([[self.l5[0], self.l5[1]]])

    def apply_reset(self):
        self.running = False
        self.run_button.setText("Run")
        self.body1_mass = self.parse_float(self.system_box, self.body1_mass)
        self.body2_mass = self.parse_float(self.secondary_box, self.body2_mass)
        self.primary_period_days = self.parse_float(
            self.period_box,
            self.primary_period_days,
        )
        self.update_primary_positions()

        x0 = self.parse_float(self.x_box, self.state[0])
        y0 = self.parse_float(self.y_box, self.state[1])
        vx0 = self.parse_float(self.vx_box, self.state[2])
        vy0 = self.parse_float(self.vy_box, self.state[3])
        self.reset_state(x0, y0, vx0, vy0)

    def set_near_l4(self):
        self.update_masses_from_inputs()
        x0 = self.l4[0] + 0.01
        y0 = self.l4[1]
        self.set_initial_condition_boxes(x0, y0, 0.0, 0.0)
        self.reset_state(x0, y0, 0.0, 0.0)

    def set_near_l5(self):
        self.update_masses_from_inputs()
        x0 = self.l5[0] + 0.01
        y0 = self.l5[1]
        self.set_initial_condition_boxes(x0, y0, 0.0, 0.0)
        self.reset_state(x0, y0, 0.0, 0.0)

    def update_masses_from_inputs(self):
        self.body1_mass = self.parse_float(self.system_box, self.body1_mass)
        self.body2_mass = self.parse_float(self.secondary_box, self.body2_mass)
        self.primary_period_days = self.parse_float(
            self.period_box,
            self.primary_period_days,
        )
        self.update_primary_positions()

    def set_initial_condition_boxes(self, x0, y0, vx0, vy0):
        self.x_box.setText(f"{x0:.8f}")
        self.y_box.setText(f"{y0:.8f}")
        self.vx_box.setText(f"{vx0:.8f}")
        self.vy_box.setText(f"{vy0:.8f}")

    def reset_state(self, x0, y0, vx0, vy0):
        self.running = False
        self.run_button.setText("Run")
        self.batch_mode = False
        self.batch_total_steps = 0
        self.remaining_batch_steps = 0
        self.run_records = []

        self.state = [x0, y0, vx0, vy0]
        self.time = 0.0
        self.x_history = [x0]
        self.y_history = [y0]
        self.update_view()

    def apply_system_preset(self, system_name):
        preset = SYSTEM_PRESETS[system_name]
        self.current_system_name = system_name
        self.body1_mass = preset["body1_mass"]
        self.body2_mass = preset["body2_mass"]
        self.primary_period_days = preset["period_days"]

        self.system_box.setText(f"{self.body1_mass:.6e}")
        self.secondary_box.setText(f"{self.body2_mass:.6e}")
        self.period_box.setText(f"{self.primary_period_days:.6f}")
        self.update_primary_positions()

        x0 = self.l4[0] + 0.01
        y0 = self.l4[1]
        self.set_initial_condition_boxes(x0, y0, 0.0, 0.0)
        self.reset_state(x0, y0, 0.0, 0.0)

    def clear_trail(self):
        self.x_history = [self.state[0]]
        self.y_history = [self.state[1]]
        self.update_view()

    def toggle_running(self):
        self.batch_mode = False
        self.batch_total_steps = 0
        self.remaining_batch_steps = 0
        self.running = not self.running
        self.run_button.setText("Pause" if self.running else "Run")
        self.update_view()

    def start_batch_run(self):
        self.read_simulation_parameters()
        batch_steps = self.parse_int(self.batch_steps_box, 50000)

        self.batch_mode = True
        self.running = True
        self.run_button.setText("Pause")
        self.batch_total_steps = batch_steps
        self.remaining_batch_steps = batch_steps
        self.run_records = []
        self.record_current_state()
        self.update_view()

    def stop_and_export_run(self):
        self.running = False
        self.run_button.setText("Run")

        if not self.batch_mode and not self.run_records:
            self.record_current_state()

        self.batch_mode = False
        self.export_run_to_csv()
        self.update_view()

    def batch_progress_fraction(self):
        if self.batch_total_steps <= 0:
            return 0.0

        completed_steps = self.batch_total_steps - self.remaining_batch_steps
        completed_steps = max(0, min(self.batch_total_steps, completed_steps))
        return completed_steps / self.batch_total_steps

    def update_simulation(self):
        if not self.running:
            return

        self.read_simulation_parameters()
        steps_this_update = self.steps_per_update

        if self.batch_mode:
            steps_this_update = min(steps_this_update, self.remaining_batch_steps)

        for _ in range(steps_this_update):
            self.state, acc_x, acc_y, r1, r2 = simulationStep(
                self.state,
                self.timestep,
                self.mu,
            )
            self.time += self.timestep
            self.x_history.append(self.state[0])
            self.y_history.append(self.state[1])

            if self.batch_mode:
                self.remaining_batch_steps -= 1
                self.record_current_state()

        if self.batch_mode and self.remaining_batch_steps <= 0:
            self.running = False
            self.batch_mode = False
            self.run_button.setText("Run")
            self.export_run_to_csv()

        self.update_view()

    def visible_history(self):
        point_count = len(self.x_history)

        if point_count <= self.max_visible_trail_points:
            return self.x_history, self.y_history

        stride = math.ceil(point_count / self.max_visible_trail_points)
        x_visible = self.x_history[::stride]
        y_visible = self.y_history[::stride]

        if x_visible[-1] != self.x_history[-1] or y_visible[-1] != self.y_history[-1]:
            x_visible = x_visible + [self.x_history[-1]]
            y_visible = y_visible + [self.y_history[-1]]

        return x_visible, y_visible

    def update_view(self):
        x_visible, y_visible = self.visible_history()
        self.trajectory_line.set_data(x_visible, y_visible)
        self.particle_dot.set_data([self.state[0]], [self.state[1]])

        progress_value = int(self.batch_progress_fraction() * 1000)
        self.progress_bar.setValue(progress_value)

        real_days = self.sim_time_to_real_days(self.time)
        one_unit_days = self.one_time_unit_in_days()
        real_time_line = "real time: N/A"
        unit_line = "1 sim unit: N/A"

        if real_days is not None:
            real_time_line = f"real time: {real_days:.4f} days"
            unit_line = f"1 sim unit: {one_unit_days:.4f} days"

        completed_steps = self.batch_total_steps - self.remaining_batch_steps
        export_line = self.last_export_filename or "none"
        self.status_label.setText(
            f"system: {self.current_system_name}\n"
            f"running: {self.running}\n"
            f"t_sim: {self.time:.4f}\n"
            f"{real_time_line}\n"
            f"{unit_line}\n"
            f"mu: {self.mu:.8f}\n"
            f"x: {self.state[0]:.5f}\n"
            f"y: {self.state[1]:.5f}\n"
            f"vx: {self.state[2]:.5f}\n"
            f"vy: {self.state[3]:.5f}\n"
            f"trail drawn: {len(x_visible)}/{len(self.x_history)}\n"
            f"csv steps: {completed_steps}/{self.batch_total_steps}\n"
            f"last export: {export_line}"
        )

        self.canvas.draw_idle()

    def record_current_state(self):
        x, y, vx, vy = self.state
        r1, r2 = particleToBody1And2Distance(x, y, self.mu)
        acc_x = accelerationX(vy, x, self.mu, r1, r2)
        acc_y = accelerationY(vx, y, self.mu, r1, r2)
        real_days = self.sim_time_to_real_days(self.time)

        self.run_records.append({
            "step_index": len(self.run_records),
            "system": self.current_system_name,
            "mu": self.mu,
            "timestep": self.timestep,
            "time_sim": self.time,
            "time_real_days": real_days if real_days is not None else "",
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "ax": acc_x,
            "ay": acc_y,
            "r1": r1,
            "r2": r2,
        })

    def export_run_to_csv(self):
        if not self.run_records:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_system_name = self.current_system_name.replace(" ", "_").replace("-", "_")
        filename = f"cr3bp_{safe_system_name}_{timestamp}.csv"

        fieldnames = [
            "step_index",
            "system",
            "mu",
            "timestep",
            "time_sim",
            "time_real_days",
            "x",
            "y",
            "vx",
            "vy",
            "ax",
            "ay",
            "r1",
            "r2",
        ]

        with open(filename, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.run_records)

        self.last_export_filename = filename


def main():
    app = QApplication(sys.argv)
    gui = CR3BPGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
