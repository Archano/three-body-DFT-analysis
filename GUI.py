import csv
import math
import os
import sys
from datetime import datetime

import numpy as np

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
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
    simulationStep as euler_step,
)
from physicsEngineRK4 import simulationStep as rk4_step
from DFTplotter import (
    load_csv_metadata,
    load_position_csv,
    point_distance_spectrum,
    position_spectrum,
    spectrum_characteristics,
)


CSV_OUTPUT_DIR = "csv_outputs"


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
    # Finding roots of a 5-nomial using bisection
    def find_root(f, a, b, tol=1e-12):
        for _ in range(100):
            c = (a + b) / 2.0
            if abs(f(c)) < tol or (b - a) / 2.0 < tol:
                return c
            if f(c) * f(a) > 0:
                a = c
            else:
                b = c
        return (a + b) / 2.0

    def f_collinear(x):
        term1 = (1 - mu) * (x + mu) / abs(x + mu) ** 3
        term2 = mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3
        return x - term1 - term2

    l1_x = find_root(f_collinear, -mu + 1e-5, 1 - mu - 1e-5)
    l2_x = find_root(f_collinear, 1 - mu + 1e-5, 2.0)
    l3_x = find_root(f_collinear, -2.0, -mu - 1e-5)

    l4 = (0.5 - mu, math.sqrt(3) / 2)
    l5 = (0.5 - mu, -math.sqrt(3) / 2)

    return (l1_x, 0.0), (l2_x, 0.0), (l3_x, 0.0), l4, l5


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
        self.l1, self.l2, self.l3, self.l4, self.l5 = lagrange_points(self.mu)

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
        tabs = QTabWidget()
        simulation_tab = QWidget()
        root_layout = QHBoxLayout(simulation_tab)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        controls = self.build_controls()
        root_layout.addWidget(controls, 0)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(8)

        self.figure = Figure(figsize=(8, 6), tight_layout=True, facecolor="#f7f8fa")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        plot_tool_row = QHBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_zoom_in_button = QPushButton("Zoom in")
        plot_zoom_in_button.clicked.connect(self.zoom_in_simulation_plot)
        plot_zoom_out_button = QPushButton("Zoom out")
        plot_zoom_out_button.clicked.connect(self.zoom_out_simulation_plot)
        plot_tool_row.addWidget(self.toolbar, 1)
        plot_tool_row.addWidget(plot_zoom_in_button)
        plot_tool_row.addWidget(plot_zoom_out_button)
        plot_layout.addLayout(plot_tool_row)

        self.status_label = QLabel()
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.status_label.setWordWrap(True)
        self.status_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.status_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px; "
            "background: #ffffff; color: #20242a; padding: 8px;"
        )
        plot_layout.addWidget(self.status_label)
        plot_layout.addWidget(self.canvas)

        root_layout.addWidget(plot_panel, 1)

        tabs.addTab(simulation_tab, "Simulation")
        tabs.addTab(self.build_processing_tab(), "Processing")
        tabs.addTab(self.build_frequency_report_tab(), "Frequency Report")
        self.setCentralWidget(tabs)

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

        self.integrator_box = QComboBox()
        self.integrator_box.addItems(["Euler", "RK4"])
        self.integrator_box.setCurrentText("RK4")
        self.integrator_box.setToolTip(
            "Euler: fast, first-order (lower accuracy at large dt)\n"
            "RK4: four-stage Runge-Kutta, fourth-order (recommended)"
        )
        sim_layout.addRow("Integrator", self.integrator_box)

        sim_buttons = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.toggle_running)
        self.reset_button = QPushButton("Apply reset")
        self.reset_button.clicked.connect(self.apply_reset)
        sim_buttons.addWidget(self.run_button)
        sim_buttons.addWidget(self.reset_button)
        sim_layout.addRow(sim_buttons)

        unstable_group = QGroupBox("Unstable Points")
        unstable_layout = QHBoxLayout(unstable_group)

        l1_button = QPushButton("Near L1")
        l1_button.clicked.connect(self.set_near_l1)
        l2_button = QPushButton("Near L2")
        l2_button.clicked.connect(self.set_near_l2)
        l3_button = QPushButton("Near L3")
        l3_button.clicked.connect(self.set_near_l3)

        unstable_layout.addWidget(l1_button)
        unstable_layout.addWidget(l2_button)
        unstable_layout.addWidget(l3_button)
        sim_layout.addRow(unstable_group)

        stable_group = QGroupBox("Stable Points")
        stable_layout = QHBoxLayout(stable_group)

        l4_button = QPushButton("Near L4")
        l4_button.clicked.connect(self.set_near_l4)
        l5_button = QPushButton("Near L5")
        l5_button.clicked.connect(self.set_near_l5)

        stable_layout.addWidget(l4_button)
        stable_layout.addWidget(l5_button)
        sim_layout.addRow(stable_group)

        view_buttons = QHBoxLayout()
        clear_button = QPushButton("Clear trail")
        clear_button.clicked.connect(self.clear_trail)
        zoom_in_button = QPushButton("Zoom in")
        zoom_in_button.clicked.connect(self.zoom_in_simulation_plot)
        zoom_out_button = QPushButton("Zoom out")
        zoom_out_button.clicked.connect(self.zoom_out_simulation_plot)
        view_buttons.addWidget(clear_button)
        view_buttons.addWidget(zoom_in_button)
        view_buttons.addWidget(zoom_out_button)
        sim_layout.addRow(view_buttons)
        layout.addWidget(sim_group)

        self.batch_steps_box = self.add_line_edit("50000")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")

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

        layout.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def build_processing_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        file_group = QGroupBox("Load Simulation CSV")
        file_layout = QVBoxLayout(file_group)

        path_row = QHBoxLayout()
        self.analysis_file_box = QLineEdit()
        self.analysis_file_box.setPlaceholderText("Select a generated cr3bp_*.csv file")
        browse_button = QPushButton("Browse CSV")
        browse_button.clicked.connect(self.browse_analysis_csv)
        last_export_button = QPushButton("Use last export")
        last_export_button.clicked.connect(self.use_last_export_for_analysis)
        path_row.addWidget(self.analysis_file_box, 1)
        path_row.addWidget(browse_button)
        path_row.addWidget(last_export_button)
        file_layout.addLayout(path_row)

        compute_row = QHBoxLayout()
        x_dft_button = QPushButton("Plot x-position DFT")
        x_dft_button.clicked.connect(lambda: self.plot_position_dft("x"))
        y_dft_button = QPushButton("Plot y-position DFT")
        y_dft_button.clicked.connect(lambda: self.plot_position_dft("y"))
        both_dft_button = QPushButton("Plot x + y DFT")
        both_dft_button.clicked.connect(lambda: self.plot_position_dft("both"))
        compute_row.addWidget(x_dft_button)
        compute_row.addWidget(y_dft_button)
        compute_row.addWidget(both_dft_button)
        file_layout.addLayout(compute_row)

        point_row = QHBoxLayout()
        body1_dft_button = QPushButton("Plot Body 1 distance DFT")
        body1_dft_button.clicked.connect(lambda: self.plot_point_distance_dft("body1"))
        body2_dft_button = QPushButton("Plot Body 2 distance DFT")
        body2_dft_button.clicked.connect(lambda: self.plot_point_distance_dft("body2"))
        l4_dft_button = QPushButton("Plot L4 distance DFT")
        l4_dft_button.clicked.connect(lambda: self.plot_point_distance_dft("l4"))
        l5_dft_button = QPushButton("Plot L5 distance DFT")
        l5_dft_button.clicked.connect(lambda: self.plot_point_distance_dft("l5"))
        point_row.addWidget(body1_dft_button)
        point_row.addWidget(body2_dft_button)
        point_row.addWidget(l4_dft_button)
        point_row.addWidget(l5_dft_button)
        file_layout.addLayout(point_row)
        layout.addWidget(file_group)

        hint = QLabel(
            "This tab loads a CSV generated from the Simulation tab and computes "
            "one-sided DFT magnitudes for the particle position columns."
        )
        hint.setWordWrap(True)
        hint.setFrameShape(QFrame.Shape.StyledPanel)
        hint.setStyleSheet("color: #444; padding: 10px;")
        layout.addWidget(hint)

        result_panel = QWidget()
        result_layout = QHBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(12)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(8)

        self.analysis_figure = Figure(figsize=(8, 5), tight_layout=True, facecolor="#f7f8fa")
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        self.analysis_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.analysis_toolbar = NavigationToolbar(self.analysis_canvas, self)
        plot_layout.addWidget(self.analysis_toolbar)
        plot_layout.addWidget(self.analysis_canvas)

        self.analysis_report_label = QLabel("No analysis run yet.")
        self.analysis_report_label.setWordWrap(True)
        self.analysis_report_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.analysis_report_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.analysis_report_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px; "
            "background: #ffffff; color: #20242a; padding: 10px;"
        )
        self.analysis_report_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.analysis_report_label.setMinimumWidth(300)
        self.analysis_report_label.setMaximumWidth(430)
        self.analysis_report_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )

        result_layout.addWidget(plot_panel, 4)
        result_layout.addWidget(self.analysis_report_label, 1)
        layout.addWidget(result_panel, 1)

        self.draw_processing_placeholder()
        return tab

    def draw_processing_placeholder(self):
        self.analysis_figure.clear()
        signal_ax, dft_ax = self.analysis_figure.subplots(1, 2)

        signal_ax.set_title("Signal (With mean removed)")
        signal_ax.set_xlabel("Time [primary orbits]")
        signal_ax.set_ylabel("Signal value")
        signal_ax.set_facecolor("#fbfbfd")
        signal_ax.grid(True, color="#d9dde5", linewidth=0.8)
        signal_ax.set_xlim(0.0, 15.0)

        dft_ax.set_title("DFT Analysis")
        dft_ax.set_xlabel("Frequency [cycles / primary orbit]")
        dft_ax.set_ylabel("DFT magnitude")
        dft_ax.set_facecolor("#fbfbfd")
        dft_ax.grid(True, color="#d9dde5", linewidth=0.8)
        dft_ax.set_xlim(0.0, math.pi)
        dft_ax.text(
            0.5,
            0.5,
            "Select a CSV and run an analysis.",
            transform=dft_ax.transAxes,
            ha="center",
            va="center",
            color="#60656f",
        )
        self.analysis_canvas.draw_idle()

    def draw_processing_plot(self, title, series, signal_series):
        self.analysis_figure.clear()
        signal_ax, dft_ax = self.analysis_figure.subplots(1, 2)

        signal_ax.set_title("Signal (With mean removed)")
        signal_ax.set_xlabel("Time [primary orbits]")
        signal_ax.set_ylabel("Signal value")
        signal_ax.set_facecolor("#fbfbfd")
        signal_ax.grid(True, color="#d9dde5", linewidth=0.8)
        signal_ax.set_xlim(0.0, 15.0)

        for label, signal_time, signal_values, color in signal_series:
            signal_ax.plot(
                signal_time,
                signal_values,
                linewidth=1.0,
                label=label,
                color=color,
            )

        signal_ax.legend(loc="upper right", frameon=True, framealpha=0.92)

        dft_ax.set_title(title)
        dft_ax.set_xlabel("Frequency [cycles / primary orbit]")
        dft_ax.set_ylabel("DFT magnitude")
        dft_ax.set_facecolor("#fbfbfd")
        dft_ax.grid(True, color="#d9dde5", linewidth=0.8)

        for label, frequencies, magnitudes, color in series:
            dft_ax.plot(
                frequencies,
                magnitudes,
                linewidth=1.1,
                label=label,
                color=color,
            )

        dft_ax.set_xlim(0.0, math.pi)
        dft_ax.legend(loc="upper right", frameon=True, framealpha=0.92)
        self.analysis_canvas.draw_idle()

    def format_characteristic_report(
        self,
        title,
        file_path,
        csv_timestep,
        sample_line,
        report_items,
        extra_lines=None,
    ):
        lines = [
            title,
            "",
            f"CSV: {file_path}",
            f"timestep: {csv_timestep}",
            sample_line,
        ]

        if extra_lines:
            lines.extend(extra_lines)

        for label, analysis in report_items:
            lines.extend([
                "",
                label,
                f"dominant frequency: {analysis['dominant_frequency']:.8g} cycles / primary orbit",
                f"dominant period: {self.format_period(analysis['dominant_period'])} primary orbits",
                f"dominant magnitude: {analysis['dominant_magnitude']:.8g}",
                f"total power: {analysis['total_power']:.8g}",
                f"top 3 concentration: {analysis['spectral_concentration_3']:.4f}",
                "top peaks:",
            ])

            for rank, (frequency, magnitude, period) in enumerate(
                zip(
                    analysis["top_frequencies"],
                    analysis["top_magnitudes"],
                    analysis["top_periods"],
                ),
                start=1,
            ):
                lines.append(
                    f"  {rank}. f={frequency:.8g}, "
                    f"period={self.format_period(period)}, "
                    f"magnitude={magnitude:.8g}"
                )

        return "\n".join(lines)

    def format_period(self, period):
        if math.isinf(period):
            return "inf"
        return f"{period:.8g}"

    def setup_plot(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Circular Restricted Three-Body Problem", pad=12)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_facecolor("#fbfbfd")
        self.ax.grid(True, color="#d9dde5", linewidth=0.8)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.axhline(0.0, color="#b7bdc7", linewidth=0.8, zorder=0)
        self.ax.axvline(0.0, color="#b7bdc7", linewidth=0.8, zorder=0)

        self.body1_marker = self.ax.scatter(
            [self.body1_x],
            [0.0],
            s=190,
            color="#1f77b4",
            edgecolors="#0b3356",
            linewidths=1.2,
            label="Body 1",
            zorder=4,
        )
        self.body2_marker = self.ax.scatter(
            [self.body2_x],
            [0.0],
            s=110,
            color="#ff9f1c",
            edgecolors="#8f5200",
            linewidths=1.0,
            label="Body 2",
            zorder=4,
        )
        self.body1_label = self.ax.text(
            self.body1_x,
            -0.08,
            "Body 1",
            ha="center",
            va="top",
            fontsize=9,
            color="#0b3356",
        )
        self.body2_label = self.ax.text(
            self.body2_x,
            -0.08,
            "Body 2",
            ha="center",
            va="top",
            fontsize=9,
            color="#8f5200",
        )

        self.l1_marker = self.ax.scatter(
            [self.l1[0]], [self.l1[1]], marker="+", s=100, color="#9b5de5",
            linewidths=1.5, label="L1", zorder=3,
        )

        self.l2_marker = self.ax.scatter(
            [self.l2[0]], [self.l2[1]], marker="+", s=100, color="#f15bb5",
            linewidths=1.5, label="L2", zorder=3,
        )

        self.l3_marker = self.ax.scatter(
            [self.l3[0]], [self.l3[1]], marker="+", s=100, color="#00bbf9",
            linewidths=1.5, label="L3", zorder=3,
        )

        self.l4_marker = self.ax.scatter(
            [self.l4[0]],
            [self.l4[1]],
            marker="x",
            s=120,
            color="#2a9d8f",
            linewidths=2.0,
            label="L4",
            zorder=3,
        )
        self.l5_marker = self.ax.scatter(
            [self.l5[0]],
            [self.l5[1]],
            marker="x",
            s=120,
            color="#e76f51",
            linewidths=2.0,
            label="L5",
            zorder=3,
        )
        self.trajectory_line, = self.ax.plot(
            self.x_history,
            self.y_history,
            color="#5b5f97",
            linewidth=1.35,
            alpha=0.9,
            label="Trajectory",
            zorder=2,
        )
        self.particle_dot, = self.ax.plot(
            [self.state[0]],
            [self.state[1]],
            marker="o",
            markersize=7,
            linestyle="None",
            color="#d62828",
            markeredgecolor="#7f0000",
            label="Particle",
            zorder=5,
        )
        self.ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    def add_line_edit(self, initial_text):
        line_edit = QLineEdit(initial_text)
        line_edit.setMinimumWidth(145)
        return line_edit

    def browse_analysis_csv(self):
        csv_dir = os.path.abspath(CSV_OUTPUT_DIR)
        os.makedirs(csv_dir, exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select simulation CSV",
            csv_dir,
            "CSV files (*.csv);;All files (*)",
        )

        if file_path:
            self.analysis_file_box.setText(file_path)
            self.analysis_report_label.setText(f"Loaded path:\n{file_path}")

    def use_last_export_for_analysis(self):
        if not self.last_export_filename:
            QMessageBox.warning(
                self,
                "No export yet",
                "No CSV has been exported in this session yet.",
            )
            return

        file_path = os.path.abspath(self.last_export_filename)
        self.analysis_file_box.setText(file_path)
        self.analysis_report_label.setText(f"Loaded last export:\n{file_path}")

    def plot_position_dft(self, component):
        file_path = self.analysis_file_box.text().strip()

        if not file_path:
            QMessageBox.warning(
                self,
                "No CSV selected",
                "Choose a generated simulation CSV first.",
            )
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(
                self,
                "CSV not found",
                f"Could not find:\n{file_path}",
            )
            return

        try:
            x_values, y_values, csv_timestep = load_position_csv(file_path)
            if not x_values or not y_values:
                raise ValueError("CSV contains no x/y position rows.")

            if component == "x":
                x_spectrum = position_spectrum(x_values, csv_timestep)
                x_analysis = spectrum_characteristics(x_spectrum)
                series = [
                    (
                        "x position",
                        x_spectrum["frequencies"],
                        x_spectrum["magnitudes"],
                        "#1f77b4",
                    ),
                ]
                signal_series = [
                    (
                        "x position (With mean removed)",
                        x_spectrum["signal_time"],
                        x_spectrum["signal"],
                        "#1f77b4",
                    ),
                ]
                title = "X Position DFT Magnitudes"
                report_items = [("x position", x_analysis)]
            elif component == "y":
                y_spectrum = position_spectrum(y_values, csv_timestep)
                y_analysis = spectrum_characteristics(y_spectrum)
                series = [
                    (
                        "y position",
                        y_spectrum["frequencies"],
                        y_spectrum["magnitudes"],
                        "#ff9f1c",
                    ),
                ]
                signal_series = [
                    (
                        "y position (With mean removed)",
                        y_spectrum["signal_time"],
                        y_spectrum["signal"],
                        "#ff9f1c",
                    ),
                ]
                title = "Y Position DFT Magnitudes"
                report_items = [("y position", y_analysis)]
            else:
                x_spectrum = position_spectrum(x_values, csv_timestep)
                y_spectrum = position_spectrum(y_values, csv_timestep)
                x_analysis = spectrum_characteristics(x_spectrum)
                y_analysis = spectrum_characteristics(y_spectrum)
                series = [
                    (
                        "x position",
                        x_spectrum["frequencies"],
                        x_spectrum["magnitudes"],
                        "#1f77b4",
                    ),
                    (
                        "y position",
                        y_spectrum["frequencies"],
                        y_spectrum["magnitudes"],
                        "#ff9f1c",
                    ),
                ]
                signal_series = [
                    (
                        "x position (With mean removed)",
                        x_spectrum["signal_time"],
                        x_spectrum["signal"],
                        "#1f77b4",
                    ),
                    (
                        "y position (With mean removed)",
                        y_spectrum["signal_time"],
                        y_spectrum["signal"],
                        "#ff9f1c",
                    ),
                ]
                title = "Position DFT Magnitudes"
                report_items = [
                    ("x position", x_analysis),
                    ("y position", y_analysis),
                ]
        except (KeyError, ValueError, OSError) as exc:
            QMessageBox.critical(
                self,
                "DFT failed",
                f"Could not compute DFT magnitudes from this CSV:\n{exc}",
            )
            return

        self.draw_processing_plot(title, series, signal_series)
        self.analysis_report_label.setText(
            self.format_characteristic_report(
                title,
                file_path,
                csv_timestep,
                f"samples: x={len(x_values)}, y={len(y_values)}",
                report_items,
            )
        )

    def plot_point_distance_dft(self, point_key):
        file_path = self.analysis_file_box.text().strip()

        if not file_path:
            QMessageBox.warning(
                self,
                "No CSV selected",
                "Choose a generated simulation CSV first.",
            )
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(
                self,
                "CSV not found",
                f"Could not find:\n{file_path}",
            )
            return

        try:
            x_values, y_values, csv_timestep = load_position_csv(file_path)
            _, csv_mu = load_csv_metadata(file_path)
            if csv_mu is None:
                raise ValueError("CSV does not contain a mu value.")
            if not x_values or not y_values:
                raise ValueError("CSV contains no x/y position rows.")

            point_label, point_x, point_y, color = self.point_dft_target(
                point_key,
                csv_mu,
            )
            spectrum = point_distance_spectrum(
                point_x,
                point_y,
                x_values,
                y_values,
                csv_timestep,
            )
            analysis = spectrum_characteristics(spectrum)
            series = [
                (
                    f"distance to {point_label}",
                    spectrum["frequencies"],
                    spectrum["magnitudes"],
                    color,
                ),
            ]
            signal_series = [
                (
                    f"distance to {point_label} (With mean removed)",
                    spectrum["signal_time"],
                    spectrum["signal"],
                    color,
                ),
            ]
            title = f"Particle Distance to {point_label} DFT Magnitudes"
            report_items = [(f"distance to {point_label}", analysis)]
        except (KeyError, ValueError, OSError) as exc:
            QMessageBox.critical(
                self,
                "DFT failed",
                f"Could not compute point-distance DFT magnitudes from this CSV:\n{exc}",
            )
            return

        self.draw_processing_plot(title, series, signal_series)
        self.analysis_report_label.setText(
            self.format_characteristic_report(
                title,
                file_path,
                csv_timestep,
                f"samples: {len(x_values)}",
                report_items,
                extra_lines=[
                    f"mu: {csv_mu}",
                    f"target: {point_label} ({point_x:.8g}, {point_y:.8g})",
                ],
            )
        )

    # Frequency tab

    def build_frequency_report_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- File loader ---
        file_group = QGroupBox("Load Simulation CSV")
        file_layout = QVBoxLayout(file_group)
        path_row = QHBoxLayout()
        self.report_file_box = QLineEdit()
        self.report_file_box.setPlaceholderText("Select a generated cr3bp_*.csv file")
        browse_btn = QPushButton("Browse CSV")
        browse_btn.clicked.connect(self.browse_report_csv)
        last_btn = QPushButton("Use last export")
        last_btn.clicked.connect(self.use_last_export_for_report)
        path_row.addWidget(self.report_file_box, 1)
        path_row.addWidget(browse_btn)
        path_row.addWidget(last_btn)
        file_layout.addLayout(path_row)
        layout.addWidget(file_group)

        # Signal sel
        signals_group = QGroupBox("Signals to Include in Period Estimate")
        signals_layout = QGridLayout(signals_group)

        self.cb_x_pos = QCheckBox("X position DFT")
        self.cb_x_pos.setChecked(True)
        self.cb_y_pos = QCheckBox("Y position DFT")
        self.cb_y_pos.setChecked(True)
        self.cb_body1_dist = QCheckBox("Body 1 distance DFT")
        self.cb_body1_dist.setChecked(True)
        self.cb_body2_dist = QCheckBox("Body 2 distance DFT")
        self.cb_body2_dist.setChecked(True)
        self.cb_l4_dist = QCheckBox("L4 distance DFT")
        self.cb_l4_dist.setChecked(True)
        self.cb_l5_dist = QCheckBox("L5 distance DFT")
        self.cb_l5_dist.setChecked(False)

        signals_layout.addWidget(self.cb_x_pos,       0, 0)
        signals_layout.addWidget(self.cb_y_pos,       0, 1)
        signals_layout.addWidget(self.cb_body1_dist,  1, 0)
        signals_layout.addWidget(self.cb_body2_dist,  1, 1)
        signals_layout.addWidget(self.cb_l4_dist,     2, 0)
        signals_layout.addWidget(self.cb_l5_dist,     2, 1)

        tip = QLabel(
            "Tip: enable L4 if your particle librates near L4, L5 if near L5. "
            "Including both when only one is relevant will reduce confidence."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #555; font-size: 11px; padding-top: 4px;")
        signals_layout.addWidget(tip, 3, 0, 1, 2)

        run_btn = QPushButton("Run All DFTs & Generate Frequency Report")
        run_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        run_btn.clicked.connect(self.run_all_dfts_report)
        signals_layout.addWidget(run_btn, 4, 0, 1, 2)
        layout.addWidget(signals_group)

        # Results
        result_panel = QWidget()
        result_layout = QHBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(12)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.report_figure = Figure(figsize=(8, 4), tight_layout=True, facecolor="#f7f8fa")
        self.report_canvas = FigureCanvas(self.report_figure)
        self.report_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.report_toolbar = NavigationToolbar(self.report_canvas, self)
        plot_layout.addWidget(self.report_toolbar)
        plot_layout.addWidget(self.report_canvas)

        self.freq_report_label = QLabel(
            "No frequency report generated yet.\n\n"
            "Select a CSV, choose which signals to include,\n"
            "then click 'Run All DFTs & Generate Frequency Report'."
        )
        self.freq_report_label.setWordWrap(True)
        self.freq_report_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.freq_report_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.freq_report_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; "
            "background: #ffffff; color: #20242a; padding: 10px;"
        )
        self.freq_report_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.freq_report_label.setMinimumWidth(320)
        self.freq_report_label.setMaximumWidth(460)
        self.freq_report_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )

        result_layout.addWidget(plot_panel, 4)
        result_layout.addWidget(self.freq_report_label, 1)
        layout.addWidget(result_panel, 1)

        self._draw_report_placeholder()
        return tab

    def browse_report_csv(self):
        csv_dir = os.path.abspath(CSV_OUTPUT_DIR)
        os.makedirs(csv_dir, exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select simulation CSV",
            csv_dir,
            "CSV files (*.csv);;All files (*)",
        )
        if file_path:
            self.report_file_box.setText(file_path)

    def use_last_export_for_report(self):
        if not self.last_export_filename:
            QMessageBox.warning(
                self,
                "No export yet",
                "No CSV has been exported in this session yet.",
            )
            return
        self.report_file_box.setText(os.path.abspath(self.last_export_filename))

    def _draw_report_placeholder(self):
        self.report_figure.clear()
        ax1, ax2 = self.report_figure.subplots(1, 2)
        for ax, title in [
            (ax1, "Estimated Period by Signal"),
            (ax2, "Period Deviation from Mean"),
        ]:
            ax.set_facecolor("#fbfbfd")
            ax.set_title(title)
            ax.grid(True, axis="y", color="#d9dde5", linewidth=0.8)
            ax.text(
                0.5, 0.5,
                "Run All DFTs to see results.",
                transform=ax.transAxes,
                ha="center", va="center", color="#60656f",
            )
        self.report_canvas.draw_idle()

    def run_all_dfts_report(self):
        file_path = self.report_file_box.text().strip()

        if not file_path:
            QMessageBox.warning(
                self, "No CSV selected",
                "Choose a generated simulation CSV first.",
            )
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(
                self, "CSV not found",
                f"Could not find:\n{file_path}",
            )
            return

        try:
            x_values, y_values, csv_timestep = load_position_csv(file_path)
            _, csv_mu = load_csv_metadata(file_path)
            if csv_mu is None:
                raise ValueError("CSV does not contain a mu value.")
            if not x_values or not y_values:
                raise ValueError("CSV contains no x/y position rows.")
        except (KeyError, ValueError, OSError) as exc:
            QMessageBox.critical(
                self, "Load failed",
                f"Could not load CSV:\n{exc}",
            )
            return

        _, _, _, l4, l5 = lagrange_points(csv_mu)

        results = []

        def _add(label, spectrum, color):
            dom_freq = spectrum["dominant_frequency"]
            dom_mag  = spectrum["dominant_magnitude"]
            if dom_freq is not None and dom_freq > 0:
                period = 1.0 / dom_freq
            else:
                period = float("inf")
            results.append({
                "label":     label,
                "frequency": dom_freq,
                "period":    period,
                "magnitude": dom_mag,
                "color":     color,
            })

        try:
            if self.cb_x_pos.isChecked():
                _add("X position",
                     position_spectrum(x_values, csv_timestep),
                     "#1f77b4")
            if self.cb_y_pos.isChecked():
                _add("Y position",
                     position_spectrum(y_values, csv_timestep),
                     "#ff9f1c")
            if self.cb_body1_dist.isChecked():
                _add("Dist. Body 1",
                     point_distance_spectrum(
                         -csv_mu, 0.0, x_values, y_values, csv_timestep),
                     "#9b5de5")
            if self.cb_body2_dist.isChecked():
                _add("Dist. Body 2",
                     point_distance_spectrum(
                         1.0 - csv_mu, 0.0, x_values, y_values, csv_timestep),
                     "#f15bb5")
            if self.cb_l4_dist.isChecked():
                _add("Dist. L4",
                     point_distance_spectrum(
                         l4[0], l4[1], x_values, y_values, csv_timestep),
                     "#2a9d8f")
            if self.cb_l5_dist.isChecked():
                _add("Dist. L5",
                     point_distance_spectrum(
                         l5[0], l5[1], x_values, y_values, csv_timestep),
                     "#e76f51")
        except (ValueError, OSError) as exc:
            QMessageBox.critical(
                self, "DFT failed",
                f"Could not compute DFT magnitudes:\n{exc}",
            )
            return

        if not results:
            QMessageBox.warning(
                self, "No signals selected",
                "Enable at least one signal checkbox above.",
            )
            return

        #Statistics on signals
        valid = [r for r in results if not math.isinf(r["period"])]
        if not valid:
            QMessageBox.warning(
                self, "No valid periods",
                "All selected signals produced zero or infinite periods.\n"
                "Try a longer simulation run or a different initial condition.",
            )
            return

        periods     = np.array([r["period"] for r in valid])
        mean_period = float(np.mean(periods))
        std_period  = float(np.std(periods))      # population std
        spread      = float(np.ptp(periods))       # max - min
        cv          = (std_period / mean_period * 100.0) if mean_period > 0 else float("inf")

        if cv < 2.0:
            confidence_label, confidence_color = "EXCELLENT", "#2a9d8f"
        elif cv < 5.0:
            confidence_label, confidence_color = "GOOD",      "#57cc99"
        elif cv < 10.0:
            confidence_label, confidence_color = "FAIR",      "#f4a261"
        else:
            confidence_label, confidence_color = "POOR",      "#e63946"

        for r in results:
            if not math.isinf(r["period"]):
                r["deviation"] = r["period"] - mean_period
                r["dev_pct"]   = (r["period"] - mean_period) / mean_period * 100.0
            else:
                r["deviation"] = float("inf")
                r["dev_pct"]   = float("inf")

        mean_real_days = (
            mean_period * self.primary_period_days
            if self.primary_period_days > 0
            else None
        )

        stats = {
            "mean_period":      mean_period,
            "std_period":       std_period,
            "cv":               cv,
            "spread":           spread,
            "confidence_label": confidence_label,
            "confidence_color": confidence_color,
            "n_valid":          len(valid),
            "n_total":          len(results),
            "csv_mu":           csv_mu,
            "csv_timestep":     csv_timestep,
            "n_samples":        len(x_values),
            "mean_real_days":   mean_real_days,
        }

        self.draw_frequency_report_plot(results, stats)
        self.freq_report_label.setText(
            self._format_frequency_report(results, stats, file_path)
        )

    def draw_frequency_report_plot(self, results, stats):
        self.report_figure.clear()
        period_ax, dev_ax = self.report_figure.subplots(1, 2)

        labels   = [r["label"] for r in results]
        periods  = [
            r["period"] if not math.isinf(r["period"]) else 0.0
            for r in results
        ]
        dev_pcts = [
            r.get("dev_pct", 0.0)
            if not math.isinf(r.get("dev_pct", float("inf"))) else 0.0
            for r in results
        ]
        colors  = [r["color"] for r in results]
        x_pos   = np.arange(len(results))

        # Left panel: period per signal + mean line
        period_ax.bar(x_pos, periods, color=colors,
                      edgecolor="#444444", linewidth=0.8, alpha=0.85)
        period_ax.axhline(
            stats["mean_period"],
            color="#d62828", linewidth=1.8, linestyle="--",
            label=f"Mean = {stats['mean_period']:.6g}",
        )
        period_ax.set_xticks(x_pos)
        period_ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
        period_ax.set_ylabel("Period [primary orbits]")
        period_ax.set_title("Estimated Period by Signal")
        period_ax.set_facecolor("#fbfbfd")
        period_ax.grid(True, axis="y", color="#d9dde5", linewidth=0.8)
        period_ax.legend(fontsize=9, frameon=True, framealpha=0.9)

        # Right panel - deviation from mean, coloured by sign
        dev_bar_colors = [
            "#2a9d8f" if d >= 0 else "#e76f51"
            for d in dev_pcts
        ]
        dev_ax.bar(x_pos, dev_pcts, color=dev_bar_colors,
                   edgecolor="#444444", linewidth=0.8, alpha=0.85)
        dev_ax.axhline(0.0, color="#333333", linewidth=1.0)
        dev_ax.set_xticks(x_pos)
        dev_ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
        dev_ax.set_ylabel("Deviation from consensus mean [%]")
        dev_ax.set_title(
            f"Period Deviation from Mean\n"
            f"Confidence: {stats['confidence_label']}  "
            f"(CV = {stats['cv']:.2f}%)"
        )
        dev_ax.set_facecolor("#fbfbfd")
        dev_ax.grid(True, axis="y", color="#d9dde5", linewidth=0.8)

        self.report_canvas.draw_idle()

    def _format_frequency_report(self, results, stats, file_path):
        lines = [
            "FREQUENCY REPORT",
            "",
            f"CSV:      {os.path.basename(file_path)}",
            f"mu:       {stats['csv_mu']:.8g}",
            f"timestep: {stats['csv_timestep']}",
            f"samples:  {stats['n_samples']}",
            f"signals:  {stats['n_valid']} valid / "
            f"{stats['n_total']} selected",
            "",
            "Per-Signal Results",
        ]

        for r in results:
            freq_str = (
                f"{r['frequency']:.8g}"
                if r["frequency"] is not None
                else "N/A"
            )
            period_str = self.format_period(r["period"])
            if not math.isinf(r.get("dev_pct", float("inf"))):
                dev_str = f"{r['dev_pct']:+.3f}%"
            else:
                dev_str = "excluded (infinite period)"
            lines += [
                "",
                f"[{r['label']}]",
                f"  freq:   {freq_str} cyc/orbit",
                f"  period: {period_str} orbits",
                f"  dev:    {dev_str}",
            ]

        lines += [
            "",
            "Consensus Estimate",
            f"mean period:  {stats['mean_period']:.8g} primary orbits",
            f"std dev:      {stats['std_period']:.8g} orbits",
            f"spread:       {stats['spread']:.8g} orbits",
            f"coeff. var.:  {stats['cv']:.4f}%",
        ]

        if stats.get("mean_real_days") is not None:
            lines.append(
                f"real time:    {stats['mean_real_days']:.4f} days"
                f"  ({self.primary_period_days:.4f} d/orbit)"
            )

        lines += [
            "",
            f"Confidence:   {stats['confidence_label']}",
            "  <2%  = Excellent",
            "  <5%  = Good",
            "  <10% = Fair",
            "  >=10% = Poor",
        ]

        return "\n".join(lines)


    # End Frequency tab


    def point_dft_target(self, point_key, mu):
        _, _, _, l4, l5 = lagrange_points(mu)
        targets = {
            "body1": ("Body 1", -mu, 0.0, "#1f77b4"),
            "body2": ("Body 2", 1.0 - mu, 0.0, "#ff9f1c"),
            "l4": ("L4", l4[0], l4[1], "#2a9d8f"),
            "l5": ("L5", l5[0], l5[1], "#e76f51"),
        }

        if point_key not in targets:
            raise ValueError(f"Unknown point target: {point_key}")

        return targets[point_key]

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
        self.l1, self.l2, self.l3, self.l4, self.l5 = lagrange_points(self.mu)

        self.body1_marker.set_offsets([[self.body1_x, 0.0]])
        self.body2_marker.set_offsets([[self.body2_x, 0.0]])
        self.l1_marker.set_offsets([[self.l1[0], self.l1[1]]])
        self.l2_marker.set_offsets([[self.l2[0], self.l2[1]]])
        self.l3_marker.set_offsets([[self.l3[0], self.l3[1]]])
        self.l4_marker.set_offsets([[self.l4[0], self.l4[1]]])
        self.l5_marker.set_offsets([[self.l5[0], self.l5[1]]])

        self.body1_label.set_position((self.body1_x, -0.08))
        self.body2_label.set_position((self.body2_x, -0.08))

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

    def set_near_l1(self):
        self.update_masses_from_inputs()
        x0, y0 = self.l1[0] + 0.01, self.l1[1]
        self.set_initial_condition_boxes(x0, y0, 0.0, 0.0)
        self.reset_state(x0, y0, 0.0, 0.0)

    def set_near_l2(self):
        self.update_masses_from_inputs()
        x0, y0 = self.l2[0] + 0.01, self.l2[1]
        self.set_initial_condition_boxes(x0, y0, 0.0, 0.0)
        self.reset_state(x0, y0, 0.0, 0.0)

    def set_near_l3(self):
        self.update_masses_from_inputs()
        x0, y0 = self.l3[0] + 0.01, self.l3[1]
        self.set_initial_condition_boxes(x0, y0, 0.0, 0.0)
        self.reset_state(x0, y0, 0.0, 0.0)

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

    def zoom_out_simulation_plot(self):
        self.zoom_simulation_plot(1.5)

    def zoom_in_simulation_plot(self):
        self.zoom_simulation_plot(1 / 1.5)

    def zoom_simulation_plot(self, zoom_factor):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_half_width = (x_max - x_min) * zoom_factor / 2
        y_half_width = (y_max - y_min) * zoom_factor / 2

        self.ax.set_xlim(x_center - x_half_width, x_center + x_half_width)
        self.ax.set_ylim(y_center - y_half_width, y_center + y_half_width)
        self.canvas.draw_idle()

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

        step_fn = (
            rk4_step
            if self.integrator_box.currentText() == "RK4"
            else euler_step
        )

        for _ in range(steps_this_update):
            self.state, acc_x, acc_y, r1, r2 = step_fn(
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
        real_time = "N/A"
        unit_time = "N/A"

        if real_days is not None:
            real_time = f"{real_days:.4f} days"
            unit_time = f"{one_unit_days:.4f} days"

        completed_steps = self.batch_total_steps - self.remaining_batch_steps
        export_line = self.last_export_filename or "none"
        self.status_label.setText(
            f"System: {self.current_system_name}   "
            f"Integrator: {self.integrator_box.currentText()}   "
            f"Running: {self.running}   "
            f"mu: {self.mu:.8f}   "
            f"t_sim: {self.time:.4f}   "
            f"real time: {real_time}   "
            f"1 sim unit: {unit_time}\n"
            f"x: {self.state[0]:.5f}   "
            f"y: {self.state[1]:.5f}   "
            f"vx: {self.state[2]:.5f}   "
            f"vy: {self.state[3]:.5f}   "
            f"trail: {len(x_visible)}/{len(self.x_history)}   "
            f"csv: {completed_steps}/{self.batch_total_steps}   "
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

        os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_system_name = self.current_system_name.replace(" ", "_").replace("-", "_")
        filename = os.path.join(
            CSV_OUTPUT_DIR,
            f"cr3bp_{safe_system_name}_{timestamp}.csv",
        )

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