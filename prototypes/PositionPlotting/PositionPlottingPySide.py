from qtpy.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import sys

# Sample tracks
alpha_track = [(0, 0), (20, 0), (40, 0), (60, 0)]
bravo_track = [(0, 100), (20, 80), (40, 60), (60, 40)]

ships = {
    "Alpha": {"track": alpha_track, "color": "green"},
    "Bravo": {"track": bravo_track, "color": "blue"},
}

max_frames = max(len(s["track"]) for s in ships.values())
frame = 0  # current simulation step

# --- Main window ---
class FleetWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fleet Simulation (Qt)")
        self.frame = 0

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.vbox = QVBoxLayout(self.main_widget)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.canvas = FigureCanvas(self.fig)
        self.vbox.addWidget(self.canvas)

        self.ax.set_xlim(-50, 100)
        self.ax.set_ylim(-50, 150)
        self.ax.set_xlabel("km (east)")
        self.ax.set_ylabel("km (north)")
        self.ax.grid(True)

        # Ship markers
        self.scatters = {name: self.ax.scatter([], [], marker="^", s=200, c=data["color"], label=name) 
                         for name, data in ships.items()}
        self.ax.legend()

        # Step counter
        self.step_text = self.ax.text(
            0.95, 0.02, f"Step: {self.frame}", 
            transform=self.ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3")
        )

        # Buttons
        self.hbox = QHBoxLayout()
        self.prev_button = QPushButton("Previous Step")
        self.next_button = QPushButton("Next Step")
        self.hbox.addWidget(self.prev_button)
        self.hbox.addWidget(self.next_button)
        self.vbox.addLayout(self.hbox)

        # Connect signals
        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button.clicked.connect(self.next_frame)

        # Initial update
        self.update_plot()

    def update_plot(self):
        for name, data in ships.items():
            track = data["track"]
            if self.frame < len(track):
                x, y = track[self.frame]
                self.scatters[name].set_offsets([x, y])
        self.step_text.set_text(f"Step: {self.frame}")
        self.canvas.draw()

    def next_frame(self):
        self.frame = (self.frame + 1) % max_frames
        self.update_plot()

    def prev_frame(self):
        self.frame = (self.frame - 1) % max_frames
        self.update_plot()

# --- Run app ---
app = QApplication(sys.argv)
window = FleetWindow()
window.show()
sys.exit(app.exec_())
