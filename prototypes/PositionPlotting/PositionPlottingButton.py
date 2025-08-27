import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Tracks (km positions per timestep)
alpha_track = [(0, 0), (20, 0), (40, 0), (60, 0), (80, 0), (100, 0)]
bravo_track = [(0, 200), (20, 180), (40, 160), (60, 140), (80, 120), (100, 100)]
charlie_track = [(200, 0), (180, 20), (160, 40), (140, 60), (120, 80), (100, 100)]

ships = {
    "Alpha": {"track": alpha_track, "color": "green"},
    "Bravo": {"track": bravo_track, "color": "blue"},
    "Charlie": {"track": charlie_track, "color": "red"},
}

max_frames = max(len(s["track"]) for s in ships.values())
frame = 0  # current simulation time

# --- Setup plot ---
fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(bottom=0.3)  # more space for buttons

ax.set_xlim(-50, 250)
ax.set_ylim(-50, 250)
ax.set_xlabel("km (east)")
ax.set_ylabel("km (north)")
ax.set_title("Fleet Movement (Manual Step)")
ax.grid(True)

# Ship markers
scatters = {name: ax.scatter([], [], marker="^", s=200, c=data["color"], label=name) 
            for name, data in ships.items()}
ax.legend()

# Salvo line
salvo_line, = ax.plot([], [], "r--", lw=2, alpha=0.7)

# Step counter (bottom right)
step_text = ax.text(
    0.95, 0.02, f"Step: {frame}", 
    transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="white",
    bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3")
)

# --- Update function ---
def update(frame):
    # Update ship positions
    for name, data in ships.items():
        track = data["track"]
        if frame < len(track):
            x, y = track[frame]
            scatters[name].set_offsets([x, y])

    # Engagement: Bravo fires on Charlie at frame 2
    if frame == 2:
        bx, by = ships["Bravo"]["track"][frame]
        cx, cy = ships["Charlie"]["track"][frame]
        salvo_line.set_data([bx, cx], [by, cy])
    else:
        salvo_line.set_data([], [])

    # Update step counter
    step_text.set_text(f"Step: {frame}")

# --- Button callbacks ---
def next_frame(event):
    global frame
    if frame < max_frames - 1:
        frame += 1
    else:
        frame = 0  # loop back
    update(frame)
    plt.draw()

def prev_frame(event):
    global frame
    if frame > 0:
        frame -= 1
    else:
        frame = max_frames - 1  # wrap around
    update(frame)
    plt.draw()

# --- Create buttons ---
ax_prev = plt.axes([0.25, 0.1, 0.2, 0.075])  # x, y, w, h
ax_next = plt.axes([0.55, 0.1, 0.2, 0.075])

button_prev = Button(ax_prev, "Previous Step")
button_next = Button(ax_next, "Next Step")

button_prev.on_clicked(prev_frame)
button_next.on_clicked(next_frame)

# Initialize
update(frame)

plt.show()
