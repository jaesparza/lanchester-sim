import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Example tracks (each a list of (x, y) km positions)
alpha_track = [(0, 0), (20, 0), (40, 0), (60, 0), (80, 0), (100, 0)]
bravo_track = [(0, 200), (20, 180), (40, 160), (60, 140), (80, 120), (100, 100)]
charlie_track = [(200, 0), (180, 20), (160, 40), (140, 60), (120, 80), (100, 100)]

ships = {
    "Alpha": {"track": alpha_track, "color": "green"},
    "Bravo": {"track": bravo_track, "color": "blue"},
    "Charlie": {"track": charlie_track, "color": "red"},
}

# Setup figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-50, 250)
ax.set_ylim(-50, 250)
ax.set_xlabel("km (east)")
ax.set_ylabel("km (north)")
ax.set_title("Fleet Movement")
ax.grid(True)

# Ship markers
scatters = {name: ax.scatter([], [], marker="^", s=200, c=data["color"], label=name) 
            for name, data in ships.items()}
ax.legend()

# Update function
def update(frame):
    for name, data in ships.items():
        track = data["track"]
        if frame < len(track):
            x, y = track[frame]
            scatters[name].set_offsets([x, y])
    return scatters.values()

ani = animation.FuncAnimation(fig, update, 
                              frames=max(len(s["track"]) for s in ships.values()),
                              interval=1000, blit=True, repeat=True)

plt.show()
