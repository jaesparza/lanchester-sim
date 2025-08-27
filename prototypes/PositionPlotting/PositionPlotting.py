import matplotlib.pyplot as plt

# Example: 3 ships at different positions
ships = {
    "Alpha": (0, 0),
    "Bravo": (200, 0),
    "Charlie": (100, 173)  # ~200 km from both
}

plt.figure(figsize=(6,6))
for name, (x, y) in ships.items():
    plt.scatter(x, y, marker="^", s=200, label=name)  # triangle = ship
    plt.text(x+5, y+5, name)

plt.xlabel("km (east)")
plt.ylabel("km (north)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
