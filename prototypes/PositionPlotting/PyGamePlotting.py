import pygame
import sys

# --- Setup ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fleet Movement")
clock = pygame.time.Clock()

# Scale factor: 1 km = 2 pixels (so 200 km ~ 400 px)
SCALE = 2

# Ship data: [ (x km, y km), ... ]
alpha_track = [(0, 0), (20, 0), (40, 0), (60, 0), (80, 0), (100, 0)]
bravo_track = [(0, 200), (20, 180), (40, 160), (60, 140), (80, 120), (100, 100)]

ships = {
    "Alpha": {"track": alpha_track, "color": (0, 255, 0)},
    "Bravo": {"track": bravo_track, "color": (0, 128, 255)},
}

# Convert km → pixels (centered on screen)
def km_to_px(x, y):
    return int(WIDTH/2 + x*SCALE), int(HEIGHT/2 - y*SCALE)

frame = 0

# --- Main loop ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((30, 30, 30))  # dark background

    # Draw grid (every 50 km)
    for dx in range(0, WIDTH, 100):
        pygame.draw.line(screen, (60, 60, 60), (dx, 0), (dx, HEIGHT))
    for dy in range(0, HEIGHT, 100):
        pygame.draw.line(screen, (60, 60, 60), (0, dy), (WIDTH, dy))

    # Draw ships at current frame
    for name, data in ships.items():
        if frame < len(data["track"]):
            x, y = data["track"][frame]
            px, py = km_to_px(x, y)
            pygame.draw.polygon(screen, data["color"], [
                (px, py-10), (px+15, py+10), (px-15, py+10)
            ])
            font = pygame.font.SysFont(None, 24)
            label = font.render(name, True, (255,255,255))
            screen.blit(label, (px+10, py-20))

    pygame.display.flip()
    clock.tick(1)  # 1 frame per second

    frame += 1
    if frame >= max(len(s["track"]) for s in ships.values()):
        frame = 0  # loop animation
