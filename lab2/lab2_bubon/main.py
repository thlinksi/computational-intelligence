import math
import random
import subprocess
import sys

NEEDED_PACKAGES = {
    "requests": "requests",
    "matplotlib": "matplotlib",
    "pygame": "pygame"
}

for module_name, pip_name in NEEDED_PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"Пакет '{pip_name}' відсутній. Встановлення...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

import requests
import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules

if not IN_NOTEBOOK:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if IN_NOTEBOOK:
    from IPython.display import clear_output, display

BASE_URL = "https://ooi-lab2.up.railway.app"

STUDENT_NAME = "Бубон Л. В."
VARIANT_NUMBER = 3

T_START = 30.0
COOLING_RATE = 0.98
T_MIN = 0.3
R_MAX = 25
STAGNATION_LIMIT = 80

MAX_SERVER_STEPS = 20000
MAX_ITERATIONS = 300000

SHOW_EVERY = 3
KEEP_WINDOW_OPEN = True
RANDOM_SEED = None

RUN_EXPERIMENTS = True
RUNS_PER_EXPERIMENT = 3
EXPERIMENTS = [(0.90, 30.0), (0.95, 30.0), (0.98, 30.0), (0.99, 30.0), (0.98, 10.0), (0.98, 50.0)]

def start_session():
    response = requests.post(
        BASE_URL + "/api/start",
        json={"student_name": STUDENT_NAME, "variant_number": VARIANT_NUMBER},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    start = (data["start_position"]["x"], data["start_position"]["y"])
    return data["session_id"], start, data["map"]


def query_cell(session_id, x, y):
    response = requests.post(
        BASE_URL + "/api/move",
        json={"session_id": session_id, "target_position": {"x": x, "y": y}},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()

def energy(signal_level):
    return -signal_level

def temperature(k):
    return max(T_MIN, T_START * (COOLING_RATE ** k))

def radius(current_temperature):
    return max(1, round(R_MAX * current_temperature / T_START))

def neighbor(position, grid, r):
    height = len(grid)
    width = len(grid[0])
    x0, y0 = position

    for _ in range(60):
        x = x0 + random.randint(-r, r)
        y = y0 + random.randint(-r, r)
        if (x, y) == (x0, y0):
            continue
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        if grid[y][x] == 1:
            continue
        return (x, y)
    return None

def acceptance_probability(delta_energy, current_temperature):
    if delta_energy < 0:
        return 1.0
    return math.exp(-delta_energy / current_temperature)

class PygameVisualizer:
    CELL = 8
    BAR = 34

    def __init__(self, grid, start):
        import pygame
        self.pg = pygame
        pygame.init()
        self.height = len(grid)
        self.width = len(grid[0])
        map_w = self.width * self.CELL
        map_h = self.height * self.CELL
        self.screen = pygame.display.set_mode((map_w, map_h + self.BAR))
        pygame.display.set_caption(f"Lab2_Bubon - Варіант {VARIANT_NUMBER}")

        self.font = pygame.font.SysFont("dejavusans,arial,liberationsans,freesans", 18)
        self.map_surface = pygame.Surface((map_w, map_h))
        self.map_surface.fill((240, 240, 240))
        for y in range(self.height):
            for x in range(self.width):
                if grid[y][x] == 1:
                    pygame.draw.rect(self.map_surface, (46, 125, 50),
                                     (x * self.CELL, y * self.CELL,
                                      self.CELL, self.CELL))
        self.path_surface = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
        self.drawn = 0
        self.closed = False
        self.clock = pygame.time.Clock()

    def _center(self, p):
        return (p[0] * self.CELL + self.CELL // 2,
                p[1] * self.CELL + self.CELL // 2)

    def update(self, path, found, title):
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.closed = True

        while self.drawn < len(path) - 1:
            pg.draw.line(self.path_surface, (30, 100, 220, 150),
                         self._center(path[self.drawn]),
                         self._center(path[self.drawn + 1]), 1)
            self.drawn += 1
        self.screen.blit(self.map_surface, (0, 0))
        self.screen.blit(self.path_surface, (0, 0))

        for p in found:
            c = self._center(p)
            pg.draw.circle(self.screen, (0, 0, 0), c, self.CELL + 2)
            pg.draw.circle(self.screen, (255, 215, 0), c, self.CELL)
        rx, ry = self._center(path[-1])
        pg.draw.rect(self.screen, (220, 20, 20),
                     (rx - self.CELL // 2, ry - self.CELL // 2,
                      self.CELL, self.CELL))
        bar_top = self.height * self.CELL
        pg.draw.rect(self.screen, (20, 20, 20),
                     (0, bar_top, self.width * self.CELL, self.BAR))
        text = self.font.render(title, True, (255, 255, 255))
        self.screen.blit(text, (8, bar_top + 7))
        pg.display.flip()

    def save(self, filename):
        self.pg.image.save(self.screen, filename)

    def wait_close(self):
        pg = self.pg
        print("Закрийт вікно візуалізації, щоб завершити.")
        while not self.closed:
            for event in pg.event.get():
                if event.type in (pg.QUIT, pg.KEYDOWN):
                    self.closed = True
            self.clock.tick(30)
        pg.quit()

class MatplotlibVisualizer:
    def __init__(self, grid, start):
        self.closed = False
        if not IN_NOTEBOOK:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        cmap = ListedColormap(["#f0f0f0", "#2e7d32"])
        self.ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        (self.path_line,) = self.ax.plot([], [], color="tab:blue",
                                         linewidth=0.6, alpha=0.6)
        (self.found_marks,) = self.ax.plot([], [], marker="*", color="gold",
                                           markeredgecolor="black",
                                           markersize=16, linestyle="")
        (self.robot,) = self.ax.plot([start[0]], [start[1]], marker="o",
                                     color="red", markersize=8, linestyle="")
    def update(self, path, found, title):
        self.path_line.set_data([p[0] for p in path], [p[1] for p in path])
        self.found_marks.set_data([p[0] for p in found], [p[1] for p in found])
        self.robot.set_data([path[-1][0]], [path[-1][1]])
        self.ax.set_title(title)
        if IN_NOTEBOOK:
            clear_output(wait=True)
            display(self.fig)
        else:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def save(self, filename):
        self.fig.savefig(filename, dpi=150)

    def wait_close(self):
        if not IN_NOTEBOOK:
            plt.ioff()
            plt.show()

class NullVisualizer:
    closed = False

    def update(self, path, found, title):
        pass

    def save(self, filename):
        pass

    def wait_close(self):
        pass

def make_visualizer(grid, start, headless=False):
    if headless:
        return NullVisualizer()
    if IN_NOTEBOOK:
        return MatplotlibVisualizer(grid, start)
    try:
        return PygameVisualizer(grid, start)
    except ImportError:

        print("pygame не встановлено - працюю через Matplotlib.")

        return MatplotlibVisualizer(grid, start)

def main(headless=False):
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    session_id, start, grid = start_session()
    print("Сесія:", session_id, "| старт:", start)

    viz = make_visualizer(grid, start, headless)

    current = start
    current_energy = math.inf

    path = [current]
    found = []
    cache = {}

    temperature_history = []
    k = 0
    no_improve = 0
    best_energy = math.inf

    steps_made = 0
    remaining = None
    last_signal = 0.0
    iteration = 0

    while iteration < MAX_ITERATIONS and steps_made < MAX_SERVER_STEPS:
        if remaining == 0 or viz.closed:
            break
        iteration += 1
        current_temperature = temperature(k)
        temperature_history.append(current_temperature)
        r = radius(current_temperature)
        candidate = neighbor(current, grid, r)
        if candidate is None:
            k += 1
            continue
        if candidate in cache:
            candidate_signal = cache[candidate]
        else:
            response = query_cell(session_id, candidate[0], candidate[1])
            candidate_signal = response["signal_level"]
            steps_made = response["steps_made"]
            remaining = response["remaining_foxes"]
            last_signal = candidate_signal
            if response["fox_caught"]:
                found.append(candidate)

                print(f"Знайдено об'єкт #{len(found)} у {candidate}, "
                      f"кроків: {steps_made}, залишилось: {remaining}")

                cache.clear()
                current = candidate
                path.append(current)
                current_energy = math.inf
                best_energy = math.inf
                k = 0
                no_improve = 0
                viz.update(path, found,
                           f"Кроків: {steps_made} | Залишилось: {remaining} | "
                           f"T: {T_START:.1f} (reheat)")
                continue
            cache[candidate] = candidate_signal
            if steps_made % SHOW_EVERY == 0:
                viz.update(path, found,
                           f"Кроків: {steps_made} | Залишилось: {remaining} | "
                           f"Сигнал: {last_signal:.2f} | T: {current_temperature:.2f}")
        candidate_energy = energy(candidate_signal)
        delta_energy = candidate_energy - current_energy
        probability = acceptance_probability(delta_energy, current_temperature)
        if random.random() < probability:
            current = candidate
            current_energy = candidate_energy
            path.append(current)
        if candidate_energy < best_energy - 1e-9:
            best_energy = candidate_energy
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= STAGNATION_LIMIT:
                k = 0
                no_improve = 0
                continue
        k += 1

    print("Знайдено об'єктів:", len(found))
    print("Залишилось:", remaining)
    print("Кроків (запитів) до сервера:", steps_made)

    if headless:
        return remaining, steps_made

    viz.update(path, found,
               f"Фінал | Кроків: {steps_made} | Знайдено: {len(found)}")
    viz.save("result_map.png")

    print("Збережено: result_map.png")

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(temperature_history)
    ax2.set_xlabel("Номер ітерації")
    ax2.set_ylabel("Температура T")
    ax2.set_title("Зміна температури (стрибки вгору - reheating)")
    ax2.grid(True)
    fig2.savefig("temperature.png", dpi=150)

    print("Збережено: temperature.png")

    if IN_NOTEBOOK:
        plt.show()
    elif KEEP_WINDOW_OPEN:
        viz.wait_close()

def run_experiments():
    global T_START, COOLING_RATE
    saved = (T_START, COOLING_RATE)
    summary = []
    for alpha, t_start in EXPERIMENTS:
        COOLING_RATE = alpha
        T_START = t_start
        steps_list = []
        success = 0
        for _ in range(RUNS_PER_EXPERIMENT):
            remaining, steps = main(headless=True)
            steps_list.append(steps)
            if remaining == 0:
                success += 1
        summary.append((alpha, t_start, sum(steps_list) / len(steps_list), success))
    T_START, COOLING_RATE = saved
    print("alpha | T_start | середня кількість кроків | успішних запусків")
    for alpha, t_start, average, success in summary:
        print(f"{alpha:.2f} | {t_start:5.1f} | {average:8.1f} | {success}/{RUNS_PER_EXPERIMENT}")

if __name__ == "__main__":
    if RUN_EXPERIMENTS:
        run_experiments()
    else:
        main()