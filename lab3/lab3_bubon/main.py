import argparse
import json
import math
import os
import random
import statistics
import time

import matplotlib
if os.environ.get("NO_SHOW"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle, Patch

STUDENT_NAME = "Бубон Л. В."
VARIANT_NUMBER = 3

HORIZONTAL, VERTICAL = 0, 1
ORIENT_NAME = {HORIZONTAL: "горизонтально", VERTICAL: "вертикально"}
W_BOUNDS, W_REEF, W_OVERLAP, W_TOUCH = 5, 5, 3, 1

POP_SIZE = 100
P_CROSS = 0.9
P_MUT = 0.1
TOURNAMENT_K = 3
ELITE = 2
MAX_GEN = 300
T_START = 3.0
T_MIN = 0.05
ALPHA = 0.95
CHAIN_LEN = 150
P_LOCAL = 0.7

def load_variant(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = data["field_size"]
    ships = list(data["ships"])
    reefs = {(int(x), int(y)) for x, y in data["reefs"]}
    if not isinstance(n, int) or n <= 0:
        raise ValueError("field_size має бути натуральним числом")
    for x, y in reefs:
        if not (0 <= x < n and 0 <= y < n):
            raise ValueError(f"Риф ({x}, {y}) за межами поля {n}x{n}")
    for length in ships:
        if length < 1 or length > n:
            raise ValueError(f"Корабель довжиною {length} не вміщується в поле {n}x{n}")
    if sum(ships) > n * n - len(reefs):
        raise ValueError("Кораблі не вміщуються у вільні клітинки поля")
    return n, ships, reefs

class Problem:
    def __init__(self, n, ships, reefs):
        self.n, self.ships, self.reefs = n, ships, reefs
        self.evals = 0
        self.domains = [self._placements(length) for length in ships]
        for i, d in enumerate(self.domains):
            if not d:
                raise ValueError(f"Для корабля №{i + 1} немає жодного місця на полі")
        self.near = [self._neighbours(d) for d in self.domains]

    @staticmethod
    def cells(length, placement):
        x, y, o = placement
        if o == HORIZONTAL:
            return [(x + k, y) for k in range(length)]
        return [(x, y + k) for k in range(length)]

    def _placements(self, length):
        res = []
        for o in (HORIZONTAL, VERTICAL):
            if length == 1 and o == VERTICAL:
                continue
            for y in range(self.n):
                for x in range(self.n):
                    cs = self.cells(length, (x, y, o))
                    if all(0 <= cx < self.n and 0 <= cy < self.n
                           and (cx, cy) not in self.reefs for cx, cy in cs):
                        res.append((x, y, o))
        return res

    @staticmethod
    def _neighbours(domain):
        return [[j for j, q in enumerate(domain)
                 if j != i and abs(q[0] - p[0]) <= 1 and abs(q[1] - p[1]) <= 1]
                for i, p in enumerate(domain)]
    def random_chromosome(self):
        return [random.randrange(len(d)) for d in self.domains]
    def decode(self, chrom):
        return [self.domains[i][g] for i, g in enumerate(chrom)]
    def analyse(self, placements, collect=False):
        n = self.n
        owner, all_cells = {}, []
        out = reef = 0
        bad = set()
        for i, (length, p) in enumerate(zip(self.ships, placements)):
            cs = self.cells(length, p)
            all_cells.append(cs)
            for c in cs:
                if not (0 <= c[0] < n and 0 <= c[1] < n):
                    out += 1
                    bad.add(c)
                elif c in self.reefs:
                    reef += 1
                    bad.add(c)
                owner.setdefault(c, []).append(i)
        overlap = sum(len(v) - 1 for v in owner.values())
        touch = 0
        for i, cs in enumerate(all_cells):
            for (x, y) in cs:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        for j in owner.get((x + dx, y + dy), ()):
                            if j > i:
                                touch += 1
                                if collect:
                                    bad.add((x, y))
                                    bad.add((x + dx, y + dy))
        if collect:
            bad.update(c for c, v in owner.items() if len(v) > 1)
        return {"bounds": out, "reef": reef, "overlap": overlap, "touch": touch}, bad

    def penalty(self, v):
        return (W_BOUNDS * v["bounds"] + W_REEF * v["reef"]
                + W_OVERLAP * v["overlap"] + W_TOUCH * v["touch"])
    def fitness(self, chrom):
        self.evals += 1
        v, _ = self.analyse(self.decode(chrom))
        return -self.penalty(v)
    def shift_gene(self, i, g):
        nb = self.near[i][g]
        return random.choice(nb) if nb else random.randrange(len(self.domains[i]))

def tournament(pop, fits, k=TOURNAMENT_K):
    idx = random.sample(range(len(pop)), k)
    return pop[max(idx, key=lambda i: fits[i])]

def crossover(p1, p2):
    L = len(p1)
    a, b = sorted(random.sample(range(L + 1), 2))
    return p1[:a] + p2[a:b] + p1[b:], p2[:a] + p1[a:b] + p2[b:]

def mutate(problem, chrom, pm):
    res = chrom[:]
    for i in range(len(res)):
        if random.random() < pm:
            if random.random() < 0.5:
                res[i] = random.randrange(len(problem.domains[i]))
            else:
                res[i] = problem.shift_gene(i, res[i])
    return res

def run_ga(problem, pop_size=POP_SIZE, pc=P_CROSS, pm=P_MUT, k=TOURNAMENT_K,
           elite=ELITE, max_gen=MAX_GEN):
    t0 = time.time()
    problem.evals = 0
    pop = [problem.random_chromosome() for _ in range(pop_size)]
    fits = [problem.fitness(c) for c in pop]
    best_hist, mean_hist = [], []
    solved_evals = None
    for gen in range(max_gen + 1):
        best_i = max(range(pop_size), key=lambda i: fits[i])
        best_hist.append(fits[best_i])
        mean_hist.append(sum(fits) / pop_size)
        if fits[best_i] == 0:
            solved_evals = problem.evals
            break
        if gen == max_gen:
            break
        order = sorted(range(pop_size), key=lambda i: fits[i], reverse=True)
        new_pop = [pop[i][:] for i in order[:elite]]               # елітизм
        new_fits = [fits[i] for i in order[:elite]]
        while len(new_pop) < pop_size:
            p1, p2 = tournament(pop, fits, k), tournament(pop, fits, k)
            c1, c2 = crossover(p1, p2) if random.random() < pc else (p1[:], p2[:])
            for c in (mutate(problem, c1, pm), mutate(problem, c2, pm)):
                if len(new_pop) < pop_size:
                    new_pop.append(c)
                    new_fits.append(problem.fitness(c))
        pop, fits = new_pop, new_fits

    best_i = max(range(pop_size), key=lambda i: fits[i])
    return {"chrom": pop[best_i], "fitness": fits[best_i], "success": fits[best_i] == 0,
            "generations": len(best_hist) - 1, "evals": solved_evals or problem.evals,
            "time": time.time() - t0, "best_hist": best_hist, "mean_hist": mean_hist}

def energy(problem, state):
    return -problem.fitness(state)
def neighbor_state(problem, state):
    new = state[:]
    i = random.randrange(len(new))
    if random.random() < P_LOCAL:
        new[i] = problem.shift_gene(i, new[i])
    else:
        new[i] = random.randrange(len(problem.domains[i]))
    return new
def run_sa(problem, t_start=T_START, t_min=T_MIN, alpha=ALPHA, chain=CHAIN_LEN):
    t0 = time.time()
    problem.evals = 0
    cur = problem.random_chromosome()
    e_cur = energy(problem, cur)
    best, e_best = cur[:], e_cur
    hist = {"T": [], "cur": [], "best": [], "acc": []}
    solved_evals = None
    T = t_start
    while T > t_min and e_best > 0:
        accepted = 0
        for _ in range(chain):
            cand = neighbor_state(problem, cur)
            e_cand = energy(problem, cand)
            delta = e_cand - e_cur
            if delta <= 0 or random.random() < math.exp(-delta / T):
                cur, e_cur = cand, e_cand
                accepted += 1
                if e_cur < e_best:
                    best, e_best = cur[:], e_cur
                    if e_best == 0:
                        solved_evals = problem.evals
                        break
        hist["T"].append(T)
        hist["cur"].append(e_cur)
        hist["best"].append(e_best)
        hist["acc"].append(accepted / chain)
        T *= alpha
    return {"chrom": best, "fitness": -e_best, "success": e_best == 0,
            "steps": len(hist["T"]), "evals": solved_evals or problem.evals,
            "time": time.time() - t0, "hist": hist}

RULES = {"bounds": "правило 2 (вихід за межі поля)", "reef": "правило 3 (корабель на рифі)",
         "overlap": "правило 4 (кораблі перетинаються)", "touch": "правило 5 (кораблі торкаються)"}

def print_solution(problem, chrom, fitness):
    pl = problem.decode(chrom)
    for i, (length, (x, y, o)) in enumerate(zip(problem.ships, pl), 1):
        print(f"Корабель №{i:2d}: довжина {length}, перша клітинка ({x}, {y}), {ORIENT_NAME[o]}")
    print("Значення фітнес-функції:", fitness)
    v, _ = problem.analyse(pl)
    if fitness == 0:
        print("Розстановка правильна, порушень немає.")
    else:
        print("Правильну розстановку не знайдено. Порушені правила:")
        for key, name in RULES.items():
            if v[key]:
                print(f"  - {name}: {v[key]}")

def draw_board(problem, chrom, ax, title=""):
    n = problem.n
    pl = problem.decode(chrom)
    _, bad = problem.analyse(pl, collect=True)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    for y in range(n):
        for x in range(n):
            color = "#dff1fb"
            if (x, y) in problem.reefs:
                color = "#6b7a8f"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#9fb6c8", lw=0.8))
    palette = ["#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5", "#dbdb8d",
               "#9edae5", "#c7c7c7", "#f7b6d2", "#c49c94", "#8fd3c1"]
    for i, (length, p) in enumerate(zip(problem.ships, pl)):
        for (x, y) in problem.cells(length, p):
            if not (0 <= x < n and 0 <= y < n):
                continue
            color = "#e53935" if (x, y) in bad else palette[i % len(palette)]
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="black", lw=1.4))
            ax.text(x + 0.5, y + 0.5, str(i + 1), ha="center", va="center",
                    fontsize=11, fontweight="bold")
    ax.set_xticks([k + 0.5 for k in range(n)])
    ax.set_xticklabels(range(n))
    ax.set_yticks([k + 0.5 for k in range(n)])
    ax.set_yticklabels(range(n))
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_title(title, pad=24)
    ax.legend(handles=[Patch(facecolor="#6b7a8f", label="риф"),
                       Patch(facecolor="#e53935", label="порушення")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)

def show_ga(problem, res, out="ga_result.png"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_board(problem, res["chrom"], a1,
               f"ГА: F = {res['fitness']}, покоління {res['generations']}")
    a2.plot(res["best_hist"], label="найкраща", color="tab:green")
    a2.plot(res["mean_hist"], label="середня", color="tab:blue")
    a2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a2.set_xlabel("покоління")
    a2.set_ylabel("фітнес F")
    a2.grid(alpha=0.3)
    a2.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.show()

def show_sa(problem, res, out="sa_result.png"):
    h = res["hist"]
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 2, 1)
    draw_board(problem, res["chrom"], ax, f"SA: F = {res['fitness']}, ступенів T: {res['steps']}")
    ax1 = fig.add_subplot(3, 2, 2)
    ax1.plot(h["cur"], label="поточна E")
    ax1.plot(h["best"], label="найкраща E")
    ax1.set_ylabel("E")
    ax1.legend()
    ax2 = fig.add_subplot(3, 2, 4)
    ax2.plot(h["T"], color="tab:red")
    ax2.set_ylabel("T")
    ax3 = fig.add_subplot(3, 2, 6)
    ax3.plot(h["acc"], color="tab:purple")
    ax3.set_ylabel("частка прийнятих")
    ax3.set_xlabel("номер ступеня температури")
    for a in (ax1, ax2, ax3):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.show()

def summarize(results):
    ok = [r for r in results if r["success"]]
    rate = len(ok) / len(results)
    return {"rate": rate,
            "iters": statistics.mean([r.get("generations", r.get("steps")) for r in ok]) if ok else None,
            "evals": statistics.mean([r["evals"] for r in ok]) if ok else None,
            "time": statistics.mean([r["time"] for r in results])}
def research_ga(problem, runs=10):
    rows = []
    for m in (10, 20, 50, 100, 200):
        rows.append(("M", m, summarize([run_ga(problem, pop_size=m) for _ in range(runs)])))
    for pm in (0.01, 0.05, 0.1, 0.3, 0.6):
        rows.append(("pm", pm, summarize([run_ga(problem, pm=pm) for _ in range(runs)])))
    return rows
def research_sa(problem, runs=10):
    return [("alpha", a, summarize([run_sa(problem, alpha=a) for _ in range(runs)]))
            for a in (0.50, 0.80, 0.90, 0.95, 0.98)]
def compare(problem, runs=30):
    ga = summarize([run_ga(problem) for _ in range(runs)])
    sa = summarize([run_sa(problem) for _ in range(runs)])
    return ga, sa
def print_rows(rows):
    for name, val, s in rows:
        it = "-" if s["iters"] is None else f"{s['iters']:.1f}"
        ev = "-" if s["evals"] is None else f"{s['evals']:.0f}"
        print(f"{name:6s} = {val:<6} | успіх {s['rate'] * 100:5.1f}% | ітерацій {it:>7} "
              f"| обчислень F {ev:>8} | час {s['time']:.2f} с")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", nargs="?", default="variant03.json")
    ap.add_argument("--mode", choices=["ga", "sa", "research", "compare"], default="ga")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    problem = Problem(*load_variant(args.variant))
    print(f"{STUDENT_NAME}, варіант {VARIANT_NUMBER}; поле {problem.n}x{problem.n}, "
          f"кораблів: {len(problem.ships)}, рифів: {len(problem.reefs)}")
    if args.mode == "ga":
        res = run_ga(problem)
        print_solution(problem, res["chrom"], res["fitness"])
        print(f"Поколінь: {res['generations']}, обчислень F: {res['evals']}, час: {res['time']:.2f} с")
        show_ga(problem, res)
    elif args.mode == "sa":
        res = run_sa(problem)
        print_solution(problem, res["chrom"], res["fitness"])
        print(f"Ступенів T: {res['steps']}, обчислень E: {res['evals']}, час: {res['time']:.2f} с")
        show_sa(problem, res)
    elif args.mode == "research":
        print("--- вплив параметрів ГА ---")
        print_rows(research_ga(problem, args.runs))
        print("--- вплив швидкості охолодження α ---")
        print_rows(research_sa(problem, args.runs))
    else:
        ga, sa = compare(problem, args.runs)
        print_rows([("GA", "", ga), ("SA", "", sa)])

if __name__ == "__main__":
    main()