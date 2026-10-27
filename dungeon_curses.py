#!/usr/bin/env python3
"""
DUNGEON HEIST - a terminal roguelike with a minimax-powered rival.

You (K) race a scheming goblin (G) for treasure in a procedurally generated,
torch-lit dungeon. Dodge spikes, smash skeletons, grab power-ups, and beat the
goblin's loot count on every level. Each level the dungeon gets nastier and the
goblin looks further ahead.

Run:  python3 dungeon_heist.py      (terminal at least 52x20)
"""
import curses
import math
import random
from collections import deque

# ==========================================
# --- 1. CONSTANTS ---
# ==========================================
ROWS, COLS = 11, 21
TORCH = 6                       # how far your torch lights the dungeon
MAX_HP = 6
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COIN, GEM = 10, 30
SKELETON_BOUNTY = 5
MIN_H, MIN_W = 20, 52

# color pair ids
KNIGHT, ENEMY, GOLD, WALL, FLOOR, GEMC, GOOD = 1, 2, 3, 4, 5, 6, 7


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(p):
    return 0 <= p[0] < ROWS and 0 <= p[1] < COLS


def neighbors(p):
    for dr, dc in DIRS:
        n = (p[0] + dr, p[1] + dc)
        if in_bounds(n):
            yield n


def bfs(start, blocked):
    """Walking distance from `start` to every reachable cell."""
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for n in neighbors(cur):
            if n not in blocked and n not in dist:
                dist[n] = dist[cur] + 1
                q.append(n)
    return dist


def can_see(a, b, walls):
    """Bresenham line of sight; walls block view but are themselves visible."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        if (x, y) != a and (x, y) in walls:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True


# ==========================================
# --- 2. LEVEL GENERATION ---
# ==========================================
class Skeleton:
    def __init__(self, pos, hp):
        self.pos = pos
        self.hp = hp
        self.tick = 0     # skeletons only act every other turn
        self.skip = 0     # extra turns to lose (stagger / freeze)


def generate_level(n):
    p_start, g_start = (0, 0), (ROWS - 1, COLS - 1)

    while True:
        density = min(0.12 + 0.01 * n, 0.20)
        walls = set()
        for r in range(ROWS):
            for c in range(COLS):
                cell = (r, c)
                if manhattan(cell, p_start) <= 2 or manhattan(cell, g_start) <= 2:
                    continue
                if random.random() < density:
                    walls.add(cell)
                    if random.random() < 0.5:        # clump walls together
                        nb = random.choice(list(neighbors(cell)))
                        if manhattan(nb, p_start) > 2 and manhattan(nb, g_start) > 2:
                            walls.add(nb)
        reach = bfs(p_start, walls)
        if g_start in reach and len(reach) >= 80:
            break

    # seal any unreachable pockets so nothing spawns where nobody can go
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) not in reach:
                walls.add((r, c))

    dp = reach
    dg = bfs(g_start, walls)
    free = [c for c in reach if c not in (p_start, g_start)]
    random.shuffle(free)

    def take(count, pred):
        out = []
        for c in free[:]:
            if len(out) == count:
                break
            if pred(c):
                out.append(c)
                free.remove(c)
        return out

    loot_cells = take(min(6 + n, 12), lambda c: dp[c] >= 3 and dg.get(c, 0) >= 3)
    loot = {}
    for i, c in enumerate(loot_cells):
        loot[c] = GEM if (i == 0 or random.random() < 0.3) else COIN

    traps = set(take(min(2 + n, 8), lambda c: dp[c] >= 3 and dg.get(c, 0) >= 3))

    kinds = ['+', 'B', 'Z']
    random.shuffle(kinds)
    if n >= 4:
        kinds.append(random.choice(['+', 'B', 'Z']))
    items = dict(zip(take(len(kinds), lambda c: True), kinds))

    hp = 2 if n < 5 else 3
    skel_cells = take(min(n, 5), lambda c: dp[c] >= 9)
    if len(skel_cells) < min(n, 5):
        skel_cells += take(min(n, 5) - len(skel_cells), lambda c: dp[c] >= 5)
    skeletons = [Skeleton(c, hp) for c in skel_cells]

    return dict(walls=walls, loot=loot, traps=traps, items=items,
                skeletons=skeletons, p_start=p_start, g_start=g_start)


# ==========================================
# --- 3. GOBLIN AI (MINIMAX + ALPHA-BETA) ---
# ==========================================
def moves_for(pos, other, blocked):
    out = []
    for n in neighbors(pos):
        if n not in blocked and n != other:
            out.append(n)
    return out or [pos]          # boxed in? pass the turn


def evaluate(g_score, p_score, g_pos, p_pos, loot, dmaps):
    """Points dominate; otherwise reward being closer to the best loot than you are."""
    score = (g_score - p_score) * 1000
    if loot:
        g_pot = max(v - 2 * dmaps[pos].get(g_pos, 99) for pos, v in loot.items())
        p_pot = max(v - 2 * dmaps[pos].get(p_pos, 99) for pos, v in loot.items())
        score += g_pot - p_pot
    return score


def minimax(p_pos, g_pos, p_score, g_score, loot, depth, maximizing,
            alpha, beta, blocked, dmaps):
    if depth == 0 or not loot:
        return evaluate(g_score, p_score, g_pos, p_pos, loot, dmaps), None

    if maximizing:                                   # goblin's turn
        best, best_move = -math.inf, None
        for m in moves_for(g_pos, p_pos, blocked):
            gain = loot.get(m, 0)
            new_loot = {k: v for k, v in loot.items() if k != m} if gain else loot
            s, _ = minimax(p_pos, m, p_score, g_score + gain, new_loot,
                           depth - 1, False, alpha, beta, blocked, dmaps)
            if s > best:
                best, best_move = s, m
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best, best_move

    best = math.inf                                  # knight's turn
    for m in moves_for(p_pos, g_pos, blocked):
        gain = loot.get(m, 0)
        new_loot = {k: v for k, v in loot.items() if k != m} if gain else loot
        s, _ = minimax(m, g_pos, p_score + gain, g_score, new_loot,
                       depth - 1, True, alpha, beta, blocked, dmaps)
        best = min(best, s)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best, None


# ==========================================
# --- 4. GAME STATE & RULES ---
# ==========================================
class Game:
    def __init__(self):
        self.level_no = 0
        self.hp = MAX_HP
        self.total = 0
        self.kills = 0
        self.log = []
        self.dead = False
        self.death_reason = ""
        self.hide_goblin = False     # hard mode: goblin vanishes in the fog (toggle with H)
        self.next_level()

    def say(self, msg):
        self.log.append(msg)
        self.log = self.log[-3:]

    def next_level(self):
        self.level_no += 1
        d = generate_level(self.level_no)
        self.walls, self.loot, self.traps = d['walls'], d['loot'], d['traps']
        self.items, self.skeletons = d['items'], d['skeletons']
        self.p_pos, self.g_pos = d['p_start'], d['g_start']
        self.p_score = self.g_score = 0
        self.g_stun = 0
        self.haste = 0
        self.explored = set()
        self.log = []
        n_sk = len(self.skeletons)
        self.say(f"Level {self.level_no}: {len(self.loot)} treasures, "
                 f"{n_sk} skeleton{'s' if n_sk != 1 else ''}, one greedy goblin.")

    # ---- vision ----
    def update_vision(self):
        vis = set()
        pr, pc = self.p_pos
        for r in range(max(0, pr - TORCH), min(ROWS, pr + TORCH + 1)):
            for c in range(max(0, pc - TORCH), min(COLS, pc + TORCH + 1)):
                if (r - pr) ** 2 + (c - pc) ** 2 <= TORCH * TORCH + 1 \
                        and can_see(self.p_pos, (r, c), self.walls):
                    vis.add((r, c))
        self.explored |= vis
        return vis

    def hurt(self, amount, reason):
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
            self.dead = True
            self.death_reason = reason

    # ---- player ----
    def player_act(self, dr, dc):
        """Returns True if the action used up a turn."""
        if dr == 0 and dc == 0:
            self.say("You catch your breath.")
            return True
        t = (self.p_pos[0] + dr, self.p_pos[1] + dc)
        if not in_bounds(t) or t in self.walls:
            return False

        for s in self.skeletons:
            if s.pos == t:
                self.hit_skeleton(s)
                return True
        if t == self.g_pos:
            self.hit_goblin()
            return True

        self.p_pos = t
        if t in self.loot:
            v = self.loot.pop(t)
            self.p_score += v
            self.say(f"You grab a {'GEM' if v >= GEM else 'coin'}! +{v}")
        if t in self.items:
            self.use_item(self.items.pop(t))
        if t in self.traps:
            self.say("SPIKES! You take 1 damage.")
            self.hurt(1, "Impaled on spikes")
        return True

    def use_item(self, kind):
        if kind == '+':
            self.hp = min(MAX_HP, self.hp + 2)
            self.say("Healing potion! +2 HP.")
        elif kind == 'B':
            self.haste = 8
            self.say("Speed boots! Enemies lose every other turn.")
        elif kind == 'Z':
            self.g_stun = max(self.g_stun, 4)
            for s in self.skeletons:
                s.skip = 4
            self.say("Freeze scroll! Everything stops for 4 turns.")

    def hit_skeleton(self, s):
        s.hp -= 1
        if s.hp <= 0:
            self.skeletons.remove(s)
            self.kills += 1
            self.p_score += SKELETON_BOUNTY
            self.say(f"You shatter a skeleton! +{SKELETON_BOUNTY}")
        else:
            s.skip = max(s.skip, 1)
            self.say("You crack the skeleton. It staggers!")

    def hit_goblin(self):
        self.g_stun = max(self.g_stun, 3)
        if self.g_score >= COIN:
            spots = [n for n in neighbors(self.g_pos)
                     if n not in self.walls and n != self.p_pos and n not in self.loot
                     and all(s.pos != n for s in self.skeletons)]
            self.g_score -= COIN
            if spots:
                self.loot[random.choice(spots)] = COIN
                self.say("You club the goblin! It drops a coin and is stunned.")
            else:
                self.say("You club the goblin! The coin is lost forever.")
        else:
            self.say("You bonk the goblin. It is stunned!")

    # ---- enemies ----
    def enemy_phase(self):
        if self.haste > 0:
            self.haste -= 1
            if self.haste % 2 == 0:
                return                       # boots: enemies lose this turn
        self.goblin_turn()
        if self.loot:
            self.skeleton_turns()

    def goblin_turn(self):
        if self.g_stun > 0:
            self.g_stun -= 1
            return
        sk_cells = {s.pos for s in self.skeletons}
        blocked = self.walls | self.traps | sk_cells
        dmaps = {pos: bfs(pos, self.walls | self.traps) for pos in self.loot}
        depth = min(3 + self.level_no, 7)    # smarter every level
        _, move = minimax(self.p_pos, self.g_pos, self.p_score, self.g_score,
                          dict(self.loot), depth, True, -math.inf, math.inf,
                          blocked, dmaps)
        if move and move != self.g_pos:
            self.g_pos = move
            if move in self.loot:
                v = self.loot.pop(move)
                self.g_score += v
                self.say(f"The goblin snatches a {'GEM' if v >= GEM else 'coin'}! (+{v})")

    def skeleton_turns(self):
        if not self.skeletons:
            return
        dmap = bfs(self.p_pos, self.walls | self.traps)
        for s in list(self.skeletons):
            if s.skip > 0:
                s.skip -= 1
                continue
            s.tick ^= 1
            if s.tick:
                continue                     # slow: acts every other turn
            if manhattan(s.pos, self.p_pos) == 1:
                self.say("A skeleton rakes you! -1 HP")
                self.hurt(1, "Hacked apart by a skeleton")
                if self.dead:
                    return
                continue
            occupied = {o.pos for o in self.skeletons if o is not s} | {self.g_pos}
            best, best_d = None, dmap.get(s.pos, 99)
            for n in neighbors(s.pos):
                if n in self.walls or n in self.traps or n in occupied or n == self.p_pos:
                    continue
                d = dmap.get(n, 99)
                if d < best_d:
                    best, best_d = n, d
            if best:
                s.pos = best


# ==========================================
# --- 5. TERMINAL GRAPHICS ---
# ==========================================
def C(n):
    return curses.color_pair(n)


def put(win, y, x, text, attr=0):
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def draw(stdscr, g, vis):
    stdscr.erase()
    put(stdscr, 0, 2, "D U N G E O N   H E I S T", C(KNIGHT) | curses.A_BOLD)
    put(stdscr, 0, 32, f"LEVEL {g.level_no}", C(GOLD) | curses.A_BOLD)

    hearts = "#" * g.hp + "." * (MAX_HP - g.hp)
    put(stdscr, 1, 2, "HP [", C(ENEMY) | curses.A_BOLD)
    put(stdscr, 1, 6, hearts, C(ENEMY) | curses.A_BOLD)
    put(stdscr, 1, 6 + MAX_HP, "]", C(ENEMY) | curses.A_BOLD)
    status = []
    if g.haste > 0:
        status.append(f"HASTE {g.haste}")
    if g.g_stun > 0:
        status.append(f"GOBLIN STUNNED {g.g_stun}")
    put(stdscr, 1, 16, "  ".join(status), C(GOOD) | curses.A_BOLD)

    put(stdscr, 2, 2, f"Knight {g.p_score:<3}", C(KNIGHT) | curses.A_BOLD)
    put(stdscr, 2, 14, f"Goblin {g.g_score:<3}", C(ENEMY) | curses.A_BOLD)
    put(stdscr, 2, 26, f"Loot left {len(g.loot)}", C(GOLD))
    put(stdscr, 2, 39, f"Total {g.total}", C(WALL))

    put(stdscr, 3, 2, "+" + "-" * (COLS * 2) + "+", C(WALL))
    put(stdscr, 4 + ROWS, 2, "+" + "-" * (COLS * 2) + "+", C(WALL))
    for r in range(ROWS):
        put(stdscr, 4 + r, 2, "|", C(WALL))
        put(stdscr, 4 + r, 3 + COLS * 2, "|", C(WALL))

    sk_cells = {s.pos for s in g.skeletons}
    for r in range(ROWS):
        for c in range(COLS):
            pos = (r, c)
            seen, known = pos in vis, pos in g.explored
            ch, attr = ' ', 0
            if pos == g.p_pos:
                ch, attr = 'K', C(KNIGHT) | curses.A_BOLD
            elif pos in g.loot:      # treasure always glows through the dark
                if g.loot[pos] >= GEM:
                    ch, attr = '*', C(GEMC) | curses.A_BOLD
                else:
                    ch, attr = '$', C(GOLD) | curses.A_BOLD
            elif pos == g.g_pos and (seen or not g.hide_goblin):
                ch = 'g' if g.g_stun > 0 else 'G'
                attr = (C(GOLD) if g.g_stun > 0 else C(ENEMY)) | curses.A_BOLD
            elif seen and pos in sk_cells:
                ch, attr = 'S', C(WALL) | curses.A_BOLD
            elif known:
                dim = 0 if seen else curses.A_DIM
                if pos in g.walls:
                    ch, attr = '#', C(WALL) | dim
                elif pos in g.traps:
                    ch, attr = '^', C(ENEMY) | dim
                elif pos in g.items:
                    ch, attr = g.items[pos], C(GOOD) | curses.A_BOLD
                else:
                    ch, attr = '.', C(FLOOR) | dim
            put(stdscr, 4 + r, 3 + c * 2, ch, attr)

    # legend, if the terminal is wide enough
    lx = 3 + COLS * 2 + 4
    if stdscr.getmaxyx()[1] >= lx + 24:
        legend = [("K  you", KNIGHT), ("G  goblin (g = stunned)", ENEMY),
                  ("S  skeleton", WALL), ("$  coin 10", GOLD), ("*  gem 30", GEMC),
                  ("^  spikes", ENEMY), ("+  heal", GOOD), ("B  speed boots", GOOD),
                  ("Z  freeze scroll", GOOD)]
        for i, (txt, col) in enumerate(legend):
            put(stdscr, 4 + i, lx, txt, C(col))

    for i, msg in enumerate(g.log):
        newest = (i == len(g.log) - 1)
        put(stdscr, 5 + ROWS + i, 2, msg[:max(10, stdscr.getmaxyx()[1] - 3)],
            C(WALL) | (curses.A_BOLD if newest else curses.A_DIM))
    put(stdscr, 5 + ROWS + 3, 2,
        "[WASD/Arrows] move/attack  [Space] wait  [H] goblin fog  [Q] quit", C(FLOOR))
    stdscr.refresh()


def banner(stdscr, lines, wait=True):
    """Centered text screen. `lines` is a list of (text, color_id or 0, bold)."""
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    top = max(0, (h - len(lines)) // 2)
    for i, (text, col, bold) in enumerate(lines):
        attr = (C(col) if col else 0) | (curses.A_BOLD if bold else 0)
        put(stdscr, top + i, max(0, (w - len(text)) // 2), text[:w - 1], attr)
    stdscr.refresh()
    return stdscr.getch() if wait else None


def too_small(stdscr):
    h, w = stdscr.getmaxyx()
    return h < MIN_H or w < MIN_W


# ==========================================
# --- 6. GAME LOOP ---
# ==========================================
KEYMAP = {
    curses.KEY_UP: (-1, 0), ord('w'): (-1, 0), ord('W'): (-1, 0),
    curses.KEY_DOWN: (1, 0), ord('s'): (1, 0), ord('S'): (1, 0),
    curses.KEY_LEFT: (0, -1), ord('a'): (0, -1), ord('A'): (0, -1),
    curses.KEY_RIGHT: (0, 1), ord('d'): (0, 1), ord('D'): (0, 1),
    ord(' '): (0, 0), ord('.'): (0, 0),
}


def play_level(stdscr, g):
    """Returns 'clear', 'dead' or 'quit'."""
    while True:
        if too_small(stdscr):
            k = banner(stdscr, [(f"Please enlarge your terminal to {MIN_W}x{MIN_H}+", ENEMY, True),
                                ("(press Q to quit)", 0, False)])
            if k in (ord('q'), ord('Q')):
                return 'quit'
            continue

        vis = g.update_vision()
        draw(stdscr, g, vis)
        key = stdscr.getch()

        if key in (ord('q'), ord('Q')):
            return 'quit'
        if key in (ord('h'), ord('H')):
            g.hide_goblin = not g.hide_goblin
            g.say("Hard mode: the goblin hides in the dark." if g.hide_goblin
                  else "Easy mode: you can always see the goblin.")
            continue                         # toggling doesn't use a turn
        if key not in KEYMAP:
            continue
        if not g.player_act(*KEYMAP[key]):
            continue                         # bumped a wall: free retry

        if g.dead:
            draw(stdscr, g, g.update_vision())
            return 'dead'
        if not g.loot:
            return 'clear'

        g.enemy_phase()

        if g.dead:
            draw(stdscr, g, g.update_vision())
            return 'dead'
        if not g.loot:
            return 'clear'


def title_screen(stdscr):
    return banner(stdscr, [
        ("D U N G E O N   H E I S T", KNIGHT, True),
        ("", 0, False),
        ("You and a scheming goblin are racing for the same treasure.", 0, False),
        ("", 0, False),
        ("Out-loot the goblin on every level or the heist is over.", GOLD, True),
        ("Bump a skeleton twice to smash it (+5). Bump the goblin to stun it", 0, False),
        ("and make it drop a coin. Your torch is short; treasure and the goblin", 0, False),
        ("always stay visible (press H for hard mode: goblin hides in the dark).", 0, False),
        ("Spikes ^ hurt.  + heals   B speed boots   Z freeze scroll", GOOD, False),
        ("Each level: more skeletons, more traps, and a smarter goblin.", ENEMY, False),
        ("", 0, False),
        ("WASD / arrows to move   Space to wait   Q to quit", FLOOR, False),
        ("", 0, False),
        ("Press any key to begin", GOLD, True),
    ])


def main(stdscr):
    curses.curs_set(0)
    curses.start_color()
    for pair, col in [(KNIGHT, curses.COLOR_CYAN), (ENEMY, curses.COLOR_RED),
                      (GOLD, curses.COLOR_YELLOW), (WALL, curses.COLOR_WHITE),
                      (FLOOR, curses.COLOR_BLUE), (GEMC, curses.COLOR_MAGENTA),
                      (GOOD, curses.COLOR_GREEN)]:
        curses.init_pair(pair, col, curses.COLOR_BLACK)

    if title_screen(stdscr) in (ord('q'), ord('Q')):
        return
    best = 0

    while True:
        g = Game()
        reason = ""
        while True:
            res = play_level(stdscr, g)
            if res == 'quit':
                return
            if res == 'dead':
                reason = g.death_reason or "You were slain"
                break

            # level finished: the goblin must not out-loot you
            if g.p_score < g.g_score:
                reason = "The goblin escaped with more loot than you"
                break
            g.total += g.p_score
            levels_done = g.level_no
            heal = "" if g.hp >= MAX_HP else "  (+1 HP)"
            g.hp = min(MAX_HP, g.hp + 1)
            banner(stdscr, [
                (f"LEVEL {levels_done} CLEARED", GOOD, True), ("", 0, False),
                (f"Knight {g.p_score}   vs   Goblin {g.g_score}", GOLD, True), ("", 0, False),
                (f"Total score: {g.total}{heal}", 0, False), ("", 0, False),
                ("Deeper we go... press any key", FLOOR, False),
            ])
            g.next_level()

        final = g.total + (g.p_score if g.dead else 0)
        best = max(best, final)
        k = banner(stdscr, [
            ("G A M E   O V E R", ENEMY, True), ("", 0, False),
            (reason, 0, True), ("", 0, False),
            (f"Reached level {g.level_no}   Skeletons smashed: {g.kills}", 0, False),
            (f"Final score: {final}   Best this session: {best}", GOLD, True), ("", 0, False),
            ("Play again? (y / n)", KNIGHT, True),
        ])
        if k not in (ord('y'), ord('Y')):
            return


if __name__ == "__main__":
    curses.wrapper(main)