#!/usr/bin/env python3
"""
DUNGEON HEIST - simple edition (pygame)

Just you and a goblin racing for treasure. No skeletons, no spikes, no HP.
The goblin plays with MINIMAX (with alpha-beta pruning) and the game lets you
watch it think:

  * the coloured numbered path on the map is the line of play the goblin expects
    (red = goblin moves, blue = the replies it expects from YOU)
  * the "Goblin Brain" panel shows how deep it looked and how much work it did
  * press E for a picture-and-words explanation of minimax

Install:  pip install pygame
Run:      python3 dungeon_heist_pygame_simple.py

Controls: WASD / arrows = move (bump the goblin to stun it)   Space = wait
          M = show/hide the goblin's plan   E = how minimax works
          T = search tree: see exactly which branches were pruned
          F = fog of war on/off
          H = hard mode (goblin hides in the dark, only works with fog on)
          Esc = quit
"""
import math
import random
import sys
import time
from collections import deque

import pygame

# ==========================================
# --- 1. CONSTANTS ---
# ==========================================
ROWS, COLS = 11, 21
TILE = 40
MAP_W, MAP_H = COLS * TILE, ROWS * TILE
HUD_H, LOG_H = 84, 120
WIDTH, HEIGHT = MAP_W, HUD_H + MAP_H + LOG_H
FPS = 60

TORCH = 6
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COIN, GEM = 10, 30
MAX_DEPTH = 7
TRACE_PLY = 2          # how many levels of the search get recorded for the tree view (T key)

C_BG = (18, 18, 26)
C_PANEL = (24, 24, 36)
C_TEXT = (235, 235, 245)
C_DIM = (140, 140, 160)
C_GOLD = (255, 215, 0)
C_GEM = (220, 90, 240)
C_RED = (235, 70, 70)
C_BLUE = (90, 170, 255)
C_GREEN = (90, 230, 120)
C_CYAN = (120, 230, 255)

SHADOW = None
GLOWS = {}


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
    """Walking distance from `start` to every reachable cell (walls in the way)."""
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

    for r in range(ROWS):                            # seal unreachable pockets
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

    kinds = ['B', 'Z']
    if n >= 4:
        kinds.append(random.choice(['B', 'Z']))
    items = dict(zip(take(len(kinds), lambda c: True), kinds))

    return dict(walls=walls, loot=loot, items=items, p_start=p_start, g_start=g_start)


# ==========================================
# --- 3. THE GOBLIN'S BRAIN: MINIMAX ---
# ==========================================
#
# THE IDEA
#   The goblin can't know what you'll do, so it assumes you will play the move
#   that is WORST for it. It imagines the game a few moves into the future:
#
#        goblin moves -> you reply -> goblin moves -> you reply -> ...
#
#   and scores each imagined ending. Then it walks back up the tree:
#     * on ITS turns it picks the child with the HIGHEST score   (maximize)
#     * on YOUR turns it assumes you pick the LOWEST score        (minimize)
#   The move at the root with the best guaranteed outcome is what it plays.
#
# ALPHA-BETA PRUNING
#   alpha = best score the goblin is already guaranteed somewhere above
#   beta  = best (lowest) score you are already guaranteed somewhere above
#   If beta <= alpha, the rest of this branch can't change the final decision,
#   so we skip it. Same answer, far less work.
#
class SearchStats:
    """Counters so the game can SHOW how much work the AI did."""

    def __init__(self):
        self.nodes = 0     # imagined positions examined
        self.pruned = 0    # sibling branches skipped by alpha-beta


def moves_for(pos, other, walls):
    """Adjacent tiles a piece can step to (not walls, not the other piece)."""
    out = [n for n in neighbors(pos) if n not in walls and n != other]
    return out or [pos]          # boxed in? pass the turn


# Loot collected EARLIER in an imagined line counts very slightly more than loot collected
# later. Without this, 'grab it now' and 'grab it in 3 moves' tie and the goblin procrastinates.
# (It is baked into the score the search carries so alpha-beta stays correct.)
TIME_BONUS = 0.001


def evaluate(g_score, p_score, g_pos, p_pos, loot, dmaps):
    """
    Score an imagined position from the GOBLIN's point of view (bigger = better).

      1. Points matter most:      (goblin points - your points) * 1000
      2. Tie-breaker when we can't see any further: who is closer to the juiciest
         remaining loot?  value - 2*walking distance, for each side.
    """
    score = (g_score - p_score) * 1000
    if loot:
        g_pot = max(v - 2 * dmaps[pos].get(g_pos, 99) for pos, v in loot.items())
        p_pot = max(v - 2 * dmaps[pos].get(p_pos, 99) for pos, v in loot.items())
        score += g_pot - p_pot
    # Tie-breaker: even if every imagined line "loses" the loot to you, prefer to end
    # up closer to it. Without this the goblin shuffles back and forth in place.
    if dmaps:
        score -= 0.5 * min(dm.get(g_pos, 99) for dm in dmaps.values())
    return score


def minimax(p_pos, g_pos, p_score, g_score, loot, depth, maximizing,
            alpha, beta, walls, dmaps, stats, trace=None, ply=0):
    """
    Returns (score, line). `line` is the best sequence of moves found, as a list of
    ('G', tile) / ('K', tile) entries, so the game can draw the AI's plan.

    `trace` (optional) is a list that receives one record per branch looked at in the
    top TRACE_PLY levels, including branches that were PRUNED, so the game can draw the tree.
    """
    stats.nodes += 1

    # Out of lookahead, or nothing left to win: score this imagined position.
    if depth == 0 or not loot:
        return evaluate(g_score, p_score, g_pos, p_pos, loot, dmaps), []

    recording = trace is not None and ply < TRACE_PLY

    def record(who, move, pruned):
        node = dict(who=who, pos=move, score=None, kids=[], pruned=pruned, cut=False, best=False)
        trace.append(node)
        return node

    if maximizing:                                   # --- GOBLIN's turn: pick the MAX
        best, best_line, best_node = -math.inf, [], None
        options = moves_for(g_pos, p_pos, walls)
        for i, move in enumerate(options):
            gain = loot.get(move, 0)                 # imagine stepping there (and grabbing loot)
            new_loot = {k: v for k, v in loot.items() if k != move} if gain else loot
            node = record('G', move, False) if recording else None
            score, line = minimax(p_pos, move, p_score, g_score + gain * (1 + TIME_BONUS * depth), new_loot,
                                  depth - 1, False, alpha, beta, walls, dmaps, stats,
                                  node['kids'] if node else None, ply + 1)
            if node:
                node['score'] = score
                node['cut'] = any(k['pruned'] for k in node['kids'])
            if score > best:
                best, best_line, best_node = score, [('G', move)] + line, node
            alpha = max(alpha, best)
            if beta <= alpha:                        # you would never allow this branch
                stats.pruned += len(options) - i - 1
                if recording:
                    for rest in options[i + 1:]:
                        record('G', rest, True)
                break
        if best_node:
            best_node['best'] = True
        return best, best_line

    best, best_line, best_node = math.inf, [], None  # --- KNIGHT's turn: assume the MIN
    options = moves_for(p_pos, g_pos, walls)
    for i, move in enumerate(options):
        gain = loot.get(move, 0)
        new_loot = {k: v for k, v in loot.items() if k != move} if gain else loot
        node = record('K', move, False) if recording else None
        score, line = minimax(move, g_pos, p_score + gain * (1 + TIME_BONUS * depth), g_score, new_loot,
                              depth - 1, True, alpha, beta, walls, dmaps, stats,
                              node['kids'] if node else None, ply + 1)
        if node:
            node['score'] = score
            node['cut'] = any(k['pruned'] for k in node['kids'])
        if score < best:
            best, best_line, best_node = score, [('K', move)] + line, node
        beta = min(beta, best)
        if beta <= alpha:                            # goblin has a better option elsewhere
            stats.pruned += len(options) - i - 1
            if recording:
                for rest in options[i + 1:]:
                    record('K', rest, True)
            break
    if best_node:
        best_node['best'] = True
    return best, best_line


# ==========================================
# --- 4. GAME STATE & RULES ---
# ==========================================
class Game:
    """Pure game rules. Visual events are queued in self.fx for the renderer."""

    def __init__(self):
        self.level_no = 0
        self.total = 0
        self.log = []
        self.fx = []
        self.hide_goblin = False
        self.brain = None
        self.next_level()

    def say(self, msg):
        self.log.append(msg)
        self.log = self.log[-3:]

    def emit(self, *event):
        self.fx.append(event)

    def ai_depth(self):
        return min(3 + self.level_no, MAX_DEPTH)     # smarter every level

    def next_level(self):
        self.level_no += 1
        d = generate_level(self.level_no)
        self.walls, self.loot, self.items = d['walls'], d['loot'], d['items']
        self.p_pos, self.g_pos = d['p_start'], d['g_start']
        self.p_score = self.g_score = 0
        self.g_stun = 0
        self.haste = 0
        self.explored = set()
        self.brain = None
        self.log = []
        self.fx = []
        self.say(f"Level {self.level_no}: {len(self.loot)} treasures. "
                 f"The goblin looks {self.ai_depth()} moves ahead.")

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

    # ---- player ----
    def player_act(self, dr, dc):
        """Returns True if the action used up a turn."""
        if dr == 0 and dc == 0:
            self.say("You catch your breath.")
            return True
        t = (self.p_pos[0] + dr, self.p_pos[1] + dc)
        if not in_bounds(t) or t in self.walls:
            return False

        if t == self.g_pos:
            self.emit('lunge', (dr, dc))
            self.hit_goblin()
            return True

        self.p_pos = t
        if t in self.loot:
            v = self.loot.pop(t)
            self.p_score += v
            col = C_GEM if v >= GEM else C_GOLD
            self.emit('burst', t, col, 16)
            self.emit('text', t, f"+{v}", col)
            self.say(f"You grab a {'GEM' if v >= GEM else 'coin'}! +{v}")
        if t in self.items:
            self.use_item(self.items.pop(t))
        return True

    def use_item(self, kind):
        if kind == 'B':
            self.haste = 8
            self.emit('burst', self.p_pos, C_CYAN, 18)
            self.emit('text', self.p_pos, "HASTE!", C_CYAN)
            self.say("Speed boots! The goblin loses every other turn.")
        elif kind == 'Z':
            self.g_stun = max(self.g_stun, 4)
            self.emit('burst', self.p_pos, C_CYAN, 34)
            self.emit('text', self.p_pos, "FREEZE!", C_CYAN)
            self.say("Freeze scroll! The goblin is frozen for 4 turns.")

    def hit_goblin(self):
        self.g_stun = max(self.g_stun, 3)
        self.emit('burst', self.g_pos, C_GOLD, 14)
        self.emit('text', self.g_pos, "STUN!", C_GOLD)
        self.emit('shake', 4)
        if self.g_score >= COIN:
            spots = [n for n in neighbors(self.g_pos)
                     if n not in self.walls and n != self.p_pos and n not in self.loot]
            self.g_score -= COIN
            if spots:
                self.loot[random.choice(spots)] = COIN
                self.say("You club the goblin! It drops a coin and is stunned.")
            else:
                self.say("You club the goblin! The coin is lost forever.")
        else:
            self.say("You bonk the goblin. It is stunned!")

    # ---- goblin ----
    def enemy_phase(self):
        if self.haste > 0:
            self.haste -= 1
            if self.haste % 2 == 0:
                return                       # boots: goblin loses this turn
        self.goblin_turn()

    def goblin_turn(self):
        if self.g_stun > 0:
            self.g_stun -= 1
            self.brain = dict(stunned=True)
            return

        dmaps = {pos: bfs(pos, self.walls) for pos in self.loot}
        depth = self.ai_depth()
        stats = SearchStats()
        origin_g, origin_k = self.g_pos, self.p_pos

        tree = []
        loot_now = dict(self.loot)
        t0 = time.perf_counter()
        score, line = minimax(self.p_pos, self.g_pos, self.p_score, self.g_score,
                              dict(self.loot), depth, True, -math.inf, math.inf,
                              self.walls, dmaps, stats, tree, 0)
        ms = (time.perf_counter() - t0) * 1000

        self.brain = dict(stunned=False, depth=depth, nodes=stats.nodes, pruned=stats.pruned,
                          score=score, line=line, ms=ms, origin_g=origin_g, origin_k=origin_k,
                          tree=tree, loot=loot_now)

        move = line[0][1] if line else self.g_pos
        if move != self.g_pos:
            self.g_pos = move
            if move in self.loot:
                v = self.loot.pop(move)
                self.g_score += v
                self.emit('burst', move, C_RED, 14)
                self.emit('text', move, f"+{v}", C_RED)
                self.say(f"The goblin snatches a {'GEM' if v >= GEM else 'coin'}! (+{v})")


# ==========================================
# --- 5. SPRITES & DRAWING HELPERS ---
# ==========================================
def init_assets():
    global SHADOW
    SHADOW = pygame.Surface((28, 12), pygame.SRCALPHA)
    pygame.draw.ellipse(SHADOW, (0, 0, 0, 90), (0, 0, 28, 12))
    for name, col, rad in [("gold", C_GOLD, 26), ("gem", C_GEM, 28),
                           ("red", C_RED, 30), ("cyan", C_CYAN, 24)]:
        s = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
        steps = 12
        for i in range(steps, 0, -1):
            a = int(8 + 70 * (1 - i / steps) ** 1.6)
            pygame.draw.circle(s, (col[0], col[1], col[2], a), (rad, rad), int(rad * i / steps))
        GLOWS[name] = s


def draw_text(surf, font, msg, color, pos, anchor="topleft", shadow=True):
    img = font.render(msg, True, color)
    rect = img.get_rect(**{anchor: pos})
    if shadow:
        sh = font.render(msg, True, (0, 0, 0))
        surf.blit(sh, (rect.x + 2, rect.y + 2))
    surf.blit(img, rect)
    return rect


def wrap_text(font, text, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_knight(s, cx, cy, t):
    b = math.sin(t * 6) * 1.5
    s.blit(SHADOW, (cx - 14, cy + 10))
    pygame.draw.rect(s, (40, 50, 90), (cx - 7, cy + 8, 5, 7), border_radius=2)
    pygame.draw.rect(s, (40, 50, 90), (cx + 2, cy + 8, 5, 7), border_radius=2)
    pygame.draw.rect(s, (70, 140, 240), (cx - 9, cy - 4 + b, 18, 15), border_radius=5)
    pygame.draw.rect(s, (40, 80, 160), (cx - 9, cy + 5 + b, 18, 3))
    pygame.draw.circle(s, (190, 160, 60), (cx - 11, cy + 3 + b), 6)
    pygame.draw.circle(s, (230, 200, 90), (cx - 11, cy + 3 + b), 3)
    pygame.draw.circle(s, (205, 215, 230), (cx, cy - 9 + b), 8)
    pygame.draw.rect(s, (35, 40, 55), (cx - 5, cy - 10 + b, 10, 3))
    pygame.draw.ellipse(s, (220, 60, 70), (cx - 3, cy - 20 + b, 6, 9))
    pygame.draw.line(s, (225, 230, 240), (cx + 10, cy + 2 + b), (cx + 16, cy - 13 + b), 3)
    pygame.draw.line(s, (230, 190, 60), (cx + 7, cy - 1 + b), (cx + 14, cy + 1 + b), 3)


def draw_goblin(s, cx, cy, t, stunned=False):
    b = math.sin(t * 7 + 1) * 1.5
    s.blit(SHADOW, (cx - 14, cy + 10))
    pygame.draw.ellipse(s, C_RED, (cx - 15, cy + 8, 30, 10), 2)
    skin = (150, 190, 110) if stunned else (95, 185, 70)
    pygame.draw.rect(s, (90, 60, 40), (cx - 6, cy + 8, 4, 7))
    pygame.draw.rect(s, (90, 60, 40), (cx + 2, cy + 8, 4, 7))
    pygame.draw.rect(s, skin, (cx - 8, cy - 3 + b, 16, 14), border_radius=5)
    pygame.draw.rect(s, (120, 80, 50), (cx - 8, cy + 5 + b, 16, 5))
    pygame.draw.polygon(s, skin, [(cx - 7, cy - 10 + b), (cx - 18, cy - 14 + b), (cx - 8, cy - 4 + b)])
    pygame.draw.polygon(s, skin, [(cx + 7, cy - 10 + b), (cx + 18, cy - 14 + b), (cx + 8, cy - 4 + b)])
    pygame.draw.circle(s, skin, (cx, cy - 8 + b), 8)
    eye = (180, 180, 60) if stunned else (255, 70, 50)
    pygame.draw.circle(s, eye, (cx - 3, cy - 9 + b), 2)
    pygame.draw.circle(s, eye, (cx + 3, cy - 9 + b), 2)
    pygame.draw.line(s, (255, 255, 255), (cx - 3, cy - 4 + b), (cx + 3, cy - 4 + b), 1)
    pygame.draw.line(s, (200, 205, 215), (cx + 9, cy + 4 + b), (cx + 15, cy - 4 + b), 3)
    if stunned:
        for i in range(3):
            a = t * 5 + i * 2.09
            pygame.draw.circle(s, C_GOLD, (int(cx + math.cos(a) * 11), int(cy - 22 + math.sin(a) * 4)), 3)


def draw_coin(s, cx, cy, t, phase):
    bob = math.sin(t * 3 + phase) * 2
    s.blit(GLOWS["gold"], (cx - 26, int(cy - 26 + bob)))
    w = max(3, int(abs(math.cos(t * 3 + phase)) * 18))
    pygame.draw.ellipse(s, (200, 150, 0), (cx - w // 2 - 1, cy - 10 + bob, w + 2, 20))
    pygame.draw.ellipse(s, C_GOLD, (cx - w // 2, cy - 9 + bob, w, 18))
    if w > 9:
        pygame.draw.ellipse(s, (255, 240, 150), (cx - w // 4, cy - 5 + bob, max(2, w // 2), 10), 2)


def draw_gem(s, cx, cy, t, phase):
    bob = math.sin(t * 2.5 + phase) * 2.5
    s.blit(GLOWS["gem"], (cx - 28, int(cy - 28 + bob)))
    pts = [(cx, cy - 12 + bob), (cx + 10, cy - 3 + bob), (cx, cy + 12 + bob), (cx - 10, cy - 3 + bob)]
    pygame.draw.polygon(s, C_GEM, pts)
    pygame.draw.polygon(s, (255, 190, 255), [(cx, cy - 12 + bob), (cx + 10, cy - 3 + bob), (cx, cy - 3 + bob)])
    pygame.draw.polygon(s, (150, 40, 170), [(cx, cy - 3 + bob), (cx + 10, cy - 3 + bob), (cx, cy + 12 + bob)])
    pygame.draw.polygon(s, (255, 255, 255), pts, 1)
    if math.sin(t * 4 + phase) > 0.8:
        pygame.draw.line(s, (255, 255, 255), (cx + 8, cy - 14 + bob), (cx + 8, cy - 6 + bob), 2)
        pygame.draw.line(s, (255, 255, 255), (cx + 4, cy - 10 + bob), (cx + 12, cy - 10 + bob), 2)


def draw_item(s, kind, cx, cy, t):
    bob = math.sin(t * 3 + cx) * 2
    s.blit(GLOWS["cyan"], (cx - 24, int(cy - 24 + bob)))
    if kind == 'B':                                                                  # speed boots
        pygame.draw.polygon(s, (150, 95, 50), [(cx - 5, cy - 10 + bob), (cx + 3, cy - 10 + bob),
                                               (cx + 3, cy + 1 + bob), (cx + 11, cy + 5 + bob),
                                               (cx + 11, cy + 10 + bob), (cx - 5, cy + 10 + bob)])
        pygame.draw.rect(s, (230, 200, 90), (cx - 5, cy - 10 + bob, 8, 3))
        for i in range(3):
            pygame.draw.line(s, C_CYAN, (cx - 16, cy - 4 + i * 5 + bob), (cx - 8, cy - 4 + i * 5 + bob), 2)
    else:                                                                            # freeze scroll
        pygame.draw.rect(s, (205, 235, 255), (cx - 9, cy - 8 + bob, 18, 16), border_radius=3)
        pygame.draw.circle(s, (150, 190, 220), (cx - 9, cy + bob), 4)
        pygame.draw.circle(s, (150, 190, 220), (cx + 9, cy + bob), 4)
        for ang in range(3):
            a = ang * math.pi / 3
            dx, dy = math.cos(a) * 6, math.sin(a) * 6
            pygame.draw.line(s, (70, 130, 200), (cx - dx, cy - dy + bob), (cx + dx, cy + dy + bob), 2)


def build_map_surface(walls):
    """Pre-render floor and walls once per level."""
    surf = pygame.Surface((MAP_W, MAP_H))
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * TILE, r * TILE
            rnd = random.Random(r * 997 + c * 131)
            shade = 44 + ((r + c) % 2) * 4 + rnd.randint(0, 3)
            pygame.draw.rect(surf, (shade, shade - 2, shade + 14), (x, y, TILE, TILE))
            pygame.draw.rect(surf, (34, 33, 46), (x, y, TILE, TILE), 1)
            for _ in range(2):
                pygame.draw.circle(surf, (56, 54, 72), (x + rnd.randint(4, TILE - 4), y + rnd.randint(4, TILE - 4)), 1)
            if (r, c) in walls:
                pygame.draw.rect(surf, (60, 57, 74), (x, y + 4, TILE, TILE - 4))
                pygame.draw.rect(surf, (96, 92, 114), (x, y, TILE, TILE - 5))
                pygame.draw.line(surf, (128, 124, 148), (x, y), (x + TILE - 1, y), 2)
                for i, by in enumerate((y + 12, y + 24)):
                    pygame.draw.line(surf, (66, 63, 82), (x, by), (x + TILE, by), 1)
                    off = 10 if i % 2 == 0 else 24
                    pygame.draw.line(surf, (66, 63, 82), (x + off, by - 12), (x + off, by), 1)
                pygame.draw.rect(surf, (50, 48, 64), (x, y, TILE, TILE - 5), 1)
    return surf


# ==========================================
# --- 6. THE APP: VIEW, ANIMATION & STATES ---
# ==========================================
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dungeon Heist - Minimax")
        pygame.key.set_repeat(170, 95)
        self.clock = pygame.time.Clock()
        init_assets()
        self.f_tiny = pygame.font.SysFont("Arial", 14, bold=True)
        self.f_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.f_med = pygame.font.SysFont("Arial", 20, bold=True)
        self.f_big = pygame.font.SysFont("Arial", 30, bold=True)
        self.f_huge = pygame.font.SysFont("Arial", 62, bold=True)
        self.world = pygame.Surface((MAP_W, MAP_H))
        self.best = 0
        self.t = 0.0
        self.running = True
        self.show_plan = True
        self.fog = True                              # fog of war on/off (F key)
        self.over_reason = ""
        self.over_final = 0
        self.explain_return = "title"
        self.state_since = 0
        self.g = None
        self.new_game()
        self.state = "title"

    # ---- setup ----
    def new_game(self):
        self.g = Game()
        self.setup_level_view()
        self.set_state("play")

    def set_state(self, state):
        self.state = state
        self.state_since = pygame.time.get_ticks()

    def setup_level_view(self):
        g = self.g
        self.map_surf = build_map_surface(g.walls)
        self.pv = [float(g.p_pos[0]), float(g.p_pos[1])]
        self.gv = [float(g.g_pos[0]), float(g.g_pos[1])]
        self.p_off = [0.0, 0.0]
        self.light = {}
        self.light_k = 0.2
        self.particles, self.texts = [], []
        self.shake = 0.0
        self.fade = 1.0
        self.busy_until = 0
        self.enemy_at = None
        self.end_at = None
        self.vis = g.update_vision()

    # ---- input ----
    def on_key(self, key):
        now = pygame.time.get_ticks()
        if self.state == "explain":                  # any key leaves the explainer
            self.set_state(self.explain_return)
            return
        if self.state == "tree":                     # any key leaves the search tree
            self.set_state("play")
            return
        if key == pygame.K_ESCAPE:
            self.running = False
            return
        if key == pygame.K_e and self.state in ("title", "play"):
            self.explain_return = self.state
            self.set_state("explain")
            return
        if key == pygame.K_t and self.state == "play":
            b = self.g.brain
            if b and not b.get('stunned') and b.get('tree'):
                self.set_state("tree")
            else:
                self.g.say("No search to show yet - make a move first.")
            return
        if now - self.state_since < 500 and self.state != "play":
            return                                   # don't skip screens by key-repeat
        if self.state == "title":
            self.new_game()
        elif self.state == "play":
            self.on_play_key(key, now)
        elif self.state == "clear":
            self.g.next_level()
            self.setup_level_view()
            self.set_state("play")
        elif self.state == "over":
            if key in (pygame.K_y, pygame.K_RETURN, pygame.K_SPACE):
                self.new_game()
            elif key == pygame.K_n:
                self.running = False

    def on_play_key(self, key, now):
        g = self.g
        if key == pygame.K_h:
            g.hide_goblin = not g.hide_goblin
            if g.hide_goblin and not self.fog:
                g.say("Hard mode ON, but it only works while fog is on (press F).")
            else:
                g.say("Hard mode: the goblin hides in the dark." if g.hide_goblin
                      else "Easy mode: you can always see the goblin.")
            return
        if key == pygame.K_m:
            self.show_plan = not self.show_plan
            return
        if key == pygame.K_f:
            self.fog = not self.fog
            g.say("Fog of war ON." if self.fog else "Fog of war OFF: the whole dungeon is lit.")
            return
        moves = {pygame.K_UP: (-1, 0), pygame.K_w: (-1, 0), pygame.K_DOWN: (1, 0), pygame.K_s: (1, 0),
                 pygame.K_LEFT: (0, -1), pygame.K_a: (0, -1), pygame.K_RIGHT: (0, 1), pygame.K_d: (0, 1),
                 pygame.K_SPACE: (0, 0)}
        if key not in moves or now < self.busy_until or self.enemy_at or self.end_at:
            return
        if not g.player_act(*moves[key]):
            return                                   # bumped a wall: free retry
        self.busy_until = now + 110
        self.vis = g.update_vision()
        if not g.loot:
            self.end_at = now + 550
        else:
            self.enemy_at = now + 170                # goblin answers after your move animates

    def run_enemy_phase(self, now):
        g = self.g
        g.enemy_phase()
        self.vis = g.update_vision()
        if not g.loot:
            self.end_at = now + 550

    def finish_level(self):
        g = self.g
        if g.p_score < g.g_score:
            self.game_over("The goblin out-looted you!", g.total)
        else:
            g.total += g.p_score
            self.set_state("clear")

    def game_over(self, reason, final):
        self.over_reason, self.over_final = reason, final
        self.best = max(self.best, final)
        self.set_state("over")

    # ---- per-frame update ----
    def process_fx(self):
        g = self.g

        def px(pos):
            return pos[1] * TILE + TILE / 2, pos[0] * TILE + TILE / 2

        for ev in g.fx:
            kind = ev[0]
            if kind == 'burst':
                x, y = px(ev[1])
                for _ in range(ev[3]):
                    a = random.uniform(0, math.tau)
                    sp = random.uniform(40, 170)
                    self.particles.append(dict(x=x, y=y, vx=math.cos(a) * sp, vy=math.sin(a) * sp - 30,
                                               life=random.uniform(0.4, 0.85), max=0.85,
                                               color=ev[2], size=random.uniform(2, 4.5)))
            elif kind == 'text':
                x, y = px(ev[1])
                self.texts.append(dict(x=x, y=y - 14, text=ev[2], color=ev[3], life=1.0))
            elif kind == 'shake':
                self.shake = max(self.shake, ev[1])
            elif kind == 'lunge':
                self.p_off = [ev[1][0] * 0.38, ev[1][1] * 0.38]
        g.fx.clear()

    def update(self, dt):
        self.t += dt
        now = pygame.time.get_ticks()
        g = self.g
        if self.state in ("play", "clear", "over"):
            self.process_fx()
            k = min(1.0, dt * 16)
            for v, tgt in ((self.pv, g.p_pos), (self.gv, g.g_pos)):
                v[0] += (tgt[0] - v[0]) * k
                v[1] += (tgt[1] - v[1]) * k
            kk = min(1.0, dt * 14)
            self.p_off = [o - o * kk for o in self.p_off]
            self.light_k = min(1.0, dt * 9)

            for p in self.particles:
                p['x'] += p['vx'] * dt
                p['y'] += p['vy'] * dt
                p['vy'] += 260 * dt
                p['life'] -= dt
            self.particles = [p for p in self.particles if p['life'] > 0]
            for tx in self.texts:
                tx['y'] -= 38 * dt
                tx['life'] -= dt
            self.texts = [tx for tx in self.texts if tx['life'] > 0]
            self.shake = max(0.0, self.shake - dt * 28)
            self.fade = max(0.0, self.fade - dt * 2.2)

        if self.state == "play":
            if self.enemy_at and now >= self.enemy_at:
                self.enemy_at = None
                self.run_enemy_phase(now)
            if self.end_at and now >= self.end_at:
                self.end_at = None
                self.finish_level()

    # ---- drawing: world ----
    def center(self, r, c):
        return int(c * TILE + TILE / 2), int(r * TILE + TILE / 2)

    def draw_darkness(self):
        g = self.g
        small = pygame.Surface((COLS, ROWS), pygame.SRCALPHA)
        for r in range(ROWS):
            for c in range(COLS):
                pos = (r, c)
                if pos in self.vis:
                    d = math.hypot(r - self.pv[0], c - self.pv[1])
                    tgt = max(0.38, 1 - 0.6 * (d / TORCH) ** 1.4) + 0.05 * math.sin(self.t * 8 + d * 1.3)
                elif pos in g.explored:
                    tgt = 0.30
                else:
                    tgt = 0.0
                cur = self.light.get(pos, 0.0)
                cur += (tgt - cur) * self.light_k
                self.light[pos] = cur
                alpha = int(255 * (1 - min(1.0, max(0.0, cur))))
                small.set_at((c, r), (6, 6, 12, alpha))
        self.world.blit(pygame.transform.smoothscale(small, (MAP_W, MAP_H)), (0, 0))

    def draw_plan(self):
        """Draw the line of play the goblin's minimax expects (red = goblin, blue = you)."""
        b = self.g.brain
        if not b or b.get('stunned') or not b['line']:
            return
        w = self.world
        chains = {'G': [b['origin_g']], 'K': [b['origin_k']]}
        numbered = []                                # (ply number, who, tile)
        for i, (who, pos) in enumerate(b['line']):
            chains[who].append(pos)
            numbered.append((i + 1, who, pos))
        colors = {'G': C_RED, 'K': C_BLUE}
        for who, pts in chains.items():
            for a, c in zip(pts, pts[1:]):
                if a != c:
                    pygame.draw.line(w, colors[who], self.center(*a), self.center(*c), 4)
        for n, who, pos in numbered:
            cx, cy = self.center(*pos)
            pygame.draw.circle(w, (10, 10, 16), (cx, cy), 11)
            pygame.draw.circle(w, colors[who], (cx, cy), 10, 3)
            draw_text(w, self.f_tiny, str(n), C_TEXT, (cx, cy), anchor="center", shadow=False)

    def draw_world(self):
        w, g, t = self.world, self.g, self.t
        w.blit(self.map_surf, (0, 0))

        for pos, kind in g.items.items():
            if pos in g.explored or not self.fog:    # fog off: every item is visible
                draw_item(w, kind, *self.center(*pos), t)

        if self.fog:                                 # fog off: skip the darkness layer entirely
            self.draw_darkness()

        for (r, c), v in g.loot.items():             # treasure always glows through the dark
            cx, cy = self.center(r, c)
            if v >= GEM:
                draw_gem(w, cx, cy, t, r * 1.7 + c)
            else:
                draw_coin(w, cx, cy, t, r * 1.3 + c * 0.7)

        # fog off: goblin is always visible (overrides hard mode)
        goblin_shown = g.g_pos in self.vis or not g.hide_goblin or not self.fog
        if self.show_plan and goblin_shown:
            self.draw_plan()

        if goblin_shown:
            cx, cy = self.center(self.gv[0], self.gv[1])
            w.blit(GLOWS["red"], (cx - 30, cy - 30))
            draw_goblin(w, cx, cy, t, stunned=g.g_stun > 0)

        cx, cy = self.center(self.pv[0] + self.p_off[0], self.pv[1] + self.p_off[1])
        draw_knight(w, cx, cy, t)

        for p in self.particles:
            a = max(0.0, p['life'] / p['max'])
            col = tuple(int(ch * (0.4 + 0.6 * a)) for ch in p['color'])
            pygame.draw.circle(w, col, (int(p['x']), int(p['y'])), max(1, int(p['size'] * a + 0.5)))
        for tx in self.texts:
            col = tuple(int(ch * min(1.0, tx['life'] * 2)) for ch in tx['color'])
            draw_text(w, self.f_med, tx['text'], col, (int(tx['x']), int(tx['y'])), anchor="center")

    # ---- drawing: HUD / panels ----
    def draw_hud(self):
        s, g = self.screen, self.g
        pygame.draw.rect(s, C_PANEL, (0, 0, WIDTH, HUD_H))
        pygame.draw.line(s, (60, 60, 84), (0, HUD_H - 1), (WIDTH, HUD_H - 1), 2)
        draw_text(s, self.f_big, "DUNGEON HEIST", C_BLUE, (16, 8))
        draw_text(s, self.f_med, f"LEVEL {g.level_no}", C_GOLD, (16, 48))

        depth = g.ai_depth()                         # how far ahead the goblin looks
        draw_text(s, self.f_small, f"GOBLIN LOOKS AHEAD: {depth} MOVES", C_RED, (230, 8))
        for i in range(MAX_DEPTH):
            col = C_RED if i < depth else (60, 40, 50)
            pygame.draw.rect(s, col, (230 + i * 26, 28, 22, 10), border_radius=3)
        draw_text(s, self.f_small, f"TOTAL {g.total}", C_TEXT, (WIDTH - 16, 12), anchor="topright")

        chips = []
        if g.haste > 0:
            chips.append((f"HASTE {g.haste}", C_CYAN))
        if g.g_stun > 0:
            chips.append((f"GOBLIN STUNNED {g.g_stun}", C_GOLD))
        if not self.fog:
            chips.append(("FOG OFF", C_GREEN))
        if g.hide_goblin and self.fog:
            chips.append(("HARD MODE", C_RED))
        x = 440
        for label, col in chips:
            rect = draw_text(s, self.f_small, label, col, (x + 8, 36), shadow=False)
            pygame.draw.rect(s, col, (x, rect.y - 3, rect.w + 16, rect.h + 6), 2, border_radius=6)
            x += rect.w + 28

        bx, by, bw, bh = 230, 52, 380, 12            # tug-of-war bar: who is winning the loot race
        total = g.p_score + g.g_score
        frac = 0.5 if total == 0 else g.p_score / total
        pygame.draw.rect(s, C_RED, (bx, by, bw, bh), border_radius=6)
        if frac > 0:
            pygame.draw.rect(s, C_BLUE, (bx, by, int(bw * frac), bh), border_top_left_radius=6,
                             border_bottom_left_radius=6)
        pygame.draw.rect(s, (255, 255, 255), (bx + bw // 2 - 1, by - 3, 2, bh + 6))
        pygame.draw.rect(s, (90, 90, 120), (bx, by, bw, bh), 2, border_radius=6)
        draw_text(s, self.f_tiny, f"KNIGHT {g.p_score}", C_BLUE, (bx, by + 15))
        draw_text(s, self.f_tiny, f"GOBLIN {g.g_score}", C_RED, (bx + bw, by + 15), anchor="topright")
        draw_text(s, self.f_tiny, f"LOOT LEFT {len(g.loot)}", C_GOLD, (bx + bw // 2, by + 15), anchor="midtop")

    def draw_bottom(self):
        s, g = self.screen, self.g
        y0 = HUD_H + MAP_H
        pygame.draw.rect(s, C_PANEL, (0, y0, WIDTH, LOG_H))
        pygame.draw.line(s, (60, 60, 84), (0, y0), (WIDTH, y0), 2)
        for i, msg in enumerate(g.log):
            newest = i == len(g.log) - 1
            draw_text(s, self.f_small, msg, C_TEXT if newest else C_DIM, (16, y0 + 8 + i * 22))
        draw_text(s, self.f_tiny, "WASD move  Space wait  M path  T tree  E help  F fog  H hide",
                  (110, 110, 140), (16, y0 + LOG_H - 24), shadow=False)

        # ---- the Goblin Brain panel ----
        px, py, pw, ph = 560, y0 + 6, WIDTH - 560 - 10, LOG_H - 12
        pygame.draw.rect(s, (34, 24, 32), (px, py, pw, ph), border_radius=8)
        pygame.draw.rect(s, C_RED, (px, py, pw, ph), 2, border_radius=8)
        draw_text(s, self.f_small, "GOBLIN BRAIN (last decision)", C_RED, (px + 10, py + 6), shadow=False)
        b = g.brain
        rows = []
        if b is None:
            rows = [("Make a move and watch", C_DIM), ("the goblin think.", C_DIM)]
        elif b.get('stunned'):
            rows = [("STUNNED - it can't think!", C_GOLD)]
        else:
            lead = round(b['score'] / 1000)
            who = "goblin" if lead > 0 else "you" if lead < 0 else "nobody"
            verdict = "even race" if lead == 0 else f"{who} +{abs(lead)} pts"
            rows = [(f"Looked {b['depth']} moves ahead", C_TEXT),
                    (f"Positions checked: {b['nodes']:,} ({b['ms']:.0f} ms)", C_TEXT),
                    (f"Branches pruned: {b['pruned']:,}", C_GREEN),
                    (f"It expects: {verdict}", C_GOLD)]
        for i, (txt, col) in enumerate(rows):
            draw_text(s, self.f_tiny, txt, col, (px + 10, py + 30 + i * 19), shadow=False)

    def overlay(self, lines, alpha=185):
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        veil.fill((0, 0, 0, alpha))
        self.screen.blit(veil, (0, 0))
        total_h = sum(h for _, _, _, h in lines)
        y = HEIGHT // 2 - total_h // 2
        for text, font, col, h in lines:
            if text:
                draw_text(self.screen, font, text, col, (WIDTH // 2, y), anchor="midtop")
            y += h

    def big_sprite(self, fn, x, y, *args, scale=3):
        tmp = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        fn(tmp, TILE // 2, TILE // 2, self.t, *args)
        self.screen.blit(pygame.transform.scale(tmp, (TILE * scale, TILE * scale)), (x, y))

    # ---- drawing: full screens ----
    def draw_title(self):
        s = self.screen
        s.fill(C_BG)
        for i in range(40):                          # drifting embers
            ex = (i * 97 + self.t * (10 + i % 7)) % WIDTH
            ey = HEIGHT - ((i * 53 + self.t * (18 + i % 5) * 3) % HEIGHT)
            pygame.draw.circle(s, (120 + i % 100, 70, 30), (int(ex), int(ey)), 1 + i % 2)
        draw_text(s, self.f_huge, "DUNGEON HEIST", C_BLUE, (WIDTH // 2, 40), anchor="midtop")
        draw_text(s, self.f_med, "You vs. a goblin who plans with MINIMAX.", C_TEXT,
                  (WIDTH // 2, 118), anchor="midtop")
        self.big_sprite(draw_knight, 200, 170)
        self.big_sprite(draw_goblin, WIDTH - 320, 170, False)
        lines = [("Grab more loot than the goblin on every level - or the heist is over.", C_GOLD),
                 ("Bump the goblin to stun it and make it drop a coin.", C_TEXT),
                 ("Boots make it lose turns. Scrolls freeze it. Each level it thinks deeper.", C_GREEN),
                 ("Watch the coloured path: it's the future the goblin is predicting.", C_RED)]
        for i, (txt, col) in enumerate(lines):
            draw_text(s, self.f_small, txt, col, (WIDTH // 2, 330 + i * 28), anchor="midtop")
        if int(self.t * 2) % 2 == 0:
            draw_text(s, self.f_big, "Press any key to begin", C_GOLD, (WIDTH // 2, 480), anchor="midtop")
        draw_text(s, self.f_small, "Press E any time to see how minimax works", C_CYAN,
                  (WIDTH // 2, 540), anchor="midtop")
        draw_text(s, self.f_tiny, "WASD / arrows - Space wait - M plan - T search tree - F fog on/off - H goblin hides - Esc quit",
                  C_DIM, (WIDTH // 2, 580), anchor="midtop")

    def draw_node(self, shape, x, y, label, color, dim=False):
        col = (90, 90, 100) if dim else color
        if shape == "max":                           # goblin: picks the highest
            pts = [(x, y - 24), (x - 24, y + 18), (x + 24, y + 18)]
        elif shape == "min":                         # knight: picks the lowest
            pts = [(x, y + 24), (x - 24, y - 18), (x + 24, y - 18)]
        else:                                        # leaf
            pts = None
        if pts:
            pygame.draw.polygon(self.screen, (20, 20, 30), pts)
            pygame.draw.polygon(self.screen, col, pts, 3)
        else:
            pygame.draw.rect(self.screen, (20, 20, 30), (x - 20, y - 16, 40, 32), border_radius=6)
            pygame.draw.rect(self.screen, col, (x - 20, y - 16, 40, 32), 3, border_radius=6)
        draw_text(self.screen, self.f_med, label, (150, 150, 160) if dim else C_TEXT,
                  (x, y), anchor="center", shadow=False)

    def draw_explain(self):
        s = self.screen
        s.fill(C_BG)
        draw_text(s, self.f_big, "HOW THE GOBLIN THINKS: MINIMAX", C_RED, (WIDTH // 2, 18), anchor="midtop")
        steps = [
            "1. The goblin imagines every move it could make.",
            "2. For each, it imagines your best reply, then its own reply, and so on - as many moves deep as the 'looks ahead' meter says.",
            "3. Each imagined future gets a score: (goblin points - your points), plus a small bonus for being closer to the loot.",
            "4. On ITS turns it picks the HIGHEST score (max). On YOUR turns it assumes you pick the LOWEST (min) - it expects you to play your best.",
            "5. Alpha-beta pruning skips branches that can't change the answer, so it thinks fast.",
        ]
        y = 66
        for step in steps:
            for ln in wrap_text(self.f_small, step, WIDTH - 80):
                draw_text(s, self.f_small, ln, C_TEXT, (40, y), shadow=False)
                y += 22
            y += 6

        cx = WIDTH // 2                              # worked example tree
        top = y + 50
        root = (cx, top)
        mids = [(cx - 190, top + 90), (cx + 190, top + 90)]
        leaves = [(cx - 290, top + 190), (cx - 90, top + 190), (cx + 90, top + 190), (cx + 290, top + 190)]
        for m, kids in ((mids[0], leaves[0:2]), (mids[1], leaves[2:4])):
            pygame.draw.line(s, (110, 110, 130), (root[0], root[1] + 18), (m[0], m[1] - 18), 2)
            for k in kids:
                pygame.draw.line(s, (110, 110, 130), (m[0], m[1] + 18), (k[0], k[1] - 16), 2)
        self.draw_node("max", *root, "3", C_RED)
        self.draw_node("min", *mids[0], "3", C_BLUE)
        self.draw_node("min", *mids[1], "2", C_BLUE)
        for pos, val, dim in zip(leaves, ("3", "5", "2", "9"), (False, False, False, True)):
            self.draw_node("leaf", *pos, val, C_GOLD, dim)
        kx, ky = leaves[3]                           # cross out the pruned leaf
        pygame.draw.line(s, C_RED, (kx - 24, ky - 20), (kx + 24, ky + 20), 4)
        pygame.draw.line(s, C_RED, (kx - 24, ky + 20), (kx + 24, ky - 20), 4)

        draw_text(s, self.f_tiny, "GOBLIN (max): takes the best", C_RED, (root[0] + 40, root[1] - 6), shadow=False)
        draw_text(s, self.f_tiny, "YOU (min): leave it the worst", C_BLUE, (mids[0][0] - 90, mids[0][1] + 28), anchor="midtop", shadow=False)
        draw_text(s, self.f_tiny, "YOU (min)", C_BLUE, (mids[1][0], mids[1][1] + 28), anchor="midtop", shadow=False)
        draw_text(s, self.f_tiny, "PRUNED: once 2 is seen, this branch is already", C_GREEN,
                  (kx, ky + 26), anchor="midtop", shadow=False)
        draw_text(s, self.f_tiny, "worse than the 3 the goblin has - skip the 9!", C_GREEN,
                  (kx, ky + 42), anchor="midtop", shadow=False)
        draw_text(s, self.f_small, "Leaves = scores of imagined endings. Goblin chooses the left branch (3).", C_DIM,
                  (cx, top + 262), anchor="midtop", shadow=False)
        draw_text(s, self.f_med, "Press any key to go back", C_GOLD, (cx, HEIGHT - 36), anchor="midtop")

    # ---- drawing: the search tree (T key) ----
    @staticmethod
    def dir_label(frm, to):
        return {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT",
                (0, 0): "STAY"}.get((to[0] - frm[0], to[1] - frm[1]), "?")

    def draw_tbox(self, x, y, w, top, bottom, color, pruned=False, best=False):
        s = self.screen
        rect = pygame.Rect(0, 0, int(w), 40)
        rect.center = (int(x), int(y))
        if pruned:
            pygame.draw.rect(s, (28, 28, 34), rect, border_radius=6)
            pygame.draw.rect(s, (95, 95, 105), rect, 2, border_radius=6)
            draw_text(s, self.f_tiny, top, (120, 120, 130), (rect.centerx, rect.y + 4), anchor="midtop", shadow=False)
            draw_text(s, self.f_tiny, "pruned", (120, 120, 130), (rect.centerx, rect.y + 21), anchor="midtop", shadow=False)
            pygame.draw.line(s, C_RED, rect.topleft, rect.bottomright, 3)
            pygame.draw.line(s, C_RED, rect.bottomleft, rect.topright, 3)
        else:
            border = C_GOLD if best else color
            pygame.draw.rect(s, (20, 20, 30), rect, border_radius=6)
            pygame.draw.rect(s, border, rect, 4 if best else 2, border_radius=6)
            draw_text(s, self.f_tiny, top, C_TEXT, (rect.centerx, rect.y + 4), anchor="midtop", shadow=False)
            draw_text(s, self.f_tiny, bottom, border, (rect.centerx, rect.y + 21), anchor="midtop", shadow=False)

    def draw_tree(self):
        s, b = self.screen, self.g.brain
        s.fill(C_BG)
        draw_text(s, self.f_big, "THE GOBLIN'S SEARCH TREE", C_RED, (WIDTH // 2, 10), anchor="midtop")
        draw_text(s, self.f_small, "Top 2 levels of its last decision. Gray boxes with a red X were PRUNED: never examined.",
                  C_DIM, (WIDTH // 2, 50), anchor="midtop", shadow=False)

        tree = b['tree']
        slots = sum(max(1, len(n['kids'])) for n in tree)
        slot_w = min(70.0, (WIDTH - 40) / slots)
        x0 = (WIDTH - slot_w * slots) / 2
        y_root, y1, y2 = 120, 250, 390
        rx = WIDTH // 2

        l1, l2, k = [], [], 0                        # layout: one slot per bottom-row box
        for n in tree:
            cnt = max(1, len(n['kids']))
            x = x0 + slot_w * (k + cnt / 2)
            l1.append((n, x))
            for j, kid in enumerate(n['kids']):
                l2.append((n, kid, x, x0 + slot_w * (k + j + 0.5)))
            k += cnt

        for n, x in l1:                              # lines first, boxes on top
            col, wd = (C_GOLD, 4) if n['best'] else ((110, 110, 130), 2)
            pygame.draw.line(s, col, (rx, y_root + 20), (x, y1 - 20), wd)
        for n, kid, x, kx in l2:
            if kid['pruned']:
                col, wd = (70, 70, 80), 1
            elif kid['best'] and n['best']:
                col, wd = C_GOLD, 4
            else:
                col, wd = (110, 110, 130), 2
            pygame.draw.line(s, col, (x, y1 + 20), (kx, y2 - 20), wd)

        self.draw_tbox(rx, y_root, 96, "GOBLIN", "picks MAX", C_RED)
        for n, x in l1:
            label = self.dir_label(b['origin_g'], n['pos']) + (" $" if n['pos'] in b['loot'] else "")
            self.draw_tbox(x, y1, 64, label, f"{n['score'] / 1000:+.1f}", C_RED, best=n['best'])
            if n['cut']:
                skipped = sum(1 for kid in n['kids'] if kid['pruned'])
                draw_text(s, self.f_tiny, f"skipped {skipped}", C_GREEN, (x, y1 + 24), anchor="midtop", shadow=False)
        for n, kid, x, kx in l2:
            label = self.dir_label(b['origin_k'], kid['pos'])
            score = "" if kid['pruned'] else f"{kid['score'] / 1000:+.1f}"
            self.draw_tbox(kx, y2, max(40, slot_w - 6), label, score, C_BLUE,
                           pruned=kid['pruned'], best=kid['best'] and n['best'])

        notes = [
            "ROW 1 = each move the goblin could make (red).  ROW 2 = your best replies to that move (blue).  $ = grabs treasure.",
            "Gold = the line the goblin chose.  Numbers = point lead if both sides play that line (+ means the goblin is ahead).",
            "Gray box with red X = PRUNED: alpha-beta skipped it.  Green 'skipped N' = the cutoff happened under that goblin move.",
            "Why skip? Once ONE of your replies already makes that goblin move worse than a move it has found, "
            "your other replies can't change its mind, so it stops looking.",
        ]
        y = 448
        for note in notes:
            for ln in wrap_text(self.f_tiny, note, WIDTH - 60):
                draw_text(s, self.f_tiny, ln, C_TEXT, (30, y), shadow=False)
                y += 18
            y += 4
        draw_text(s, self.f_small,
                  f"Whole search: {b['nodes']:,} positions checked, {b['pruned']:,} branches skipped "
                  f"(deeper levels get pruned too, not drawn here).", C_GREEN, (WIDTH // 2, y + 8),
                  anchor="midtop", shadow=False)
        draw_text(s, self.f_med, "Press any key to go back", C_GOLD, (WIDTH // 2, HEIGHT - 36), anchor="midtop")

    def draw(self):
        if self.state == "title":
            self.draw_title()
            pygame.display.flip()
            return
        if self.state == "tree":
            self.draw_tree()
            pygame.display.flip()
            return
        if self.state == "explain":
            self.draw_explain()
            pygame.display.flip()
            return

        self.screen.fill(C_BG)
        self.draw_world()
        ox = oy = 0
        if self.shake > 0:
            ox = int(random.uniform(-self.shake, self.shake))
            oy = int(random.uniform(-self.shake, self.shake))
        self.screen.blit(self.world, (ox, HUD_H + oy))
        self.draw_hud()
        self.draw_bottom()

        if self.fade > 0:
            veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            veil.fill((0, 0, 0, int(255 * self.fade)))
            self.screen.blit(veil, (0, 0))

        g = self.g
        if self.state == "clear":
            self.overlay([
                (f"LEVEL {g.level_no} CLEARED", self.f_huge, C_GREEN, 80),
                (f"Knight {g.p_score}   vs   Goblin {g.g_score}", self.f_big, C_GOLD, 50),
                (f"Total score: {g.total}", self.f_med, C_TEXT, 50),
                (f"Next level: the goblin looks {min(3 + g.level_no + 1, MAX_DEPTH)} moves ahead...  (press any key)",
                 self.f_med, C_DIM, 40)])
        elif self.state == "over":
            self.overlay([
                ("GAME OVER", self.f_huge, C_RED, 80),
                (self.over_reason, self.f_big, C_TEXT, 50),
                (f"Knight {g.p_score}  vs  Goblin {g.g_score}   (reached level {g.level_no})", self.f_med, C_DIM, 36),
                (f"Final score: {self.over_final}     Best: {self.best}", self.f_big, C_GOLD, 56),
                ("Play again?  Y / N", self.f_big, C_BLUE, 40)], alpha=205)
        pygame.display.flip()

    # ---- main loop ----
    def run(self):
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.on_key(event.key)
            if self.state in ("title", "explain", "tree"):
                self.t += dt
            else:
                self.update(dt)
            self.draw()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    App().run()