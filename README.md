# Dungeon Heist ⚔️💰 (Minimax AI in Pygame)

A turn-based tactical zero-sum game built with **Python** and **Pygame** to demonstrate the **Minimax Algorithm** with lookahead search and heuristic evaluation.

---

## 🎮 How to Play

### 1. Launch the Game
Run the game script from your terminal:
```bash
python dungeon_heist.py
```

### 2. Controls
- **Movement:** `W`, `A`, `S`, `D` or **Arrow Keys** (Up, Left, Down, Right)
- **Restart Game:** `R`
- **Quit Game:** `ESC` or close window

---

## 📜 Game Rules & Board Layout
- **Grid:** 4x4 dungeon grid.
- **Knight (Player - Blue `P`):** Starts at top-left `(0, 0)`.
- **Goblin (Minimax AI - Red `G`):** Starts at bottom-right `(3, 3)`.
- **Gold Coins (Yellow `$`):** 4 coins placed at `(0, 3)`, `(3, 0)`, `(1, 2)`, `(2, 1)`. Each worth **10 points**.
- **Stone Walls (Gray `#`):** Impassable obstacles placed at `(1, 1)` and `(2, 2)`.
- **Turn-based Movement:** Each entity moves 1 orthogonal step per turn. You cannot walk through walls, step out of bounds, or occupy the opponent's tile.
- **Victory Condition:** Game terminates once all 4 coins are picked up. The entity with the highest gold score wins!

---

## 🧠 How Minimax Works in "Dungeon Heist"

1. **Zero-Sum Nature:** Every coin the Goblin grabs (+10) is a coin the Player cannot get.
2. **Minimax Tree (Depth 4 Plies):**
   - **Goblin (Maximizer):** Evaluates all available legal moves looking to maximize its relative score advantage.
   - **Player (Minimizer):** Assumes the human player will respond with the move that minimizes the Goblin's score advantage.
3. **Terminal Nodes:** Evaluated at depth 0 or when `len(coins) == 0`.
4. **Heuristic Evaluation:**
   $$\text{Base Score} = (\text{Goblin Score} - \text{Player Score}) \times 100$$
   $$\text{Tie-Breaker} = - \min_{\text{coin} \in \text{coins}}(\text{Manhattan Distance}(\text{Goblin}, \text{coin}))$$
   This encourages the AI to pursue the closest coin even when scores are tied across a 4-ply lookahead horizon.

---

## 📂 Architecture Overview
All logic is self-contained in [dungeon_heist.py](file:///c:/ai%20dungeon%20game/dungeon_heist.py):
1. **Game Setup & Constants:** Grid parameters, colors, walls, starting coordinates.
2. **Core Mechanics:** `get_valid_moves()`, bounds checking, collision detection, Manhattan distance.
3. **AI Core:** `evaluate_state()` and `minimax()` with alternating maximizing/minimizing layers.
4. **State Machine (`DungeonHeistGame`):** Manages turns (`PLAYER_TURN`, `AI_THINKING`, `GAME_OVER`), scores, and coin states.
5. **UI Rendering (`DungeonRenderer`):** Custom dungeon tiles, animated status banners, entity badges, and game over overlay.
6. **Main Loop (`main`):** Event handling, responsive AI delay ticker, and 60 FPS rendering.
