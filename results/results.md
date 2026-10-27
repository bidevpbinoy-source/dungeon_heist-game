# Dungeon Heist - experiment results

All numbers come from `run_experiments.py`, which runs the game's real minimax code with fixed random seeds.
Simulated matches use simple bot opponents, not human players.

## Exp 1 - Alpha-beta pruning vs plain minimax (mean of 40 random positions)

| Depth | Plain nodes | Alpha-beta nodes | Nodes saved | Plain ms | Alpha-beta ms | Speed-up | Same score |
|---|---|---|---|---|---|---|---|
| 2 | 13 | 11 | 19% | 0.1 | 0.1 | 1.0x | yes |
| 3 | 43 | 28 | 34% | 0.3 | 0.2 | 1.3x | yes |
| 4 | 138 | 70 | 49% | 1.0 | 0.5 | 1.9x | yes |
| 5 | 441 | 172 | 61% | 3.2 | 1.2 | 2.6x | yes |
| 6 | 1431 | 413 | 71% | 10.1 | 2.9 | 3.6x | yes |
| 7 | 4717 | 1013 | 79% | 34.0 | 7.0 | 4.8x | yes |

## Exp 2 - Goblin strength vs depth (60 maps per cell, level-3 maps, knight moves first)

| Opponent | Depth | Goblin wins | Ties | Knight wins | Goblin win rate (95% CI) | Goblin share of loot |
|---|---|---|---|---|---|---|
| random | 1 | 59 | 0 | 1 | 98% (91-100) | 92% |
| random | 2 | 59 | 0 | 1 | 98% (91-100) | 92% |
| random | 3 | 59 | 0 | 1 | 98% (91-100) | 92% |
| random | 4 | 59 | 0 | 1 | 98% (91-100) | 92% |
| random | 5 | 59 | 0 | 1 | 98% (91-100) | 92% |
| random | 6 | 59 | 0 | 1 | 98% (91-100) | 91% |
| random | 7 | 59 | 0 | 1 | 98% (91-100) | 91% |
| greedy | 1 | 26 | 0 | 34 | 43% (32-56) | 49% |
| greedy | 2 | 25 | 0 | 35 | 42% (30-54) | 48% |
| greedy | 3 | 28 | 0 | 32 | 47% (35-59) | 49% |
| greedy | 4 | 28 | 0 | 32 | 47% (35-59) | 49% |
| greedy | 5 | 27 | 0 | 33 | 45% (33-58) | 49% |
| greedy | 6 | 27 | 0 | 33 | 45% (33-58) | 49% |
| greedy | 7 | 25 | 0 | 35 | 42% (30-54) | 48% |
| minimax-3 | 1 | 24 | 0 | 36 | 40% (29-53) | 46% |
| minimax-3 | 2 | 24 | 0 | 36 | 40% (29-53) | 47% |
| minimax-3 | 3 | 24 | 0 | 36 | 40% (29-53) | 47% |
| minimax-3 | 4 | 25 | 0 | 35 | 42% (30-54) | 47% |
| minimax-3 | 5 | 24 | 0 | 36 | 40% (29-53) | 47% |
| minimax-3 | 6 | 24 | 0 | 36 | 40% (29-53) | 47% |
| minimax-3 | 7 | 21 | 0 | 39 | 35% (24-48) | 46% |

## Exp 3 - Procrastination bug (goblin vs idle knight, 200-turn cap)

| Evaluation function | Runs that never finished | Mean turns to clear (finished runs) |
|---|---|---|
| Original | 58/80 | 56 |
| + closeness tie-break | 55/80 | 56 |
| + sooner-is-better (final) | 1/80 | 61 |
