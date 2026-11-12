# Rocket Re-Lander: learned landing pilot

A Rust/Macroquad lander with a small learned controller. The visible game and headless trainer share 60 Hz physics, wind, fuel constraints, and landing checks.

## Watch the model learn from failure

```powershell
cargo run --offline --release --bin training-demo
```

The window opens at **generation 0 with zero weights**, paused for your introduction. Click **Resume** to watch the first crash, followed by real training and improving pilots. This demo uses the same simulation and training algorithm as the command-line trainer.

- The left panel replays the champion after each generation on the same level-three start, showing its trajectory, controls, and crash reason or successful touchdown.
- The right panel plots champion landing rate, exploratory candidate landing rate, and champion reward across generations. These are practice scores, not held-out evaluation scores.
- Click **Pause**, **Slower/Faster**, **Previous**, or **Next** to explain and compare. **Start over** resets to zero weights. Keyboard shortcuts appear on the buttons.
- **Export** writes the selected generation to `demo-model.txt` and all graph points to `training-demo-history.csv` in the project directory. The demo never replaces `landing-model.txt`.

Defaults: practice levels 1–3, 100 generations, 4x replay speed. To choose maximum level and generation count:

```powershell
cargo run --offline --release --bin training-demo -- 3 30
```

A generation compares 80 candidates on 24 starts per level, retains the best 12, and updates its search. The champion is kept, but new exploratory pilots can still fail. Learning is real; the demo deliberately waits for each champion replay so the audience can follow. Replays use fixed start #7, not an average flight. Previous generations replay once and wait; Next moves forward again. Speed controls change flight playback, not the physical time step or the learning rules.

## Try level 3 immediately

Run from this project directory:

```powershell
cargo run --offline --release -- --level 3 --ml
```

This starts a gusty level-three flight with the saved ML pilot enabled. For the normal progression from level 1, use `cargo run --offline --release`, press **M**, then **Enter**. **M** switches manual/ML control; **L** reloads `landing-model.txt`. Enter retries a crash or advances after a landing. Close and restart an old game to pick up code changes; L is sufficient for updated weights alone.

## Current model results

The first checkpoint practiced only calm level-one landings. The updated trainer practices levels **1 through the requested maximum**, keeping earlier skills while learning wind and gusts. The current checkpoint continued from the original weights for **240 generations**, using **24 starts per level**, or 72 practice cases per candidate. Landing rules and physics were unchanged.

| Level | Environment | Held-out evaluation | Separate regression audit |
| --- | --- | --- | --- |
| 1 | Calm | 1000/1000 | 1000/1000 |
| 2 | Steady wind | 1000/1000 | 1000/1000 |
| 3 | Wind, gusts, narrower pad | 1000/1000 | 1000/1000 |

Evaluation seeds are 10001–11000; audit seeds are 20001–21000. Neither set participates in weight selection. These are measured simulation results, not a guarantee for all possible conditions. Level 4 introduces a moving pad and is outside the current checkpoint's training coverage.

The original calm-only checkpoint is preserved in `models/level1-original.txt`.

## Beginner explanation and commented code

Start with **[the teaching guide](docs/BEGINNERS.md)**. It includes a two-minute explanation to read aloud, a table of the 12 inputs, a worked calculation, vocabulary, training steps, and classroom demo commands.

The source files also contain `BEGINNER WALKTHROUGH` sections:

- `src/learning.rs`: observations, normalization, weights, actions, episodes, and rewards.
- `src/training.rs`: numbered steps explaining the shared candidate search and selection.
- `src/bin/train.rs`: command-line training and held-out evaluation.
- `src/bin/training-demo.rs`: animated replays, graphs, controls, and safe export.
- `src/physics.rs`: forces, coordinates, simulation steps, and landing checks.
- `src/main.rs`: where the learned pilot takes control of the visible game.

## Train or evaluate

```powershell
# Continue training on levels 1, 2, and 3: generations, maximum level.
cargo run --offline --release --bin train -- 240 3

# Evaluate each level from 1 through 3 without changing the model.
cargo run --offline --release --bin train -- 1 3 --evaluate

# Start a new search from zero weights instead of resuming the saved pilot.
cargo run --offline --release --bin train -- 300 3 --fresh

# Extend training to include moving pads (not yet validated).
cargo run --offline --release --bin train -- 300 4
```

Training replaces `landing-model.txt` with the best policy for the selected mixture. Back up the file first to keep the current pilot. Resume loads weights but starts a fresh optimizer distribution. The default maximum level is 3. No Python, GPU, or ML service is needed.

## How it learns

The policy is one neural layer with 12 normalized/engineered inputs and two clipped linear outputs: throttle and fin deflection. All 24 weights are learned. There is no scripted landing controller or demonstration pilot.

The cross-entropy method evaluates 80 candidate pilots per generation, selects the best 12, and moves its parameter sampling distribution toward them. Every candidate faces identical practice cases for fair comparison. Reward favors successful, aligned, slow, stable landings with remaining fuel. Episodes stop after contact, leaving the world, or 60 simulated seconds. The final evaluation is reported separately and never chooses weights. Playing and retrying flights does not train the pilot.

## Checks

```powershell
cargo test --offline -- --nocapture
cargo clippy --offline --all-targets -- -D warnings
```

Tests cover learning from zero over 12 generations, equivalence of incremental and batch training, deterministic flights, fuel exhaustion, and at least 990/1000 audit landings on each of levels 1–3. Replacing the saved model with a weaker or differently specialized checkpoint can fail the model regression test.

The lockfile uses Macroquad 0.4.15 and its locally cached dependencies for offline builds in this workspace.
