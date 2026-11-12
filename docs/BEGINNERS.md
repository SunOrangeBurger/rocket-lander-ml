# Teaching a rocket to land: a beginner walkthrough

## A two-minute explanation you can say aloud

“Imagine teaching a pilot inside a video game. The game knows where the rocket is, how quickly it is falling, how much it is leaning, where the landing pad is, and how the wind is blowing. We turn those measurements into a list of numbers.

“Our model multiplies those numbers by adjustable values called weights. Adding the products gives two commands: how hard to fire the engine and how far to deflect the fins. The simulator moves the rocket for one sixtieth of a second, then the model makes another decision.

“Initially the weights are poor. During training, we try many different sets of weights. Each pilot flies the same practice challenges. A successful landing earns a high score; missing the pad, falling too fast, or tipping over loses points. We use the best pilots to guide our next guesses. Repeating this process produces a better set of weights.

“Finally we save the best weights and test them on different starting conditions. When we play the game, the saved pilot makes decisions; it is no longer changing its weights.”

## 1. Four parts of the project

| File | Role | Read first |
| --- | --- | --- |
| `src/main.rs` | The visible game: keyboard input, drawing, and switching to the ML pilot | `Game::step` |
| `src/physics.rs` | The world: gravity, thrust, wind, fuel, motion, and landing rules | `advance`, then `landing_error` |
| `src/learning.rs` | The brain, saved weights, and one complete practice flight | `Policy::act`, then `episode` |
| `src/training.rs` | The teacher: generates candidates, scores them, and improves guesses | Numbered STEP comments in `Trainer::advance_candidate` |
| `src/bin/train.rs` | Runs training quickly and reports held-out results | `main`, then `report` |
| `src/bin/training-demo.rs` | Shows the learning process with replays and graphs | `Demo::update`, then `draw_stats` |

The code contains a `BEGINNER WALKTHROUGH` at the top of each of these files. The trainer runs without drawing a window, which makes thousands of practice flights practical.

## 2. What does the model see?

An **observation** is the information available for one decision. Our observation has 12 entries, in exactly the order shown in `Policy::act`:

| Index | Input | Why it helps |
| --- | --- | --- |
| 0 | Always 1, called the bias input | Lets the model learn a constant starting offset |
| 1 | Altitude / 2100 | How far the rocket is from the ground |
| 2 | Vertical speed / 200 | How fast it is falling; positive means downward |
| 3 | Estimated stopping distance / altitude, capped at 5 | Whether there is enough room left to brake |
| 4 | Horizontal distance to pad / 600 | Which direction to travel |
| 5 | Sideways speed relative to pad / 100 | Whether to accelerate sideways or brake sideways |
| 6 | Lean angle, in radians | Whether the rocket is upright |
| 7 | Spin, in radians per second | Whether it is still rotating |
| 8 | Current wind / 100 | How much the surroundings push it sideways |
| 9 | Fuel / starting fuel | How much engine use remains |
| 10 | Square root of normalized altitude | Gives a simple layer another way to respond to height |
| 11 | Pad velocity / 100 | Useful when the landing pad moves |

Dividing by a typical size is **normalization**. At 1050 m, the normalized altitude is 1050 / 2100 = 0.5. Otherwise height might be in the thousands while angle is near 0.1, making it harder to search for useful weights.

The stopping-distance estimate uses `speed² / (2 × available braking acceleration)`. It is an approximation: real braking here also depends on tilt and drag. We supply it as a helpful input; the learned weights still decide what action to take. This is **feature engineering**: humans choose useful measurements and the model learns how to combine them.

## 3. How do numbers become actions?

A **policy** is a rule for choosing actions from observations. Ours is a small single neural layer:

```text
throttle = clamp(sum(input[i] * throttle_weight[i]), 0, 1)
fins     = clamp(sum(input[i] * fin_weight[i]), -1, 1)
```

There are 12 weights for each of 2 outputs: **24 learned weights** total. No hidden layers are present. `clamp` limits a number to the allowed range: throttle 0 means engine off, 1 means full burn; fin 0 means centered.

For a deliberately simplified example, suppose only three throttle weights were nonzero:

```text
bias input = 1.0, weight = -0.2       contribution = -0.2
fall-speed input = 0.8, weight = 0.5 contribution =  0.4
braking-ratio input = 0.6, weight = 1 contribution = 0.6
sum = -0.2 + 0.4 + 0.6 = 0.8 -> 80% throttle
```

These are teaching numbers, not the saved model's actual weights. In the real code, all 12 contributions are added separately for throttle and fins.

## 4. What happens after a decision?

An **action** sets the engine and fins. `advance` then:

1. Limits the controls and burns fuel; an empty tank disables thrust.
2. Adds gravity, air drag, wind force, engine thrust, and fin force.
3. Updates velocity from acceleration, then position from velocity.
4. Updates spin and lean based on fin settings and available control authority.

Both training and the game call this same function at **60 steps per simulated second**. Screen coordinates increase downward, so a positive vertical speed is falling. Rendering only makes those calculations visible; it does not teach the model.

When a foot reaches the ground, `landing_error` checks pad position, vertical speed, sideways speed relative to the pad, lean, and spin. Every check must pass. A model cannot earn a successful landing simply by exploiting a generous training-only rule.

## 5. How does learning happen?

A complete flight is an **episode**. At the end, `episode` calculates a **reward**, the score used by the trainer. This differs from the game score displayed to the player.

Success earns 15 points plus a small remaining-fuel bonus. Distance from the pad, sideways motion, impact speed, lean, and spin subtract points. Excessive impact speed and missing the pad add extra penalties. Leaving the world or taking more than 60 simulated seconds also incurs a penalty.

Those smaller penalties are **reward shaping**. They give the search direction before it discovers a successful landing. An almost-correct attempt should usually score better than one that flies far away.

The training algorithm is the **cross-entropy method**, a form of evolutionary policy search:

1. Keep a center value and an exploration spread for each weight.
2. Make a generation of 80 candidate pilots, including the current champion.
3. Fly every candidate on the same practice cases and average its rewards.
4. Keep the best 12 candidates, called elites.
5. Move the center toward their average weights and adjust the spread.
6. Repeat, saving the best pilot found.

Candidates are weight arrays, not individual actions. The algorithm adjusts weights from whole-flight scores. It does not use backpropagation, labeled correct actions, or a scripted expert pilot.

## 6. Why did level 3 fail before?

The first pilot practiced only calm level-one flights. Level 3 has steady wind, changing gusts, and a narrower pad. Seeing a wind input is insufficient if training never teaches the model how to use it.

The trainer now practices **levels 1 through the requested maximum**, with 24 different starting cases per level. For maximum level 3, that is 72 cases per candidate, including both wind directions. Mixing earlier levels prevents improving at gusts while forgetting calm landings.

The new checkpoint was continued from the earlier model for 240 generations. Landing rules, wind strength, and available fuel were unchanged. The earlier checkpoint remains in `models/level1-original.txt` for comparison.

## 7. How do we know it learned useful behavior?

Training cases are the practice questions. **Held-out cases** are exam questions: different random seeds that never choose weights or update the search distribution. A seed makes a particular starting setup repeatable.

The saved pilot landed 1000/1000 evaluator flights on each of levels 1, 2, and 3. It also landed 1000/1000 on each level in a separate audit. The evaluator reports 1000 held-out flights per level. Automated regression tests use the separate 1000-case audit set per level, requiring at least 99% success on levels 1–3. These are empirical checks, not a guarantee of success for every possible start or harder level. The simulator also gives exact state and wind measurements; sensor noise is not modeled.

Learning and evaluation are different activities. `--evaluate` runs the saved model without changing it. Pressing Enter to retry a game flight also leaves its weights unchanged.

## 8. Classroom demo

Run from the project directory. Close and restart an already-running game to use changed code; press L to load changed weights into a game that supports reloading.

```powershell
# Jump straight into gusty level 3 with the saved ML pilot flying.
cargo run --offline --release -- --level 3 --ml

# Evaluate all three levels without changing the model.
cargo run --offline --release --bin train -- 1 3 --evaluate

# Continue learning on levels 1, 2, and 3.
cargo run --offline --release --bin train -- 240 3
```

Before demonstrating training, copy `landing-model.txt` somewhere safe: training replaces it. Add `--fresh` to start from zero weights instead of the checkpoint. Starting from zero can take longer and may produce different performance. The fixed random seed makes the search repeatable given the same initial weights and arguments; it is still a search, not a guaranteed solution.

Suggested questions for learners:

- What happens if the pilot ignores sideways speed?
- Why is a successful landing bonus alone hard to learn from?
- Why test new starting conditions instead of only practice flights?
- Why practice calm flights while learning to fly through gusts?
- Which numbers are fixed rules, and which numbers are learned weights?

## 9. Show learning live, from the first failure

```powershell
cargo run --offline --release --bin training-demo
```

This is an actual training session. It always starts with zero weights and opens paused at generation 0, so you can introduce the lesson before clicking **Resume**. It never loads the already-trained game pilot.

A useful presentation sequence:

1. **Before training:** point out 0 successful practice landings. All weights are zero, so the engine stays off and fins stay centered. Resume and let the rocket fail.
2. **First guesses:** watch the candidate counter. The trainer tries 79 new weight arrays alongside the retained champion; each is compared on the same 72 cases across levels 1–3.
3. **First improvement:** the reward graph may rise before any landings succeed. Explain that getting closer to the pad or hitting less hard can improve the reward.
4. **First success:** distinguish the one replay from the champion's score across all 72 practice cases. One successful replay alone does not establish reliability.
5. **Continued exploration:** compare the mint champion line with the yellow line for all candidates. Many new guesses can fail even while the best pilot succeeds.
6. **Compare:** pause, then click Previous to replay earlier weights on the same starting setup. Click Next to move forward. An older snapshot replays once and waits; it does not keep advancing training automatically.
7. **Export:** save the selected weights and the graph data for discussion. A fresh attempt can be started with Start over.

Controls are clickable, with keyboard alternatives: Space pauses/resumes; minus/plus changes replay speed; Left/Right reviews saved generations; N moves forward or starts the next generation; R resets the demo; S exports. The default is 100 generations. To shorten the lesson, pass maximum level and generation limit: `cargo run --offline --release --bin training-demo -- 3 30`.

The left panel always demonstrates fixed start #7 at the selected maximum level. The right panel summarizes the 24 practice starts per level. The displayed graphs are **training metrics**, not test accuracy. The champion is retained between generations, so its reward cannot decrease on this fixed practice set, although its landing count need not increase every generation. The all-candidates line includes the retained champion and the exploratory guesses.

Training happens between replays, using the same `Trainer` as the CLI. Replaying the same weights does not itself teach anything. Slowing playback helps the lesson; it does not change the physics or search algorithm. The fixed seed and zero initial weights make this demonstration reproducible.

Export creates `demo-model.txt` for the selected snapshot and `training-demo-history.csv` for all completed generations, in the project directory. The latter has generation, champion reward, successful practice landings, practice case count, and candidate landing rate (a fraction from 0 to 1). It can be plotted later. Demo exports do not overwrite the proven `landing-model.txt` used by the game.
