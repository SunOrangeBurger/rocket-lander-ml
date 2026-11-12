//! BEGINNER WALKTHROUGH: the pilot's brain and practice flights.
//!
//! 1. `Policy::act` reads the current rocket state (the observation).
//! 2. It multiplies 12 input numbers by learned weights and adds the results.
//! 3. Two outputs become engine throttle and fin deflection (the action).
//! 4. `physics::advance` moves the rocket for 1/60 second. Repeat until contact.
//! 5. `episode` grades the whole flight; `bin/train.rs` tries better weights.
//!
//! Training changes weights. Playing a saved model only USES those weights;
//! watching a flight or pressing retry does not teach it anything by itself.
//! This is a single-layer policy with engineered inputs, not a language model.
//! Full teaching notes and a worked example: docs/BEGINNERS.md.
use crate::physics::*;
use std::{fs, io};
// Each output has its own set of 12 weights: 2 * 12 = 24 learned numbers.
pub const INPUTS: usize = 12;
pub const WEIGHTS: usize = INPUTS * 2;
#[derive(Clone, Debug)]
pub struct Policy {
    pub weights: [f32; WEIGHTS],
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            weights: [0.0; WEIGHTS],
        }
    }
}
impl Policy {
    /// One decision, called 60 times per simulated second.
    /// No keyboard input, training, or random action selection happens here.
    pub fn act(&self, r: &mut Rocket, level: &Level, t: f32) {
        let alt = (GROUND_Y - r.pos.y - FOOT_Y).max(1.0);
        // OBSERVE: scale inputs to comparable sizes so learning is easier.
        // For example, 1000 m / 2100 is about 0.48 instead of a huge input.
        // The order is part of the saved-model format: do not rearrange it!
        let inputs = [
            1.0,             // 0: bias, a learned constant offset
            alt / 2100.0,    // 1: height above ground
            r.vel.y / 200.0, // 2: fall speed (positive = down)
            // 3: stopping distance / remaining height. Near 1 means little
            // braking room remains. This estimate is a feature, not an autopilot.
            (r.vel.y.max(0.0).powi(2) / (2.0 * (MAX_THRUST - GRAVITY) * alt)).min(5.0),
            (pad_x(level, t) - r.pos.x) / 600.0, // 4: pad to left (-) or right (+)
            (r.vel.x - pad_vx(level, t)) / 100.0, // 5: sideways speed relative to pad
            r.angle,                             // 6: current lean, in radians
            r.spin,                              // 7: how fast the lean changes
            wind_at(level, t, alt) / 100.0,      // 8: current wind (also changes with time)
            r.fuel / FUEL_START,                 // 9: remaining fuel fraction
            (alt / 2100.0).sqrt(),               // 10: another height feature for curved behavior
            pad_vx(level, t) / 100.0,            // 11: moving pad's sideways speed
        ];
        // DECIDE: a dot product = input0*weight0 + input1*weight1 + ...
        // Row 0 uses weights 0..11 for throttle; row 1 uses 12..23 for fins.
        // No hidden layers and no backpropagation are needed for this small model.
        let output = |row: usize| -> f32 {
            inputs
                .iter()
                .zip(&self.weights[row * INPUTS..(row + 1) * INPUTS])
                .map(|(x, w)| x * w)
                .sum()
        };
        // Clipped linear activation keeps the full engine-off/full-burn range available.
        r.throttle = output(0).clamp(0.0, 1.0);
        r.fin = output(1).clamp(-1.0, 1.0);
    }
    /// A checkpoint stores learned numbers, not a recording of a successful flight.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let body = self
            .weights
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        fs::write(path, format!("lander-policy-v1\n{body}\n"))
    }
    pub fn load(path: &str) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut words = text.split_whitespace();
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "Invalid landing policy");
        if words.next() != Some("lander-policy-v1") {
            return Err(invalid());
        }
        let mut policy = Self::default();
        for w in &mut policy.weights {
            *w = words
                .next()
                .ok_or_else(invalid)?
                .parse()
                .map_err(|_| invalid())?;
            if !w.is_finite() {
                return Err(invalid());
            }
        }
        if words.next().is_some() {
            return Err(invalid());
        }
        Ok(policy)
    }
}
/// Local RNG isolates training cases from rendering and particle randomness.
pub struct Rng(pub u64);
impl Rng {
    pub fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / 16777216.0
    }
    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.uniform()
    }
    pub fn normal(&mut self) -> f32 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f32::consts::TAU * self.uniform()).cos()
    }
}
/// Result of a whole flight. The visible demo shows the same failure reason.
#[derive(Clone, Debug)]
pub struct Episode {
    pub reward: f32,
    pub landed: bool,
    pub reason: String,
}
/// Repeatable starting conditions shared by headless training and its live demo.
pub fn seeded_start(seed: u64, difficulty: u32) -> (Level, Rocket) {
    let mut rng = Rng(seed.wrapping_mul(0x9e3779b97f4a7c15).max(1));
    // Level 1: calm. Level 2: steady wind. Level 3: wind + gusts + narrower pad.
    // Starting offsets, velocity, lean, and wind direction vary between seeds.
    let k = difficulty.saturating_sub(1) as f32;
    let level = Level {
        wind_base: if difficulty == 1 {
            0.0
        } else {
            rng.range(-1.0, 1.0).signum() * (20.0 + 12.0 * (k - 1.0)).min(110.0)
        },
        gust: if difficulty >= 3 {
            (10.0 + 6.0 * (k - 2.0)).min(50.0)
        } else {
            0.0
        },
        pad_w: (200.0 - 18.0 * k).max(80.0),
        pad_home: rng.range(1000.0, 1600.0),
        pad_range: if difficulty >= 4 {
            (120.0 + 20.0 * (k - 3.0)).min(300.0)
        } else {
            0.0
        },
        pad_freq: 0.3,
    };
    let r = Rocket {
        pos: macroquad::prelude::vec2(
            (level.pad_home + rng.range(-600.0, 600.0)).clamp(300.0, WORLD_W - 300.0),
            100.0,
        ),
        vel: macroquad::prelude::vec2(rng.range(-20.0, 20.0), 60.0),
        angle: rng.range(-0.15, 0.15),
        spin: 0.0,
        throttle: 0.0,
        fin: 0.0,
        fuel: FUEL_START,
    };
    (level, r)
}

/// One flight that can be advanced a step at a time for animation.
/// Headless evaluation calls these same steps without waiting for the screen.
pub struct Flight {
    pub level: Level,
    pub rocket: Rocket,
    pub t: f32,
    pub result: Option<Episode>,
    steps: usize,
}
impl Flight {
    pub fn new(seed: u64, difficulty: u32) -> Self {
        let (level, rocket) = seeded_start(seed, difficulty);
        Self {
            level,
            rocket,
            t: 0.0,
            result: None,
            steps: 0,
        }
    }
    pub fn step(&mut self, policy: &Policy) {
        if self.result.is_some() {
            return;
        }
        self.t += DT;
        self.steps += 1;
        policy.act(&mut self.rocket, &self.level, self.t);
        advance(&mut self.rocket, &self.level, self.t, DT);
        let touched = contact(&self.rocket) >= 0.0;
        let outside =
            self.rocket.pos.x < 0.0 || self.rocket.pos.x > WORLD_W || self.rocket.pos.y < -700.0;
        if touched || outside || self.steps >= 3600 {
            self.result = Some(grade(&self.rocket, &self.level, self.t, touched, outside));
        }
    }
}

fn grade(r: &Rocket, level: &Level, t: f32, touched: bool, outside: bool) -> Episode {
    let error = landing_error(r, level, t);
    let landed = touched && error.is_none();
    let dx = (r.pos.x - pad_x(level, t)).abs();
    let vx = (r.vel.x - pad_vx(level, t)).abs();
    // GRADE: successful landing earns a large bonus. Smaller penalties tell
    // unsuccessful pilots which attempts were closer to success. A miss by 10 m
    // should score better than a miss by 1000 m. This is called reward shaping.
    // Reward is used during training only; it does not change landing rules.
    let reward = -dx / 100.0
        - vx / 30.0
        - r.vel.y.abs() / 80.0
        - r.angle.abs() * 3.0
        - r.spin.abs()
        - (r.vel.y - MAX_VY).max(0.0) / 20.0
        - (dx - level.pad_w / 2.0).max(0.0) / 50.0
        + if landed {
            15.0 + r.fuel / FUEL_START
        } else {
            0.0
        }
        - if touched {
            0.0
        } else {
            30.0 + (GROUND_Y - r.pos.y).abs() / 100.0
        };
    let reason = if touched {
        error.unwrap_or_else(|| "Safe touchdown!".into())
    } else if outside {
        "Flew outside the world".into()
    } else {
        "Timed out after 60 simulated seconds".into()
    };
    Episode {
        reward,
        landed,
        reason,
    }
}

/// An episode is a complete practice flight. Drawing is unnecessary for learning.
pub fn episode(policy: &Policy, seed: u64, difficulty: u32) -> Episode {
    let mut flight = Flight::new(seed, difficulty);
    while flight.result.is_none() {
        flight.step(policy);
    }
    flight.result.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fuel_exhaustion_cuts_engine() {
        let level = Level {
            wind_base: 0.0,
            gust: 0.0,
            pad_w: 200.0,
            pad_home: 1300.0,
            pad_range: 0.0,
            pad_freq: 0.3,
        };
        let mut r = Rocket {
            pos: macroquad::prelude::vec2(1300.0, 100.0),
            vel: macroquad::prelude::Vec2::ZERO,
            angle: 0.0,
            spin: 0.0,
            throttle: 1.0,
            fin: 0.0,
            fuel: 0.0,
        };
        advance(&mut r, &level, DT, DT);
        assert_eq!(r.throttle, 0.0);
        assert!(r.vel.y > 0.0);
    }
    #[test]
    fn seeded_episode_is_repeatable() {
        let p = Policy::default();
        assert_eq!(episode(&p, 42, 1).reward, episode(&p, 42, 1).reward);
        assert!(!episode(&p, 42, 1).landed);
    }

    // A separate audit set catches a replacement checkpoint that forgets early
    // levels or cannot handle gusts. None of these seeds participates in training.
    #[test]
    fn saved_model_lands_on_levels_one_through_three() {
        let p = Policy::load(concat!(env!("CARGO_MANIFEST_DIR"), "/landing-model.txt")).unwrap();
        for level in 1..=3 {
            let successes = (20001..21001)
                .filter(|&seed| episode(&p, seed, level).landed)
                .count();
            println!("Audit level {level}: {successes}/1000 landings");
            assert!(
                successes >= 990,
                "Expected >=990/1000 level {level} landings, got {successes}"
            );
        }
    }
}
