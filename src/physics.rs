//! BEGINNER WALKTHROUGH: the world that the pilot must learn to control.
//!
//! This module is shared by the visible game and invisible training flights.
//! Coordinates follow screen convention: x increases right, y increases DOWN.
//! Therefore positive vertical velocity means falling. `DT` is 1/60 second.
//! The equations and safe-landing limits are fixed rules; they are not learned.

use macroquad::prelude::*;
pub const WORLD_W: f32 = 2600.0;
pub const GROUND_Y: f32 = 2200.0;

pub const GRAVITY: f32 = 50.0;
pub const MAX_THRUST: f32 = 135.0;
pub const DRAG_K: f32 = 0.0003;
pub const FUEL_START: f32 = 60.0;
pub const BURN_RATE: f32 = 6.0;
pub const FOOT_Y: f32 = 33.0;

// Safe landing limits
pub const MAX_VY: f32 = 80.0;
pub const MAX_VX: f32 = 30.0;
pub const MAX_TILT: f32 = 0.2;
pub const MAX_SPIN: f32 = 1.0;

#[derive(Clone)]
pub struct Level {
    pub wind_base: f32,
    pub gust: f32,
    pub pad_w: f32,
    pub pad_home: f32,
    pub pad_range: f32,
    pub pad_freq: f32,
}

#[derive(Clone, Copy)]
pub struct Rocket {
    pub pos: Vec2,
    pub vel: Vec2,
    pub angle: f32,
    pub spin: f32,
    pub throttle: f32,
    pub fin: f32,
    pub fuel: f32,
}

pub const DT: f32 = 1.0 / 60.0;
pub fn pad_x(level: &Level, t: f32) -> f32 {
    level.pad_home + level.pad_range * (level.pad_freq * t).sin()
}
pub fn pad_vx(level: &Level, t: f32) -> f32 {
    level.pad_range * level.pad_freq * (level.pad_freq * t).cos()
}
pub fn wind_at(level: &Level, t: f32, alt: f32) -> f32 {
    let g = (t * 0.7).sin() + 0.5 * (t * 1.9 + 1.0).sin();
    let w = level.wind_base + level.gust * g / 1.5;
    w * (0.35 + 0.65 * (alt / 1500.0).clamp(0.0, 1.0))
}
/// Simulate one time step: apply fuel limits, add forces, update motion,
/// then update rotation. A model can request a full burn with no fuel, but the
/// simulator enforces an empty tank, so training cannot bypass the game rules.
pub fn advance(r: &mut Rocket, level: &Level, t: f32, dt: f32) {
    r.throttle = r.throttle.clamp(0.0, 1.0);
    r.fin = r.fin.clamp(-1.0, 1.0);
    if r.fuel <= 0.0 {
        r.throttle = 0.0;
    }
    r.fuel = (r.fuel - r.throttle * BURN_RATE * dt).max(0.0);
    // --- physics ---
    let alt = GROUND_Y - (r.pos.y + FOOT_Y);
    let wind = wind_at(level, t, alt);
    let air = r.vel - vec2(wind, 0.0);
    let air_speed = air.length();
    let up = vec2(r.angle.sin(), -r.angle.cos());
    let right = vec2(r.angle.cos(), r.angle.sin());

    let mut acc = vec2(0.0, GRAVITY);
    acc += air * (-DRAG_K * air_speed); // air drag
    acc += vec2(wind * 0.25, 0.0); // wind shoves the rocket sideways
    acc += up * (r.throttle * MAX_THRUST); // engine
    acc += right * (r.fin * air_speed.min(400.0) * 0.15); // fin lift (needs airflow)
                                                          // acceleration changes velocity; velocity changes position.
                                                          // For example, acceleration 50 for 1/60 second adds about 0.83 to speed.
    r.vel += acc * dt;
    r.pos += r.vel * dt;

    // fins command a tilt; response is stronger with airflow or engine gimbal
    let authority = 0.3 + 0.7 * (air_speed / 250.0).min(1.0) + 0.5 * r.throttle;
    let target = r.fin * 0.7;
    let ang_acc = authority * (8.0 * (target - r.angle) - 5.0 * r.spin);
    r.spin += ang_acc * dt;
    r.angle += r.spin * dt;
}
/// Check the lower of the two rotated feet, not just the rocket's center.
/// A nonnegative result means a foot has reached or crossed ground height.
pub fn contact(r: &Rocket) -> f32 {
    let foot = |x: f32| r.pos.y + x * r.angle.sin() + FOOT_Y * r.angle.cos();
    foot(-16.0).max(foot(16.0)) - GROUND_Y
}
/// Landing requires ALL five checks: on pad, slow fall, low sideways speed,
/// upright, and low spin. `None` means every check passed.
pub fn landing_error(r: &Rocket, level: &Level, t: f32) -> Option<String> {
    if (r.pos.x - pad_x(level, t)).abs() > level.pad_w / 2.0 {
        Some("Missed the landing pad - splashdown!".into())
    } else if r.vel.y > MAX_VY {
        Some(format!(
            "Hit too hard: {:.0} m/s (max {:.0})",
            r.vel.y, MAX_VY
        ))
    } else if (r.vel.x - pad_vx(level, t)).abs() > MAX_VX {
        Some("Sliding sideways at touchdown!".into())
    } else if r.angle.abs() > MAX_TILT {
        Some("Tipped over at touchdown!".into())
    } else if r.spin.abs() > MAX_SPIN {
        Some("Still spinning at touchdown!".into())
    } else {
        None
    }
}
