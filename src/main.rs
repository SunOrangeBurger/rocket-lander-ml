//! Visible game: draws the world and accepts keyboard input.
//! Teaching entry points: learning.rs (brain), physics.rs (world), bin/train.rs
//! (learning algorithm), and docs/BEGINNERS.md (beginner lesson).

use equinox_stall_2::{learning::Policy, physics::*};
use macroquad::prelude::*;
use macroquad::rand::gen_range;

const SCREEN_W: f32 = 900.0;
const SCREEN_H: f32 = 700.0;
fn window_conf() -> Conf {
    Conf {
        window_title: "Rocket Re-Lander".to_owned(),
        window_width: SCREEN_W as i32,
        window_height: SCREEN_H as i32,
        ..Default::default()
    }
}

struct Particle {
    pos: Vec2,
    vel: Vec2,
    life: f32,
    max_life: f32,
    color: Color,
    size: f32,
}

#[derive(Clone)]
enum State {
    Intro,
    Playing,
    Landed(i32),
    Crashed(String),
    GameOver(String),
}

struct Game {
    state: State,
    level_n: u32,
    level: Level,
    rocket: Rocket,
    t: f32,
    score: i32,
    lives: i32,
    banner: f32,
    particles: Vec<Particle>,
    stars: Vec<Vec2>,
    clouds: Vec<(Vec2, f32)>,
    policy: Policy,
    autopilot: bool,
    model_status: String,
    accumulator: f32,
}

fn make_level(n: u32) -> Level {
    let k = n as f32 - 1.0;
    let sign = if gen_range(0, 2) == 0 { -1.0 } else { 1.0 };
    let wind_base = if n == 1 {
        0.0
    } else {
        sign * (20.0 + 12.0 * (k - 1.0)).min(110.0)
    };
    let gust = if n >= 3 {
        (10.0 + 6.0 * (k - 2.0)).min(50.0)
    } else {
        0.0
    };
    let pad_w = (200.0 - 18.0 * k).max(80.0);
    let pad_range = if n >= 4 {
        (120.0 + 20.0 * (k - 3.0)).min(300.0)
    } else {
        0.0
    };
    Level {
        wind_base,
        gust,
        pad_w,
        pad_home: gen_range(1000.0, WORLD_W - 1000.0),
        pad_range,
        pad_freq: 0.3,
    }
}

fn new_rocket(level: &Level) -> Rocket {
    let x = (level.pad_home + gen_range(-600.0, 600.0)).clamp(300.0, WORLD_W - 300.0);
    Rocket {
        pos: vec2(x, 100.0),
        vel: vec2(gen_range(-20.0, 20.0), 60.0),
        angle: gen_range(-0.15, 0.15),
        spin: 0.0,
        throttle: 0.0,
        fin: 0.0,
        fuel: FUEL_START,
    }
}

fn rot(p: Vec2, a: f32) -> Vec2 {
    let (s, c) = a.sin_cos();
    vec2(p.x * c - p.y * s, p.x * s + p.y * c)
}

fn quad(a: Vec2, b: Vec2, c: Vec2, d: Vec2, color: Color) {
    draw_triangle(a, b, c, color);
    draw_triangle(a, c, d, color);
}

fn center_text(s: &str, y: f32, size: f32, color: Color) {
    let d = measure_text(s, None, size as u16, 1.0);
    draw_text(s, (SCREEN_W - d.width) / 2.0, y, size, color);
}

fn sky_color(alt: f32) -> Color {
    let t = (alt / 2200.0).clamp(0.0, 1.0);
    let lo = (0.45, 0.70, 0.95);
    let hi = (0.01, 0.01, 0.06);
    Color::new(
        lo.0 + (hi.0 - lo.0) * t,
        lo.1 + (hi.1 - lo.1) * t,
        lo.2 + (hi.2 - lo.2) * t,
        1.0,
    )
}

impl Game {
    fn new() -> Self {
        let level = make_level(1);
        let rocket = new_rocket(&level);
        let mut stars = Vec::new();
        for _ in 0..220 {
            stars.push(vec2(
                gen_range(0.0, WORLD_W),
                gen_range(0.0, GROUND_Y - 400.0),
            ));
        }
        let mut clouds = Vec::new();
        for _ in 0..40 {
            clouds.push((
                vec2(gen_range(0.0, WORLD_W), gen_range(1100.0, 1950.0)),
                gen_range(30.0, 80.0),
            ));
        }
        let (policy, model_status) = match Policy::load("landing-model.txt") {
            Ok(p) => (p, "Trained model loaded".to_string()),
            Err(_) => (
                Policy::default(),
                "No model: run the trainer first".to_string(),
            ),
        };
        Game {
            state: State::Intro,
            level_n: 1,
            level,
            rocket,
            t: 0.0,
            score: 0,
            lives: 3,
            banner: 0.0,
            particles: Vec::new(),
            stars,
            clouds,
            policy,
            autopilot: false,
            model_status,
            accumulator: 0.0,
        }
    }

    fn start_level(&mut self, n: u32) {
        self.level_n = n;
        self.level = make_level(n);
        self.retry_level();
    }

    fn retry_level(&mut self) {
        self.rocket = new_rocket(&self.level);
        self.t = 0.0;
        self.accumulator = 0.0;
        self.banner = 2.5;
        self.particles.clear();
        self.state = State::Playing;
    }

    fn pad_x(&self) -> f32 {
        pad_x(&self.level, self.t)
    }
    fn pad_vx(&self) -> f32 {
        pad_vx(&self.level, self.t)
    }
    fn wind_at(&self, alt: f32) -> f32 {
        wind_at(&self.level, self.t, alt)
    }

    fn camera(&self) -> Vec2 {
        let cx = (self.rocket.pos.x - SCREEN_W / 2.0).clamp(0.0, WORLD_W - SCREEN_W);
        let cy = (self.rocket.pos.y - 350.0).clamp(-200.0, GROUND_Y + 80.0 - SCREEN_H);
        vec2(cx, cy)
    }

    fn update(&mut self, dt: f32) {
        if is_key_pressed(KeyCode::M) {
            self.autopilot = !self.autopilot;
        }
        if is_key_pressed(KeyCode::L) {
            match Policy::load("landing-model.txt") {
                Ok(p) => {
                    self.policy = p;
                    self.model_status = "Trained model loaded".into();
                }
                Err(e) => self.model_status = format!("Model load failed: {e}"),
            }
        }
        // particles always animate
        for p in self.particles.iter_mut() {
            p.pos += p.vel * dt;
            p.vel *= 1.0 - 0.8 * dt;
            p.life -= dt;
        }
        self.particles.retain(|p| p.life > 0.0);

        let enter = is_key_pressed(KeyCode::Enter) || is_key_pressed(KeyCode::Space);
        match self.state.clone() {
            State::Intro => {
                if enter {
                    self.retry_level();
                }
            }
            State::Playing => {
                self.banner = (self.banner - dt).max(0.0);
                self.accumulator += dt;
                let mut first_tick = true;
                while self.accumulator >= DT && matches!(self.state, State::Playing) {
                    self.step(DT, first_tick);
                    first_tick = false;
                    self.accumulator -= DT;
                }
            }
            State::Landed(_) => {
                if enter {
                    let n = self.level_n + 1;
                    self.start_level(n);
                }
            }
            State::Crashed(_) => {
                if enter || is_key_pressed(KeyCode::R) {
                    self.retry_level();
                }
            }
            State::GameOver(_) => {
                if enter || is_key_pressed(KeyCode::R) {
                    let autopilot = self.autopilot;
                    *self = Game::new();
                    self.autopilot = autopilot;
                    self.retry_level();
                }
            }
        }
    }

    fn step(&mut self, dt: f32, first_tick: bool) {
        self.t += dt;
        let mut r = self.rocket;

        // --- input ---
        if is_key_down(KeyCode::Up) || is_key_down(KeyCode::W) {
            r.throttle += 0.7 * dt;
        }
        if is_key_down(KeyCode::Down) || is_key_down(KeyCode::S) {
            r.throttle -= 0.7 * dt;
        }
        if first_tick && is_key_pressed(KeyCode::X) {
            r.throttle = 1.0;
        }
        if first_tick && is_key_pressed(KeyCode::Z) {
            r.throttle = 0.0;
        }
        if is_key_down(KeyCode::Left) || is_key_down(KeyCode::A) {
            r.fin -= 1.6 * dt;
        }
        if is_key_down(KeyCode::Right) || is_key_down(KeyCode::D) {
            r.fin += 1.6 * dt;
        }
        if first_tick && is_key_pressed(KeyCode::C) {
            r.fin = 0.0;
        }
        r.throttle = r.throttle.clamp(0.0, 1.0);
        r.fin = r.fin.clamp(-1.0, 1.0);
        // In ML mode the learned policy supplies the two controls instead of
        // the player. It uses exactly the same physics and fuel limits below.
        if self.autopilot {
            self.policy.act(&mut r, &self.level, self.t);
        }
        advance(&mut r, &self.level, self.t, dt);

        // exhaust smoke
        if r.throttle > 0.05 {
            for _ in 0..2 {
                let base = rot(vec2(0.0, FOOT_Y), r.angle) + r.pos;
                let dir = rot(vec2(gen_range(-0.25, 0.25), 1.0), r.angle);
                let speed = gen_range(60.0, 140.0) * (0.4 + r.throttle);
                let g = gen_range(0.5, 0.9);
                self.particles.push(Particle {
                    pos: base,
                    vel: r.vel * 0.3 + dir * speed,
                    life: 0.6,
                    max_life: 0.6,
                    color: Color::new(g, g, g, 1.0),
                    size: gen_range(3.0, 6.0),
                });
            }
        }

        self.rocket = r;

        // --- out of bounds ---
        if r.pos.x < 0.0 || r.pos.x > WORLD_W {
            self.crash("Lost downrange - way off course!".to_string());
            return;
        }
        if r.pos.y < -700.0 {
            self.crash("Flew off into space...".to_string());
            return;
        }

        // --- touchdown ---
        let f1 = rot(vec2(-16.0, FOOT_Y), r.angle) + r.pos;
        let f2 = rot(vec2(16.0, FOOT_Y), r.angle) + r.pos;
        let lowest = f1.y.max(f2.y);
        if lowest >= GROUND_Y {
            self.touchdown(lowest - GROUND_Y);
        }
    }

    fn touchdown(&mut self, overshoot: f32) {
        let r = self.rocket;
        let half = self.level.pad_w / 2.0;
        let dx = r.pos.x - self.pad_x();

        let reason = landing_error(&r, &self.level, self.t);

        match reason {
            None => {
                let accuracy = 1.0 - dx.abs() / half;
                let softness = 1.0 - (r.vel.y.max(0.0) / MAX_VY);
                let fuel_bonus = r.fuel / FUEL_START;
                let points = (400.0
                    + 300.0 * accuracy
                    + 200.0 * softness
                    + 300.0 * fuel_bonus
                    + 100.0 * self.level_n as f32) as i32;
                self.score += points;
                self.rocket.pos.y -= overshoot;
                self.rocket.vel = Vec2::ZERO;
                self.rocket.spin = 0.0;
                self.rocket.throttle = 0.0;
                self.state = State::Landed(points);
            }
            Some(msg) => self.crash(msg),
        }
    }

    fn crash(&mut self, msg: String) {
        let pos = self.rocket.pos;
        for _ in 0..90 {
            let a = gen_range(0.0, std::f32::consts::TAU);
            let s = gen_range(40.0, 280.0);
            let life = gen_range(0.6, 1.6);
            let color = match gen_range(0, 3) {
                0 => ORANGE,
                1 => YELLOW,
                _ => GRAY,
            };
            self.particles.push(Particle {
                pos,
                vel: vec2(a.cos(), a.sin()) * s,
                life,
                max_life: life,
                color,
                size: gen_range(3.0, 9.0),
            });
        }
        self.lives -= 1;
        self.rocket.throttle = 0.0;
        self.state = if self.lives <= 0 {
            State::GameOver(msg)
        } else {
            State::Crashed(msg)
        };
    }

    // ------------------------------------------------------------------ draw

    fn draw(&self) {
        let cam = self.camera();
        self.draw_world(cam);
        draw_text(
            if self.autopilot {
                "M: ML PILOT"
            } else {
                "M: MANUAL PILOT"
            },
            280.0,
            28.0,
            22.0,
            YELLOW,
        );
        draw_text(&self.model_status, 280.0, 50.0, 18.0, WHITE);
        draw_text("L: reload trained model", 280.0, 70.0, 18.0, LIGHTGRAY);
        match &self.state {
            State::Intro => self.draw_intro(),
            State::Playing => {
                self.draw_hud(cam);
                if self.banner > 0.0 {
                    center_text(&format!("LEVEL {}", self.level_n), 150.0, 60.0, WHITE);
                    center_text(&self.level_hint(), 190.0, 26.0, LIGHTGRAY);
                }
            }
            State::Landed(p) => {
                self.draw_hud(cam);
                self.overlay();
                center_text("TOUCHDOWN!", 270.0, 80.0, GREEN);
                center_text(&format!("+{} points", p), 330.0, 36.0, WHITE);
                center_text(
                    &format!("Total score: {}", self.score),
                    370.0,
                    30.0,
                    LIGHTGRAY,
                );
                center_text("Press ENTER for next level", 440.0, 30.0, YELLOW);
            }
            State::Crashed(msg) => {
                self.draw_hud(cam);
                self.overlay();
                center_text("CRASHED!", 270.0, 80.0, RED);
                center_text(msg, 330.0, 30.0, WHITE);
                center_text(
                    &format!("Rockets left: {}", self.lives),
                    370.0,
                    28.0,
                    LIGHTGRAY,
                );
                center_text("Press ENTER to retry", 440.0, 30.0, YELLOW);
            }
            State::GameOver(msg) => {
                self.draw_hud(cam);
                self.overlay();
                center_text("GAME OVER", 270.0, 80.0, RED);
                center_text(msg, 330.0, 30.0, WHITE);
                center_text(&format!("Final score: {}", self.score), 375.0, 36.0, YELLOW);
                center_text("Press ENTER to play again", 440.0, 30.0, YELLOW);
            }
        }
    }

    fn level_hint(&self) -> String {
        let mut parts = Vec::new();
        if self.level.wind_base != 0.0 {
            parts.push("windy");
        }
        if self.level.gust > 0.0 {
            parts.push("gusts");
        }
        if self.level.pad_range > 0.0 {
            parts.push("moving barge");
        }
        if parts.is_empty() {
            "calm day".to_string()
        } else {
            parts.join(", ")
        }
    }

    fn overlay(&self) {
        draw_rectangle(
            0.0,
            0.0,
            SCREEN_W,
            SCREEN_H,
            Color::new(0.0, 0.0, 0.0, 0.45),
        );
    }

    fn draw_world(&self, cam: Vec2) {
        // sky gradient in bands
        let bands = 40;
        let bh = SCREEN_H / bands as f32;
        for i in 0..bands {
            let sy = i as f32 * bh;
            let alt = GROUND_Y - (cam.y + sy);
            draw_rectangle(0.0, sy, SCREEN_W, bh + 1.0, sky_color(alt));
        }

        // stars
        for s in &self.stars {
            let alt = GROUND_Y - s.y;
            let a = ((alt / 2200.0) - 0.35) / 0.65;
            if a > 0.0 {
                let sx = s.x - cam.x;
                let sy = s.y - cam.y;
                if sx > -5.0 && sx < SCREEN_W + 5.0 && sy > -5.0 && sy < SCREEN_H + 5.0 {
                    draw_circle(sx, sy, 1.3, Color::new(1.0, 1.0, 1.0, a.clamp(0.0, 1.0)));
                }
            }
        }

        // clouds (give a sense of speed and wind)
        for (p, size) in &self.clouds {
            let sx = p.x - cam.x;
            let sy = p.y - cam.y;
            if sx > -150.0 && sx < SCREEN_W + 150.0 && sy > -100.0 && sy < SCREEN_H + 100.0 {
                let c = Color::new(1.0, 1.0, 1.0, 0.45);
                draw_circle(sx, sy, *size, c);
                draw_circle(sx + size * 0.9, sy + size * 0.15, size * 0.75, c);
                draw_circle(sx - size * 0.9, sy + size * 0.2, size * 0.7, c);
            }
        }

        // sea
        let sea_y = GROUND_Y - cam.y + 6.0;
        if sea_y < SCREEN_H {
            draw_rectangle(
                0.0,
                sea_y,
                SCREEN_W,
                SCREEN_H - sea_y + 10.0,
                Color::new(0.05, 0.2, 0.38, 1.0),
            );
            let mut x = 0.0;
            while x < SCREEN_W {
                let wy = sea_y + ((x + cam.x) * 0.03 + self.t * 2.0).sin() * 3.0;
                draw_line(x, wy, x + 30.0, wy, 2.0, Color::new(0.2, 0.45, 0.7, 1.0));
                x += 30.0;
            }
        }

        // landing barge
        let px = self.pad_x() - cam.x;
        let py = GROUND_Y - cam.y;
        let half = self.level.pad_w / 2.0;
        if py < SCREEN_H + 40.0 {
            draw_rectangle(
                px - half - 20.0,
                py,
                self.level.pad_w + 40.0,
                16.0,
                Color::new(0.2, 0.2, 0.22, 1.0),
            );
            draw_rectangle(
                px - half,
                py - 2.0,
                self.level.pad_w,
                6.0,
                Color::new(0.4, 0.4, 0.44, 1.0),
            );
            draw_rectangle(px - half, py - 2.0, 6.0, 6.0, ORANGE);
            draw_rectangle(px + half - 6.0, py - 2.0, 6.0, 6.0, ORANGE);
            draw_rectangle(px - 3.0, py - 2.0, 6.0, 6.0, YELLOW);
            let blink = (self.t * 4.0).sin() > 0.0;
            let lc = if blink {
                RED
            } else {
                Color::new(0.4, 0.0, 0.0, 1.0)
            };
            draw_circle(px - half - 14.0, py - 3.0, 3.0, lc);
            draw_circle(px + half + 14.0, py - 3.0, 3.0, lc);
        }

        // rocket
        let show_rocket = matches!(self.state, State::Playing | State::Landed(_) | State::Intro);
        if show_rocket {
            self.draw_rocket(cam);
        }

        // particles
        for p in &self.particles {
            let k = (p.life / p.max_life).clamp(0.0, 1.0);
            let mut c = p.color;
            c.a = k;
            draw_circle(
                p.pos.x - cam.x,
                p.pos.y - cam.y,
                p.size * (0.4 + 0.6 * k),
                c,
            );
        }
    }

    fn draw_rocket(&self, cam: Vec2) {
        let r = &self.rocket;
        let sp = r.pos - cam;
        let a = r.angle;
        let p = |x: f32, y: f32| rot(vec2(x, y), a) + sp;

        // flame
        if r.throttle > 0.01 {
            let len = (12.0 + 50.0 * r.throttle) * gen_range(0.8, 1.1);
            draw_triangle(p(-5.0, 30.0), p(5.0, 30.0), p(0.0, 30.0 + len), ORANGE);
            draw_triangle(
                p(-3.0, 30.0),
                p(3.0, 30.0),
                p(0.0, 30.0 + len * 0.6),
                YELLOW,
            );
        }

        // legs
        let leg = Color::new(0.75, 0.75, 0.8, 1.0);
        let l1 = p(-7.0, 20.0);
        let l2 = p(-16.0, FOOT_Y);
        let l3 = p(7.0, 20.0);
        let l4 = p(16.0, FOOT_Y);
        draw_line(l1.x, l1.y, l2.x, l2.y, 3.0, leg);
        draw_line(l3.x, l3.y, l4.x, l4.y, 3.0, leg);

        // body, stripe, nose
        quad(
            p(-7.0, -22.0),
            p(7.0, -22.0),
            p(7.0, 30.0),
            p(-7.0, 30.0),
            WHITE,
        );
        quad(
            p(-7.0, -2.0),
            p(7.0, -2.0),
            p(7.0, 8.0),
            p(-7.0, 8.0),
            Color::new(0.15, 0.15, 0.18, 1.0),
        );
        draw_triangle(
            p(-7.0, -22.0),
            p(7.0, -22.0),
            p(0.0, -42.0),
            Color::new(0.85, 0.2, 0.2, 1.0),
        );

        // grid fins (deflect with the fin setting)
        for side in [-1.0f32, 1.0] {
            let c = p(side * 11.0, -17.0);
            let dir = rot(vec2(1.0, 0.0), a + r.fin * 0.9 * side);
            let e1 = c - dir * 6.0;
            let e2 = c + dir * 6.0;
            draw_line(e1.x, e1.y, e2.x, e2.y, 3.0, ORANGE);
        }
    }

    fn draw_hud(&self, _cam: Vec2) {
        let r = &self.rocket;
        let alt = (GROUND_Y - (r.pos.y + FOOT_Y)).max(0.0);
        let rel_vx = r.vel.x - self.pad_vx();
        let wind = self.wind_at(alt);

        draw_rectangle(10.0, 10.0, 250.0, 238.0, Color::new(0.0, 0.0, 0.0, 0.5));
        let good = Color::new(0.4, 1.0, 0.5, 1.0);
        let bad = Color::new(1.0, 0.45, 0.4, 1.0);
        let ok = |c: bool| if c { good } else { bad };

        draw_text(format!("ALT       {:>6.0} m", alt), 22.0, 36.0, 24.0, WHITE);
        draw_text(
            format!("FALL SPD  {:>6.0}", r.vel.y),
            22.0,
            60.0,
            24.0,
            ok(r.vel.y <= MAX_VY),
        );
        draw_text(
            format!("DRIFT     {:>6.0}", rel_vx),
            22.0,
            84.0,
            24.0,
            ok(rel_vx.abs() <= MAX_VX),
        );
        draw_text(
            format!("TILT      {:>6.0} deg", r.angle.to_degrees()),
            22.0,
            108.0,
            24.0,
            ok(r.angle.abs() <= MAX_TILT),
        );

        let bar = |y: f32, label: &str, frac: f32, color: Color| {
            draw_text(label, 22.0, y + 14.0, 20.0, LIGHTGRAY);
            draw_rectangle(100.0, y, 150.0, 16.0, Color::new(0.2, 0.2, 0.2, 1.0));
            draw_rectangle(100.0, y, 150.0 * frac.clamp(0.0, 1.0), 16.0, color);
        };
        let fuel_frac = r.fuel / FUEL_START;
        let fuel_col = if fuel_frac > 0.3 { good } else { bad };
        bar(122.0, "FUEL", fuel_frac, fuel_col);
        bar(146.0, "THROTTLE", r.throttle, ORANGE);

        // fin gauge (centered)
        draw_text("FINS", 22.0, 188.0, 20.0, LIGHTGRAY);
        draw_rectangle(100.0, 174.0, 150.0, 16.0, Color::new(0.2, 0.2, 0.2, 1.0));
        draw_rectangle(174.0, 172.0, 2.0, 20.0, WHITE);
        let fw = 75.0 * r.fin;
        draw_rectangle(175.0_f32.min(175.0 + fw), 174.0, fw.abs(), 16.0, SKYBLUE);

        draw_text("SAFE: fall<80 drift<30 tilt<11", 18.0, 214.0, 18.0, GRAY);
        draw_text("Land in the pad's center!", 18.0, 236.0, 18.0, GRAY);

        // right panel: wind + score
        draw_rectangle(
            SCREEN_W - 190.0,
            10.0,
            180.0,
            118.0,
            Color::new(0.0, 0.0, 0.0, 0.5),
        );
        draw_text(
            format!("LEVEL {}", self.level_n),
            SCREEN_W - 178.0,
            36.0,
            24.0,
            WHITE,
        );
        draw_text(
            format!("SCORE {}", self.score),
            SCREEN_W - 178.0,
            60.0,
            24.0,
            YELLOW,
        );
        draw_text(
            format!("ROCKETS {}", self.lives),
            SCREEN_W - 178.0,
            84.0,
            24.0,
            WHITE,
        );
        draw_text("WIND", SCREEN_W - 178.0, 112.0, 22.0, LIGHTGRAY);
        let ax = SCREEN_W - 70.0;
        let ay = 106.0;
        let len = (wind / 110.0 * 50.0).clamp(-55.0, 55.0);
        draw_line(ax, ay, ax + len, ay, 4.0, SKYBLUE);
        if len.abs() > 4.0 {
            let d = len.signum();
            draw_triangle(
                vec2(ax + len + 9.0 * d, ay),
                vec2(ax + len, ay - 7.0),
                vec2(ax + len, ay + 7.0),
                SKYBLUE,
            );
        }
        draw_text(
            format!("{:+.0}", wind),
            SCREEN_W - 118.0,
            112.0,
            20.0,
            WHITE,
        );

        // pad direction arrows when off screen
        let cam = self.camera();
        let pad_sx = self.pad_x() - cam.x;
        let dist = self.pad_x() - r.pos.x;
        if pad_sx < 30.0 {
            draw_triangle(
                vec2(14.0, 330.0),
                vec2(34.0, 316.0),
                vec2(34.0, 344.0),
                YELLOW,
            );
            draw_text(format!("PAD {:.0}", dist.abs()), 40.0, 337.0, 22.0, YELLOW);
        } else if pad_sx > SCREEN_W - 30.0 {
            draw_triangle(
                vec2(SCREEN_W - 14.0, 330.0),
                vec2(SCREEN_W - 34.0, 316.0),
                vec2(SCREEN_W - 34.0, 344.0),
                YELLOW,
            );
            draw_text(
                format!("PAD {:.0}", dist.abs()),
                SCREEN_W - 110.0,
                337.0,
                22.0,
                YELLOW,
            );
        } else if GROUND_Y - cam.y > SCREEN_H - 10.0 {
            // pad is horizontally in view but still below the bottom of the screen
            let x = pad_sx.clamp(60.0, SCREEN_W - 60.0);
            draw_triangle(
                vec2(x, SCREEN_H - 12.0),
                vec2(x - 14.0, SCREEN_H - 34.0),
                vec2(x + 14.0, SCREEN_H - 34.0),
                YELLOW,
            );
            let label = format!("PAD {:+.0}", dist);
            let d = measure_text(&label, None, 22, 1.0);
            draw_text(&label, x - d.width / 2.0, SCREEN_H - 42.0, 22.0, YELLOW);
        }
    }

    fn draw_intro(&self) {
        self.overlay();
        center_text("ROCKET RE-LANDER", 170.0, 72.0, WHITE);
        center_text(
            "Bring your booster home. Land softly on the barge.",
            215.0,
            28.0,
            LIGHTGRAY,
        );
        let lines = [
            "UP / DOWN  or  W / S    throttle",
            "Z cut engine      X full burn",
            "LEFT / RIGHT  or  A / D    steer grid fins",
            "C    center fins    M    toggle ML pilot    L    reload model",
            "",
            "Fins tilt the rocket and push it sideways.",
            "They bite hard when falling fast, barely at all when slow.",
            "Wind shoves you off course. Fuel is limited.",
            "Touch down upright, slow, and on the pad.",
        ];
        let mut y = 290.0;
        for l in lines {
            center_text(l, y, 26.0, WHITE);
            y += 30.0;
        }
        center_text("Press ENTER to launch", 620.0, 36.0, YELLOW);
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    macroquad::rand::srand(macroquad::miniquad::date::now() as u64);
    let mut game = Game::new();
    // Optional shortcuts for demonstrations: jump straight into a chosen level.
    let mut args = std::env::args().skip(1);
    let mut initial_level = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ml" => game.autopilot = true,
            "--level" => {
                let n: u32 = args
                    .next()
                    .expect("--level needs a number")
                    .parse()
                    .expect("level must be a positive integer");
                assert!(n > 0, "level must be at least 1");
                initial_level = Some(n);
            }
            _ => panic!("Unknown argument {arg}; use --level NUMBER and/or --ml"),
        }
    }
    if let Some(n) = initial_level {
        game.start_level(n);
    }
    loop {
        game.update(get_frame_time().min(0.1));
        game.draw();
        next_frame().await;
    }
}
