//! Live teaching demo: real training, with an animated champion after each generation.
//! Starts from zero, keeps snapshots for comparison, and saves only on request.
use equinox_stall_2::{
    learning::{Flight, Policy},
    physics::*,
    training::{GenerationStats, Trainer, POPULATION},
};
use macroquad::prelude::*;
use std::{fs, io};

const BG: Color = Color::new(0.035, 0.055, 0.10, 1.0);
const PANEL: Color = Color::new(0.065, 0.09, 0.15, 1.0);
const MUTED: Color = Color::new(0.58, 0.67, 0.80, 1.0);
const MINT: Color = Color::new(0.28, 0.92, 0.70, 1.0);
const SALMON: Color = Color::new(1.0, 0.43, 0.42, 1.0);
const GOLD: Color = Color::new(1.0, 0.77, 0.30, 1.0);
fn config() -> Conf {
    Conf {
        window_title: "Landing school - watch a pilot learn".into(),
        window_width: 1180,
        window_height: 760,
        window_resizable: false,
        ..Default::default()
    }
}
#[derive(Clone)]
struct Snapshot {
    policy: Policy,
    stats: GenerationStats,
}
struct Demo {
    trainer: Trainer,
    snapshots: Vec<Snapshot>,
    selected: usize,
    flight: Flight,
    trail: Vec<Vec2>,
    max_level: u32,
    paused: bool,
    speed: f32,
    accumulator: f32,
    hold: f32,
    training: bool,
    message: String,
    limit: usize,
}
impl Demo {
    fn new(max_level: u32, limit: usize) -> Self {
        let trainer = Trainer::new(Policy::default(), max_level);
        let snapshots = vec![Snapshot {
            policy: trainer.best.clone(),
            stats: trainer.stats.clone(),
        }];
        Self {
            trainer,
            snapshots,
            selected: 0,
            flight: Flight::new(7, max_level),
            trail: Vec::new(),
            max_level,
            paused: true,
            speed: 4.0,
            accumulator: 0.0,
            hold: 0.0,
            training: false,
            message: "Generation 0: zero weights. Click Resume to watch the first attempt.".into(),
            limit,
        }
    }
    fn replay(&mut self, selected: usize) {
        self.selected = selected;
        self.flight = Flight::new(7, self.max_level);
        self.trail.clear();
        self.accumulator = 0.0;
        self.hold = 0.0;
        self.training = false;
        self.message = format!(
            "Replaying generation {} on the same start. The graphs summarize all practice starts.",
            selected
        );
    }
    fn export(&self) -> io::Result<()> {
        // Never overwrite the game's proven checkpoint with an early failing pilot.
        self.snapshots[self.selected]
            .policy
            .save(concat!(env!("CARGO_MANIFEST_DIR"), "/demo-model.txt"))?;
        let mut csv = String::from(
            "generation,champion_reward,champion_landings,practice_cases,candidate_landing_rate\n",
        );
        for snapshot in &self.snapshots {
            let s = &snapshot.stats;
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                s.generation, s.reward, s.landings, s.cases, s.candidate_landing_rate
            ));
        }
        fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/training-demo-history.csv"),
            csv,
        )
    }
    fn update(&mut self, dt: f32) {
        if is_key_pressed(KeyCode::R) || clicked(554.0, 170.0) {
            *self = Self::new(self.max_level, self.limit);
        }
        if is_key_pressed(KeyCode::Space) || clicked(24.0, 174.0) {
            self.paused = !self.paused;
        }
        if is_key_pressed(KeyCode::Equal)
            || is_key_pressed(KeyCode::KpAdd)
            || clicked(1032.0, 124.0)
        {
            self.speed = (self.speed * 2.0).min(32.0);
        }
        if is_key_pressed(KeyCode::Minus)
            || is_key_pressed(KeyCode::KpSubtract)
            || clicked(908.0, 112.0)
        {
            self.speed = (self.speed / 2.0).max(0.5);
        }
        if (is_key_pressed(KeyCode::Left) || clicked(210.0, 160.0)) && self.selected > 0 {
            self.replay(self.selected - 1);
            self.paused = false;
        }
        if is_key_pressed(KeyCode::Right) && self.selected + 1 < self.snapshots.len() {
            self.replay(self.selected + 1);
            self.paused = false;
        }
        if is_key_pressed(KeyCode::S) || clicked(736.0, 160.0) {
            self.message = match self.export() {
                Ok(()) => format!(
                    "Saved generation {} to demo-model.txt + training-demo-history.csv",
                    self.selected
                ),
                Err(e) => format!("Save failed: {e}"),
            };
        }
        // N moves to the next saved generation, or starts real training at the frontier.
        if is_key_pressed(KeyCode::N) || clicked(382.0, 160.0) {
            if self.selected + 1 < self.snapshots.len() {
                self.replay(self.selected + 1);
            } else if self.trainer.stats.generation < self.limit {
                self.training = true;
            }
            self.paused = false;
        }
        if self.paused {
            return;
        }
        if self.training {
            // One candidate per frame: normal event handling continues while learning.
            if let Some(stats) = self.trainer.advance_candidate() {
                self.snapshots.push(Snapshot {
                    policy: self.trainer.best.clone(),
                    stats,
                });
                self.replay(self.snapshots.len() - 1);
            }
            return;
        }
        if self.flight.result.is_none() {
            self.accumulator += dt * self.speed;
            while self.accumulator >= DT && self.flight.result.is_none() {
                self.flight.step(&self.snapshots[self.selected].policy);
                self.trail.push(self.flight.rocket.pos);
                self.accumulator -= DT;
            }
        } else {
            self.hold += dt;
            // Older snapshots replay once and wait so the audience can compare them.
            if self.hold >= 1.5
                && self.selected + 1 == self.snapshots.len()
                && self.trainer.stats.generation < self.limit
            {
                self.training = true;
            }
        }
    }
    fn draw(&self) {
        clear_background(BG);
        draw_text("LANDING SCHOOL", 24.0, 36.0, 32.0, WHITE);
        draw_text(
            "Watch a pilot learn from real practice flights",
            326.0,
            35.0,
            23.0,
            MUTED,
        );
        self.draw_controls();
        self.draw_flight();
        self.draw_stats();
        draw_rectangle(24.0, 666.0, 1132.0, 68.0, PANEL);
        fit_text(&self.message, 40.0, 692.0, 1100.0, 19.0, WHITE);
        draw_text("Training starts from zero. Replays show the champion; candidate results include exploratory failures.", 40.0, 718.0, 18.0, MUTED);
    }
    fn draw_controls(&self) {
        for (x, w, label) in [
            (
                24.0,
                174.0,
                if self.paused {
                    "Resume [SPACE]"
                } else {
                    "Pause [SPACE]"
                },
            ),
            (210.0, 160.0, "Previous [LEFT]"),
            (382.0, 160.0, "Next [N]"),
            (554.0, 170.0, "Start over [R]"),
            (736.0, 160.0, "Export [S]"),
            (908.0, 112.0, "Slower [-]"),
            (1032.0, 124.0, "Faster [+]"),
        ] {
            let hover = Rect::new(x, 66.0, w, 32.0).contains(Vec2::from(mouse_position()));
            draw_rectangle(
                x,
                66.0,
                w,
                32.0,
                if hover {
                    Color::new(0.13, 0.23, 0.30, 1.0)
                } else {
                    PANEL
                },
            );
            draw_text(
                label,
                x + 10.0,
                87.0,
                18.0,
                if hover { MINT } else { WHITE },
            );
        }
    }
    fn draw_flight(&self) {
        let area = Rect::new(24.0, 116.0, 620.0, 532.0);
        draw_rectangle(area.x, area.y, area.w, area.h, PANEL);
        // Fixed overview preserves the trajectory so early misses and later control are visible.
        let map = |p: Vec2| vec2(106.0 + p.x * 0.175, 116.0 + (p.y + 700.0) * 0.175);
        for alt in [0.0, 500.0, 1000.0, 1500.0, 2000.0] {
            let y = map(vec2(0.0, GROUND_Y - alt)).y;
            draw_line(100.0, y, 570.0, y, 1.0, Color::new(0.15, 0.2, 0.29, 1.0));
            draw_text(format!("{alt:.0} m"), 36.0, y + 5.0, 16.0, MUTED);
        }
        let ground = map(vec2(0.0, GROUND_Y)).y;
        draw_rectangle(
            100.0,
            ground + 3.0,
            470.0,
            20.0,
            Color::new(0.04, 0.17, 0.26, 1.0),
        );
        let level = &self.flight.level;
        let px = map(vec2(pad_x(level, self.flight.t), GROUND_Y)).x;
        draw_rectangle(
            px - level.pad_w * 0.175 / 2.0,
            ground - 3.0,
            level.pad_w * 0.175,
            6.0,
            GOLD,
        );
        for points in self.trail.windows(2) {
            let a = map(points[0]);
            let b = map(points[1]);
            draw_line(a.x, a.y, b.x, b.y, 1.5, Color::new(0.2, 0.65, 0.65, 0.65));
        }
        let r = &self.flight.rocket;
        let pos = map(r.pos);
        let shape = |x: f32, y: f32| {
            let (s, c) = r.angle.sin_cos();
            pos + vec2(x * c - y * s, x * s + y * c)
        };
        if r.throttle > 0.01 && self.flight.result.is_none() {
            draw_triangle(
                shape(-3.0, 10.0),
                shape(3.0, 10.0),
                shape(0.0, 15.0 + r.throttle * 22.0),
                ORANGE,
            );
        }
        let color = match &self.flight.result {
            Some(e) if !e.landed => SALMON,
            _ => WHITE,
        };
        draw_triangle(
            shape(-5.0, 10.0),
            shape(5.0, 10.0),
            shape(0.0, -17.0),
            color,
        );
        for side in [-1.0, 1.0] {
            let a = shape(side * 4.0, 6.0);
            let b = shape(side * 10.0, 14.0);
            draw_line(a.x, a.y, b.x, b.y, 2.0, color);
        }
        let status = if self.paused {
            "PAUSED"
        } else if self.training {
            "PRACTICING"
        } else {
            "CHAMPION REPLAY"
        };
        draw_text(status, 40.0, 144.0, 21.0, MINT);
        draw_text(
            format!(
                "Level {} | same start #7 | {:.1}x",
                self.max_level, self.speed
            ),
            40.0,
            168.0,
            18.0,
            MUTED,
        );
        if let Some(result) = &self.flight.result {
            let color = if result.landed { MINT } else { SALMON };
            draw_text(
                if result.landed { "LANDED" } else { "FAILED" },
                40.0,
                202.0,
                28.0,
                color,
            );
            fit_text(&result.reason, 40.0, 227.0, 580.0, 20.0, color);
            draw_text(
                format!("This flight's reward: {:.2}", result.reward),
                40.0,
                252.0,
                19.0,
                MUTED,
            );
        }
        draw_text(
            format!(
                "Fall {:>4.0} m/s    Fuel {:>3.0}%    Throttle {:>3.0}%",
                r.vel.y,
                100.0 * r.fuel / FUEL_START,
                100.0 * r.throttle
            ),
            40.0,
            605.0,
            18.0,
            WHITE,
        );
    }
    fn draw_stats(&self) {
        draw_rectangle(668.0, 116.0, 488.0, 532.0, PANEL);
        let snapshot = &self.snapshots[self.selected];
        let s = &snapshot.stats;
        draw_text(
            format!("GENERATION {} / {}", s.generation, self.limit),
            688.0,
            148.0,
            26.0,
            WHITE,
        );
        draw_text(
            format!(
                "Practice levels 1-{} | {} starts per pilot",
                self.max_level, s.cases
            ),
            688.0,
            175.0,
            18.0,
            MUTED,
        );
        draw_text(
            format!("Champion: {} / {} landings", s.landings, s.cases),
            688.0,
            214.0,
            25.0,
            MINT,
        );
        draw_text(
            format!("Mean reward: {:.2}", s.reward),
            688.0,
            244.0,
            23.0,
            WHITE,
        );
        if self.training {
            let count = self.trainer.evaluated_candidates();
            draw_text(
                format!("Trying candidate {count} / {POPULATION}..."),
                688.0,
                276.0,
                19.0,
                GOLD,
            );
            draw_rectangle(688.0, 286.0, 444.0, 5.0, BG);
            draw_rectangle(
                688.0,
                286.0,
                444.0 * count as f32 / POPULATION as f32,
                5.0,
                GOLD,
            );
        } else {
            let text = if s.generation == 0 {
                "Before any learning: no engine or steering.".into()
            } else {
                format!(
                    "Candidate attempts landed: {:.1}%",
                    100.0 * s.candidate_landing_rate
                )
            };
            draw_text(&text, 688.0, 276.0, 19.0, GOLD);
        }
        draw_text("LANDING RATE ON PRACTICE CASES", 688.0, 321.0, 18.0, MUTED);
        self.chart(Rect::new(702.0, 337.0, 418.0, 88.0), false);
        draw_text("Champion", 688.0, 450.0, 17.0, MINT);
        draw_text("All candidates", 804.0, 450.0, 17.0, GOLD);
        draw_text("0 to 100%", 1022.0, 450.0, 17.0, MUTED);
        draw_text(
            "CHAMPION REWARD (HIGHER IS BETTER)",
            688.0,
            482.0,
            18.0,
            MUTED,
        );
        self.chart(Rect::new(702.0, 496.0, 418.0, 80.0), true);
        draw_text(
            "Horizontal axis: completed generations",
            688.0,
            600.0,
            17.0,
            MUTED,
        );
        draw_text(
            "These are practice scores, not a held-out exam.",
            688.0,
            627.0,
            18.0,
            MUTED,
        );
    }
    fn chart(&self, rect: Rect, reward: bool) {
        draw_rectangle(rect.x, rect.y, rect.w, rect.h, BG);
        let max_gen = self.snapshots.len().saturating_sub(1).max(10) as f32;
        let low = if reward {
            self.snapshots
                .iter()
                .map(|s| s.stats.reward)
                .fold(0.0, f32::min)
                - 1.0
        } else {
            0.0
        };
        let high = if reward {
            self.snapshots
                .iter()
                .map(|s| s.stats.reward)
                .fold(0.0, f32::max)
                + 1.0
        } else {
            1.0
        };
        let point = |s: &GenerationStats, candidates: bool| {
            let value = if reward {
                s.reward
            } else if candidates {
                s.candidate_landing_rate
            } else {
                s.landings as f32 / s.cases as f32
            };
            vec2(
                rect.x + rect.w * s.generation as f32 / max_gen,
                rect.y + rect.h * (1.0 - (value - low) / (high - low)),
            )
        };
        for pair in self.snapshots.windows(2) {
            for candidates in [false, true] {
                if reward && candidates {
                    continue;
                }
                let a = point(&pair[0].stats, candidates);
                let b = point(&pair[1].stats, candidates);
                draw_line(
                    a.x,
                    a.y,
                    b.x,
                    b.y,
                    2.0,
                    if candidates { GOLD } else { MINT },
                );
            }
        }
        let p = point(&self.snapshots[self.selected].stats, false);
        draw_line(p.x, rect.y, p.x, rect.y + rect.h, 1.0, WHITE);
        draw_circle(p.x, p.y, 4.0, MINT);
        draw_text("0", rect.x, rect.y + rect.h + 15.0, 14.0, MUTED);
        draw_text(
            format!("{max_gen:.0}"),
            rect.x + rect.w - 22.0,
            rect.y + rect.h + 15.0,
            14.0,
            MUTED,
        );
    }
}
fn clicked(x: f32, width: f32) -> bool {
    is_mouse_button_pressed(MouseButton::Left)
        && Rect::new(x, 66.0, width, 32.0).contains(Vec2::from(mouse_position()))
}
fn fit_text(text: &str, x: f32, y: f32, width: f32, size: f32, color: Color) {
    let measured = measure_text(text, None, size as u16, 1.0).width;
    draw_text(
        text,
        x,
        y,
        (size * (width / measured.max(1.0)).min(1.0)).max(12.0),
        color,
    );
}
#[macroquad::main(config)]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let level: u32 = args
        .get(1)
        .map(|s| s.parse().expect("maximum level must be 1-3"))
        .unwrap_or(3);
    let generations: usize = args
        .get(2)
        .map(|s| s.parse().expect("generations must be positive"))
        .unwrap_or(100);
    assert!((1..=3).contains(&level) && generations > 0);
    let mut demo = Demo::new(level, generations);
    loop {
        demo.update(get_frame_time().min(0.1));
        demo.draw();
        next_frame().await;
    }
}
