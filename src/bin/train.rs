//! BEGINNER WALKTHROUGH: teaching by trial and error.
//!
//! Run this program to change the model's weights. The game only reads weights.
//! A candidate is one possible pilot; an episode is one complete landing attempt.
//! A generation tests many candidates and uses the best ones to propose new pilots.
//! Candidate selection lives in training.rs, shared with training-demo.rs.
//! See docs/BEGINNERS.md for a presentation script and a worked numerical example.
use equinox_stall_2::{
    learning::{episode, Policy},
    training::Trainer,
};

// A test set is like an exam whose questions were not used in lessons.
// These scores are reported, never used to choose weights or update the search.
fn report(p: &Policy, max_level: u32) {
    for level in 1..=max_level {
        let mut successes = 0;
        let mut reward = 0.0;
        for seed in 10001..11001 {
            let e = episode(p, seed, level);
            successes += e.landed as usize;
            reward += e.reward;
        }
        println!(
            "Held-out level {level}: {successes}/1000 landed, mean reward {:.2}",
            reward / 1000.0
        );
    }
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let generations: usize = args
        .get(1)
        .map(|s| s.parse().expect("generations must be a positive integer"))
        .unwrap_or(300);
    let max_level: u32 = args
        .get(2)
        .map(|s| s.parse().expect("maximum level must be a positive integer"))
        .unwrap_or(3);
    assert!(generations > 0 && max_level > 0);
    assert!(
        args.iter()
            .skip(3)
            .all(|s| s == "--evaluate" || s == "--fresh"),
        "Supported flags: --evaluate, --fresh"
    );
    if args.iter().any(|s| s == "--evaluate") {
        let p = Policy::load("landing-model.txt").expect("load landing-model.txt");
        report(&p, max_level);
        return;
    }
    let initial = if args.iter().any(|s| s == "--fresh") {
        Policy::default()
    } else {
        Policy::load("landing-model.txt").unwrap_or_default()
    };
    let mut trainer = Trainer::new(initial, max_level);
    println!(
        "Training levels 1..={max_level}; baseline reward {:.2}",
        trainer.stats.reward
    );
    for _ in 0..generations {
        let stats = trainer.run_generation();
        trainer.best.save("landing-model.txt").expect("save model");
        if stats.generation % 10 == 1 || stats.generation == generations {
            println!(
                "Generation {:>4}: reward {:>7.2}, training landings {}/{}",
                stats.generation, stats.reward, stats.landings, stats.cases
            );
        }
    }
    report(&trainer.best, max_level);
    println!("Saved landing-model.txt. In game: M toggles pilot, L reloads model.");
}
