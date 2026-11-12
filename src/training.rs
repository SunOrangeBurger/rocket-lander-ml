//! BEGINNER WALKTHROUGH: the same teacher drives the CLI and live demo.
//!
//! One `advance_candidate` call tests one proposed weight array on every practice
//! case. After 80 candidates, select the best 12 and update the guesses. Splitting
//! this work into small calls keeps the visual demo responsive between candidates.
use crate::learning::{episode, Policy, Rng, WEIGHTS};
pub const POPULATION: usize = 80;
pub const ELITES: usize = 12;
pub const CASES_PER_LEVEL: u64 = 24;

/// Champion metrics and the exploration success rate for a completed generation.
#[derive(Clone, Debug)]
pub struct GenerationStats {
    pub generation: usize,
    pub reward: f32,
    pub landings: usize,
    pub cases: usize,
    pub candidate_landing_rate: f32,
}

// Every candidate sees the same cases. Include earlier levels to retain skills.
pub fn training_score(p: &Policy, max_level: u32) -> (f32, usize) {
    let mut reward = 0.0;
    let mut successes = 0;
    for level in 1..=max_level {
        for seed in 1..=CASES_PER_LEVEL {
            let e = episode(p, seed, level);
            reward += e.reward;
            successes += e.landed as usize;
        }
    }
    (
        reward / (CASES_PER_LEVEL * max_level as u64) as f32,
        successes,
    )
}

pub struct Trainer {
    pub best: Policy,
    pub stats: GenerationStats,
    max_level: u32,
    mean: Policy,
    sigma: [f32; WEIGHTS],
    rng: Rng,
    population: Vec<(f32, Policy, usize)>,
    candidate_landings: usize,
}
impl Trainer {
    pub fn new(initial: Policy, max_level: u32) -> Self {
        assert!(max_level > 0);
        let (reward, landings) = training_score(&initial, max_level);
        let cases = (CASES_PER_LEVEL * max_level as u64) as usize;
        Self {
            mean: initial.clone(),
            best: initial.clone(),
            max_level,
            stats: GenerationStats {
                generation: 0,
                reward,
                landings,
                cases,
                candidate_landing_rate: landings as f32 / cases as f32,
            },
            sigma: [0.8; WEIGHTS],
            rng: Rng(730201),
            population: vec![(reward, initial, landings)],
            candidate_landings: landings,
        }
    }
    pub fn evaluated_candidates(&self) -> usize {
        self.population.len()
    }
    pub fn advance_candidate(&mut self) -> Option<GenerationStats> {
        // STEP 1: propose a pilot by perturbing the mean weights.
        let mut p = self.mean.clone();
        for (i, w) in p.weights.iter_mut().enumerate() {
            *w += self.rng.normal() * self.sigma[i];
        }
        // STEP 2: grade the pilot on full flights, not just the displayed replay.
        let (score, landings) = training_score(&p, self.max_level);
        self.candidate_landings += landings;
        self.population.push((score, p, landings));
        if self.population.len() < POPULATION {
            return None;
        }
        self.population.sort_by(|a, b| b.0.total_cmp(&a.0));
        if self.population[0].0 > self.stats.reward {
            self.stats.reward = self.population[0].0;
            self.best = self.population[0].1.clone();
            self.stats.landings = self.population[0].2;
        }
        // STEP 3: move guesses toward the elites. Preserve some exploration.
        for (i, spread) in self.sigma.iter_mut().enumerate() {
            let avg = self.population[..ELITES]
                .iter()
                .map(|(_, p, _)| p.weights[i])
                .sum::<f32>()
                / ELITES as f32;
            let variance = self.population[..ELITES]
                .iter()
                .map(|(_, p, _)| (p.weights[i] - avg).powi(2))
                .sum::<f32>()
                / ELITES as f32;
            self.mean.weights[i] = 0.3 * self.mean.weights[i] + 0.7 * avg;
            *spread = (0.3 * *spread + 0.7 * variance.sqrt()).max(0.04);
        }
        self.stats.generation += 1;
        self.stats.candidate_landing_rate =
            self.candidate_landings as f32 / (POPULATION * self.stats.cases) as f32;
        self.population = vec![(self.stats.reward, self.best.clone(), self.stats.landings)];
        self.candidate_landings = self.stats.landings;
        Some(self.stats.clone())
    }
    pub fn run_generation(&mut self) -> GenerationStats {
        loop {
            if let Some(stats) = self.advance_candidate() {
                return stats;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incremental_training_matches_full_generation() {
        let mut slow = Trainer::new(Policy::default(), 1);
        let mut fast = Trainer::new(Policy::default(), 1);
        assert_eq!(slow.stats.landings, 0);
        for _ in 1..POPULATION - 1 {
            assert!(slow.advance_candidate().is_none());
        }
        let stats = slow.advance_candidate().unwrap();
        let other = fast.run_generation();
        assert_eq!(stats.generation, 1);
        assert_eq!(stats.reward, other.reward);
        assert_eq!(slow.best.weights, fast.best.weights);
        assert!(stats.reward >= -100.0);
        assert!((0.0..=1.0).contains(&stats.candidate_landing_rate));
    }

    #[test]
    fn zero_pilot_learns_from_failure() {
        let mut trainer = Trainer::new(Policy::default(), 3);
        let baseline = trainer.stats.reward;
        assert_eq!(trainer.stats.landings, 0);
        for _ in 0..12 {
            trainer.run_generation();
        }
        assert!(
            trainer.stats.landings >= 60,
            "Expected >=60/72 practice landings after 12 generations, got {}",
            trainer.stats.landings
        );
        assert!(trainer.stats.reward > baseline + 20.0);
        println!(
            "From zero: 0/72 -> {}/72 practice landings after 12 generations",
            trainer.stats.landings
        );
    }
}
