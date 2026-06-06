//! # ternary-shard-merge
//!
//! Merge distributed ternary weight shards back together with conflict resolution,
//! all-reduce simulation, gradient aggregation, and statistics tracking.
//!
//! Ternary weights are values in {-1, 0, +1}. When distributed across workers or
//! devices, shards may overlap or conflict. This crate provides deterministic
//! strategies to reconstruct the original weights from those shards.

use std::collections::HashMap;

/// A ternary weight value: -1, 0, or +1.
pub type Ternary = i8;

/// A shard is an ordered list of ternary weights tagged with a shard index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shard {
    /// Monotonically increasing shard index for ordering.
    pub index: usize,
    /// The ternary weight values in this shard.
    pub weights: Vec<Ternary>,
}

impl Shard {
    /// Create a new shard with the given index and weights.
    pub fn new(index: usize, weights: Vec<Ternary>) -> Self {
        for &w in &weights {
            assert!(w == -1 || w == 0 || w == 1, "Ternary weights must be -1, 0, or +1, got {w}");
        }
        Self { index, weights }
    }
}

/// Statistics from a merge operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeStatistics {
    /// Total number of shards merged.
    pub shard_count: usize,
    /// Total number of weight values in the merged result.
    pub total_values: usize,
    /// Number of positions that had conflicts (disagreement across shards).
    pub conflicts: usize,
    /// Number of positions resolved by majority vote.
    pub majority_resolved: usize,
    /// Number of positions that were unanimous (no conflict).
    pub unanimous: usize,
    /// Distribution of final values: count of -1, 0, +1.
    pub value_distribution: [usize; 3],
    /// Number of workers that contributed data.
    pub worker_count: usize,
}

impl MergeStatistics {
    /// Compute the conflict rate as a fraction of total values.
    pub fn conflict_rate(&self) -> f64 {
        if self.total_values == 0 {
            0.0
        } else {
            self.conflicts as f64 / self.total_values as f64
        }
    }

    /// Fraction of positions that were unanimous.
    pub fn agreement_rate(&self) -> f64 {
        if self.total_values == 0 {
            1.0
        } else {
            self.unanimous as f64 / self.total_values as f64
        }
    }
}

/// Merge sorted (ordered) shards into a single weight vector.
///
/// Shards must have monotonically increasing indices. The result is the
/// concatenation of all shards in index order.
///
/// # Panics
///
/// Panics if shard indices are not strictly increasing starting from 0.
pub fn merge_sorted(shards: &[Shard]) -> Vec<Ternary> {
    if shards.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<&Shard> = shards.iter().collect();
    sorted.sort_by_key(|s| s.index);

    for (i, s) in sorted.iter().enumerate() {
        assert_eq!(
            s.index, i,
            "Expected shard index {} but found {}",
            i, s.index
        );
    }

    sorted.iter().flat_map(|s| s.weights.clone()).collect()
}

/// Merge shards where positions may overlap, resolving conflicts by majority vote.
///
/// Each shard may cover overlapping indices into the output. When multiple shards
/// disagree at a position, the majority value wins. Ties are broken toward 0.
///
/// # Arguments
///
/// * `shards` - The shards to merge.
/// * `total_len` - The expected length of the output vector.
/// * `coverage` - For each shard, a `(start, end)` range into the output vector.
///
/// # Panics
///
/// Panics if any coverage range is out of bounds for `total_len`.
pub fn merge_with_conflict_resolution(
    shards: &[Shard],
    total_len: usize,
    coverage: &[(usize, usize)],
) -> (Vec<Ternary>, MergeStatistics) {
    assert_eq!(shards.len(), coverage.len());

    let mut votes: Vec<HashMap<Ternary, usize>> = vec![HashMap::new(); total_len];
    let mut worker_set: HashMap<usize, usize> = HashMap::new();

    for (shard, &(start, end)) in shards.iter().zip(coverage.iter()) {
        assert!(start <= end, "Invalid coverage range: start ({start}) > end ({end})");
        assert!(end <= total_len, "Coverage end ({end}) exceeds total_len ({total_len})");
        let len = end - start;
        assert!(
            len == shard.weights.len(),
            "Shard {} has {} weights but coverage [{}, {}) has length {}",
            shard.index, shard.weights.len(), start, end, len
        );
        *worker_set.entry(shard.index).or_insert(0) += 1;

        for (i, &w) in shard.weights.iter().enumerate() {
            *votes[start + i].entry(w).or_insert(0) += 1;
        }
    }

    let mut result = Vec::with_capacity(total_len);
    let mut conflicts = 0;
    let mut majority_resolved = 0;
    let mut unanimous = 0;
    let mut dist = [0usize; 3]; // -1, 0, +1

    for vote_map in &votes {
        if vote_map.is_empty() {
            result.push(0);
            dist[1] += 1;
            unanimous += 1;
            continue;
        }

        if vote_map.len() == 1 {
            let (&val, _) = vote_map.iter().next().unwrap();
            result.push(val);
            match val {
                -1 => dist[0] += 1,
                0 => dist[1] += 1,
                1 => dist[2] += 1,
                _ => unreachable!(),
            }
            unanimous += 1;
        } else {
            conflicts += 1;
            // Majority vote; ties broken toward 0
            let mut best_val = 0i8;
            let mut best_count = 0usize;
            // Check 0 first for tie-breaking priority
            if let Some(&c) = vote_map.get(&0) {
                best_count = c;
            }
            for &val in &[-1i8, 1i8] {
                if let Some(&c) = vote_map.get(&val) {
                    if c > best_count {
                        best_count = c;
                        best_val = val;
                    }
                }
            }
            result.push(best_val);
            match best_val {
                -1 => dist[0] += 1,
                0 => dist[1] += 1,
                1 => dist[2] += 1,
                _ => unreachable!(),
            }
            majority_resolved += 1;
        }
    }

    let stats = MergeStatistics {
        shard_count: shards.len(),
        total_values: total_len,
        conflicts,
        majority_resolved,
        unanimous,
        value_distribution: dist,
        worker_count: worker_set.len(),
    };

    (result, stats)
}

/// Simulate all-reduce across N workers, each producing a ternary weight shard
/// covering the full vector.
///
/// Each worker's output is a full-length ternary vector. The all-reduce combines
/// them via majority vote at each position. The result should be identical to
/// what any single worker would produce if there are no disagreements, or the
/// consensus otherwise.
///
/// This is conceptually equivalent to calling `merge_with_conflict_resolution`
/// with full-coverage shards.
pub fn all_reduce_sim(worker_outputs: &[Vec<Ternary>], num_workers: usize) -> Vec<Ternary> {
    assert_eq!(
        worker_outputs.len(),
        num_workers,
        "Expected {num_workers} worker outputs, got {}",
        worker_outputs.len()
    );

    if worker_outputs.is_empty() {
        return Vec::new();
    }

    let len = worker_outputs[0].len();
    for (i, out) in worker_outputs.iter().enumerate() {
        assert_eq!(
            out.len(),
            len,
            "Worker {} output length {} doesn't match expected {}", i, out.len(), len
        );
    }

    let mut result = Vec::with_capacity(len);

    for pos in 0..len {
        let mut counts: HashMap<Ternary, usize> = HashMap::new();
        for output in worker_outputs {
            *counts.entry(output[pos]).or_insert(0) += 1;
        }

        let mut best_val = 0i8;
        let mut best_count = 0usize;
        if let Some(&c) = counts.get(&0) {
            best_count = c;
        }
        for &val in &[-1i8, 1i8] {
            if let Some(&c) = counts.get(&val) {
                if c > best_count {
                    best_count = c;
                    best_val = val;
                }
            }
        }
        result.push(best_val);
    }

    result
}

/// Aggregate ternary gradients from multiple workers by averaging.
///
/// Since ternary gradients are {-1, 0, +1}, the average is computed as a
/// floating-point value and then rounded back to the nearest ternary value.
/// Values in (-0.5, 0.5) round to 0, values >= 0.5 round to +1, and
/// values <= -0.5 round to -1.
///
/// # Panics
///
/// Panics if any gradient has a different length than the first.
pub fn gradient_aggregation(gradients: &[Vec<Ternary>]) -> Vec<Ternary> {
    if gradients.is_empty() {
        return Vec::new();
    }

    let len = gradients[0].len();
    for (i, g) in gradients.iter().enumerate() {
        assert_eq!(g.len(), len, "Gradient {} has length {}, expected {}", i, g.len(), len);
    }

    let n = gradients.len() as f64;
    let mut result = Vec::with_capacity(len);

    for pos in 0..len {
        let sum: f64 = gradients.iter().map(|g| g[pos] as f64).sum();
        let avg = sum / n;
        let rounded = if avg >= 0.5 {
            1
        } else if avg <= -0.5 {
            -1
        } else {
            0
        };
        result.push(rounded);
    }

    result
}

/// Compute merge statistics for a set of shards and the merged result.
///
/// This is a convenience function that computes stats without doing conflict
/// resolution — it simply analyzes the distribution of the already-merged result.
pub fn merge_statistics(merged: &[Ternary], shard_count: usize, worker_count: usize) -> MergeStatistics {
    let mut dist = [0usize; 3];
    for &v in merged {
        match v {
            -1 => dist[0] += 1,
            0 => dist[1] += 1,
            1 => dist[2] += 1,
            _ => panic!("Invalid ternary value: {v}"),
        }
    }

    MergeStatistics {
        shard_count,
        total_values: merged.len(),
        conflicts: 0,
        majority_resolved: 0,
        unanimous: merged.len(),
        value_distribution: dist,
        worker_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_sorted_reconstructs_original() {
        let original: Vec<Ternary> = vec![1, 0, -1, 1, 0, 0, -1, 1, -1, 0, 1, 1];
        let s1 = Shard::new(0, vec![1, 0, -1, 1]);
        let s2 = Shard::new(1, vec![0, 0, -1, 1]);
        let s3 = Shard::new(2, vec![-1, 0, 1, 1]);

        let merged = merge_sorted(&[s1, s2, s3]);
        assert_eq!(merged, original);
    }

    #[test]
    fn test_merge_sorted_empty() {
        let merged = merge_sorted(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_sorted_single_shard() {
        let s = Shard::new(0, vec![1, -1, 0]);
        let merged = merge_sorted(&[s]);
        assert_eq!(merged, vec![1, -1, 0]);
    }

    #[test]
    fn test_merge_sorted_out_of_order() {
        let s1 = Shard::new(0, vec![1, 0]);
        let s2 = Shard::new(1, vec![-1, 1]);
        // Pass in reverse order; merge_sorted should still work
        let merged = merge_sorted(&[s2, s1]);
        assert_eq!(merged, vec![1, 0, -1, 1]);
    }

    #[test]
    fn test_merge_sorted_gap_panics() {
        let s1 = Shard::new(0, vec![1]);
        let s2 = Shard::new(2, vec![-1]); // gap at index 1
        let result = std::panic::catch_unwind(|| merge_sorted(&[s1, s2]));
        assert!(result.is_err());
    }

    #[test]
    fn test_conflict_resolution_picks_majority() {
        // Three shards overlap on a 4-element vector
        let s1 = Shard::new(0, vec![1, 0, -1, 1]);
        let s2 = Shard::new(1, vec![1, -1, -1, 0]);
        let s3 = Shard::new(2, vec![-1, 0, -1, 1]);

        let coverage = [(0, 4), (0, 4), (0, 4)];
        let (merged, stats) = merge_with_conflict_resolution(&[s1, s2, s3], 4, &coverage);

        // Position 0: 1,1,-1 → majority=1
        assert_eq!(merged[0], 1);
        // Position 1: 0,-1,0 → two 0s, one -1 → majority=0
        assert_eq!(merged[1], 0);
        // Position 2: -1,-1,-1 → unanimous=-1
        assert_eq!(merged[2], -1);
        // Position 3: 1,0,1 → two 1s, one 0 → majority=1
        assert_eq!(merged[3], 1);

        assert_eq!(stats.conflicts, 3); // positions 0, 1, 3 all have disagreement
        assert_eq!(stats.unanimous, 1); // only position 2 is unanimous
        assert_eq!(stats.total_values, 4);
        assert_eq!(stats.shard_count, 3);
    }

    #[test]
    fn test_conflict_resolution_partial_overlap() {
        let s1 = Shard::new(0, vec![1, 1, 1]);
        let s2 = Shard::new(1, vec![0, 0]);
        // s1 covers [0,3), s2 covers [1,3)
        let coverage = [(0, 3), (1, 3)];
        let (merged, stats) = merge_with_conflict_resolution(&[s1, s2], 3, &coverage);

        // Position 0: only s1 → 1
        assert_eq!(merged[0], 1);
        // Position 1: s1=1, s2=0 → conflict, tie → 0
        assert_eq!(merged[1], 0);
        // Position 2: s1=1, s2=0 → conflict, tie → 0
        assert_eq!(merged[2], 0);

        assert_eq!(stats.conflicts, 2);
    }

    #[test]
    fn test_conflict_resolution_unanimous() {
        let s1 = Shard::new(0, vec![1, -1, 0]);
        let s2 = Shard::new(1, vec![1, -1, 0]);
        let coverage = [(0, 3), (0, 3)];
        let (merged, stats) = merge_with_conflict_resolution(&[s1, s2], 3, &coverage);

        assert_eq!(merged, vec![1, -1, 0]);
        assert_eq!(stats.conflicts, 0);
        assert_eq!(stats.unanimous, 3);
        assert_eq!(stats.value_distribution, [1, 1, 1]); // one -1, one 0, one +1
    }

    #[test]
    fn test_all_reduce_same_as_single_worker() {
        // All workers produce the same output
        let output = vec![1, 0, -1, 1, 0];
        let worker_outputs: Vec<Vec<Ternary>> = (0..4).map(|_| output.clone()).collect();
        let reduced = all_reduce_sim(&worker_outputs, 4);
        assert_eq!(reduced, output);
    }

    #[test]
    fn test_all_reduce_majority_vote() {
        let w1 = vec![1, -1, 0, 1];
        let w2 = vec![1, 0, 0, -1];
        let w3 = vec![-1, -1, 0, 1];
        let reduced = all_reduce_sim(&[w1, w2, w3], 3);

        // Position 0: 1,1,-1 → majority=1
        assert_eq!(reduced[0], 1);
        // Position 1: -1,0,-1 → majority=-1
        assert_eq!(reduced[1], -1);
        // Position 2: 0,0,0 → unanimous=0
        assert_eq!(reduced[2], 0);
        // Position 3: 1,-1,1 → majority=1
        assert_eq!(reduced[3], 1);
    }

    #[test]
    fn test_all_reduce_empty() {
        let reduced = all_reduce_sim(&[], 0);
        assert!(reduced.is_empty());
    }

    #[test]
    fn test_gradient_aggregation_correctness() {
        // 4 workers, all gradients identical → should return same
        let g = vec![1, 0, -1, 0];
        let grads: Vec<Vec<Ternary>> = (0..4).map(|_| g.clone()).collect();
        let aggregated = gradient_aggregation(&grads);
        assert_eq!(aggregated, g);
    }

    #[test]
    fn test_gradient_aggregation_averaging() {
        let g1 = vec![1, 1, 1, 0];
        let g2 = vec![-1, 1, 0, 0];
        let g3 = vec![-1, -1, 0, 0];
        let g4 = vec![0, 1, -1, 0];

        let aggregated = gradient_aggregation(&[g1, g2, g3, g4]);

        // Position 0: (1 + -1 + -1 + 0)/4 = -0.25 → 0
        assert_eq!(aggregated[0], 0);
        // Position 1: (1 + 1 + -1 + 1)/4 = 0.5 → 1
        assert_eq!(aggregated[1], 1);
        // Position 2: (1 + 0 + 0 + -1)/4 = 0.0 → 0
        assert_eq!(aggregated[2], 0);
        // Position 3: (0 + 0 + 0 + 0)/4 = 0.0 → 0
        assert_eq!(aggregated[3], 0);
    }

    #[test]
    fn test_gradient_aggregation_single_worker() {
        let g = vec![1, -1, 0];
        let aggregated = gradient_aggregation(&[g.clone()]);
        assert_eq!(aggregated, g);
    }

    #[test]
    fn test_gradient_aggregation_empty() {
        let aggregated = gradient_aggregation(&[]);
        assert!(aggregated.is_empty());
    }

    #[test]
    fn test_merge_statistics_distribution() {
        let merged = vec![1, 1, 1, 0, 0, -1];
        let stats = merge_statistics(&merged, 3, 2);
        assert_eq!(stats.value_distribution, [1, 2, 3]); // 1×(-1), 2×0, 3×(+1)
        assert_eq!(stats.total_values, 6);
        assert_eq!(stats.shard_count, 3);
        assert_eq!(stats.worker_count, 2);
    }

    #[test]
    fn test_merge_statistics_rates() {
        let stats = MergeStatistics {
            shard_count: 4,
            total_values: 100,
            conflicts: 10,
            majority_resolved: 10,
            unanimous: 90,
            value_distribution: [30, 40, 30],
            worker_count: 4,
        };
        assert!((stats.conflict_rate() - 0.1).abs() < 1e-9);
        assert!((stats.agreement_rate() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_merge_statistics_empty() {
        let stats = merge_statistics(&[], 0, 0);
        assert_eq!(stats.total_values, 0);
        assert_eq!(stats.conflict_rate(), 0.0);
        assert_eq!(stats.agreement_rate(), 1.0);
    }
}
