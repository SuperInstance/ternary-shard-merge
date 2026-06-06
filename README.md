# ternary-shard-merge

Merge distributed ternary weight shards back together — with majority-vote conflict resolution, all-reduce simulation, gradient aggregation, and statistics tracking.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Why this exists

When you train a ternary neural network across multiple GPUs, each device holds a shard of the model's weights. During distributed training, these shards may overlap — different workers see different mini-batches and compute different ternary gradients for the same weights. When it's time to synchronize, you can't average {-1, 0, +1} like you'd average floats. You need a voting scheme.

This crate provides the merge strategies that make distributed ternary training coherent: majority vote for overlapping shards, consensus-based all-reduce, and float-averaging-then-rounding for gradient aggregation.

## The key insight

Ternary weights can't be averaged. The mean of −1 and +1 is 0 — a value neither worker agreed on. This is fundamentally different from FP32 distributed training where gradients average cleanly. The solution: vote. When 3 workers produce [+1, +1, −1] for the same weight, majority vote picks +1. When 2 workers produce [+1, −1] and disagree, the tie-breaking rule matters: this crate defaults to 0 (sparsity-preserving — a safer bet for model stability).

## Quick Start

```rust
use ternary_shard_merge::*;

// ── Ordered merge: concatenate non-overlapping shards ──
let s1 = Shard::new(0, vec![1, 0, -1, 1]);
let s2 = Shard::new(1, vec![0, 0, -1, 1]);
let s3 = Shard::new(2, vec![-1, 0, 1, 1]);
let merged = merge_sorted(&[s1, s2, s3]);
assert_eq!(merged, vec![1, 0, -1, 1, 0, 0, -1, 1, -1, 0, 1, 1]);

// ── Overlapping shards with conflict resolution ──
let s1 = Shard::new(0, vec![1, 0, -1, 1]);
let s2 = Shard::new(1, vec![1, -1, -1, 0]);
let s3 = Shard::new(2, vec![-1, 0, -1, 1]);
let (merged, stats) = merge_with_conflict_resolution(
    &[s1, s2, s3], 4, &[(0, 4), (0, 4), (0, 4)],
);
println!("Conflicts: {} / {} ({:.1}%)",
    stats.conflicts, stats.total_values, stats.conflict_rate() * 100.0);

// ── All-reduce: majority consensus across N workers ──
let worker_outputs = vec![
    vec![1, -1, 0, 1], vec![1, 0, 0, -1], vec![-1, -1, 0, 1],
];
let consensus = all_reduce_sim(&worker_outputs, 3);
// [1, -1, 0, 1] — majority at each position

// ── Gradient aggregation: average and round ──
let g1 = vec![1, 1, 1, 0];
let g2 = vec![-1, 1, 0, 0];
let g3 = vec![-1, -1, 0, 0];
let g4 = vec![0, 1, -1, 0];
let aggregated = gradient_aggregation(&[g1, g2, g3, g4]);
// [-0.25, 0.5, 0.0, 0.0] → rounds to [0, 1, 0, 0]
```

## Architecture

```
  Worker 0          Worker 1          Worker 2
  [1,0,-1,1]       [1,-1,-1,0]      [-1,0,-1,1]
       │                 │                 │
       └─────────┬───────┴─────────────────┘
                 ▼
    merge_with_conflict_resolution()
       ┌─────────────────────────────┐
       │  Per-position voting:        │
       │  pos 0: [1, 1, -1] → 1      │  majority
       │  pos 1: [0, -1, 0] → 0      │  majority (tie → 0)
       │  pos 2: [-1, -1, -1] → -1   │  unanimous
       │  pos 3: [1, 0, 1] → 1       │  majority
       └──────────┬──────────────────┘
                  ▼
         [1, 0, -1, 1] + MergeStatistics

  Alternative paths:
  ─ merge_sorted() → simple concatenation (non-overlapping shards)
  ─ all_reduce_sim() → N workers, full-coverage consensus
  ─ gradient_aggregation() → float average, round to nearest trit
```

## API Reference

### Shard

```rust
pub struct Shard {
    pub index: usize,        // monotonic ordering index
    pub weights: Vec<Ternary>, // {-1, 0, +1} values
}
Shard::new(index: usize, weights: Vec<Ternary>) -> Self  // validates ternary values
```

### MergeStatistics

```rust
pub struct MergeStatistics {
    pub shard_count: usize,
    pub total_values: usize,
    pub conflicts: usize,          // positions with disagreement
    pub majority_resolved: usize,  // conflicts resolved by vote
    pub unanimous: usize,          // positions with full agreement
    pub value_distribution: [usize; 3],  // count of {-1, 0, +1}
    pub worker_count: usize,
}
stats.conflict_rate() -> f64;    // conflicts / total
stats.agreement_rate() -> f64;   // unanimous / total
```

### Functions

| Signature | Description |
|-----------|-------------|
| `merge_sorted(shards: &[Shard]) -> Vec<Ternary>` | Concatenate in index order. Panics on gaps. |
| `merge_with_conflict_resolution(shards, total_len, coverage) -> (Vec<Ternary>, MergeStatistics)` | Majority vote for overlapping regions. Ties → 0. |
| `all_reduce_sim(outputs: &[Vec<Ternary>], n: usize) -> Vec<Ternary>` | N-worker consensus at each position. |
| `gradient_aggregation(gradients: &[Vec<Ternary>]) -> Vec<Ternary>` | Float average, round to nearest trit. |
| `merge_statistics(merged, shard_count, workers) -> MergeStatistics` | Analyze an already-merged result. |

## Real-world example

A fishing fleet has 8 boats, each running a ternary neural network to classify sonar returns. The fleet wants a shared model that benefits from all 8 boats' data. Each boat trains locally and periodically sends its weight shard to a central server.

The boats' weight shards overlap — all 8 cover the full model, but they've seen different data. At each synchronization round, the server runs `merge_with_conflict_resolution` with all 8 shards covering the same indices. The merge statistics tell the fleet:

- **Agreement rate**: 92% — most weights are stable across boats
- **Conflict hotspots**: attention head 3 has 40% conflict rate — the boats see very different fish species, and head 3 learned region-specific features
- **Tie-breaking**: 3% of positions are tied — these default to 0, effectively pruning uncertain connections

After 20 rounds, the agreement rate climbs to 98% and the shared model outperforms any single boat's model by 12%.

## Ecosystem connections

- **[`ternary-quantize`](https://github.com/SuperInstance/ternary-quantize)** — produces the ternary weights being sharded and merged
- **[`ternary-pipeline-parallel`](https://github.com/SuperInstance/ternary-pipeline-parallel)** — shards layers across stages; this crate merges them back
- **[`ternary-tensor-parallel`](https://github.com/SuperInstance/ternary-tensor-parallel)** — splits within layers; uses column/row parallelism instead of shard merging
- **[`ternary-transformer`](https://github.com/SuperInstance/ternary-transformer)** — the model being trained and merged

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `merge_sorted` | O(n·k) | n shards, k values each |
| `merge_with_conflict_resolution` | O(n·k + positions × shard_count) | Voting overhead |
| `all_reduce_sim` | O(n·k) | n workers, k positions |
| `gradient_aggregation` | O(n·k) | n workers, k positions |

All operations are single-pass. No external dependencies.

## Design decisions

- **Tie-breaking toward 0**: When majority vote is tied, we bias toward sparsity (0) rather than ±1. This is generally safer for model stability — a zeroed weight is a pruned connection, not an inverted one.
- **Validation everywhere**: Invalid ternary values, mismatched lengths, gap-in-index assertions all panic with clear messages.
- **Integer-only**: No floating-point except in gradient aggregation (which averages then rounds back).

## Open questions

- **Weighted voting**: Right now every shard gets one vote. Should shards from workers that processed more data get proportionally more weight?
- **Byzantine fault tolerance**: If one worker is compromised and sends adversarial weights, majority vote with 3 workers can't recover. What's the minimum worker count for BFT merge?
- **Communication efficiency**: Sending full weight shards is bandwidth-heavy. Could we send only the positions where a worker's weights changed since the last round?

## Testing

```bash
cargo test
```

16 tests: sorted merge reconstruction, empty/single-shard edge cases, out-of-order input, gap detection (panics), conflict resolution majority vote, partial overlap, unanimous agreement, all-reduce consistency (same input = same output), majority voting, gradient aggregation averaging and rounding, single worker passthrough, empty input, statistics distribution and rate calculations.

## License

MIT
