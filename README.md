# ternary-shard-merge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Merge distributed ternary weight shards back together — with conflict resolution, all-reduce simulation, gradient aggregation, and statistics tracking.

## Overview

In distributed training of ternary neural networks (where weights are constrained to {-1, 0, +1}), model weights are often sharded across multiple workers or devices. After training or during inference, these shards must be merged back into a coherent weight vector. When shards overlap or disagree, robust conflict resolution strategies are essential.

`ternary-shard-merge` provides:

- **Deterministic ordered merging** — reconstruct weights from sequentially indexed shards
- **Majority-vote conflict resolution** — when shards overlap and disagree, the majority value wins (ties broken toward 0)
- **All-reduce simulation** — combine outputs from N workers via consensus voting
- **Gradient aggregation** — average ternary gradients from multiple workers with rounding
- **Merge statistics** — track conflicts, agreement rates, and value distributions

All operations are pure Rust with zero external dependencies.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-shard-merge = { git = "https://github.com/SuperInstance/ternary-shard-merge" }
```

## Quick Start

```rust
use ternary_shard_merge::*;

// Create ordered shards
let s1 = Shard::new(0, vec![1, 0, -1, 1]);
let s2 = Shard::new(1, vec![0, 0, -1, 1]);
let s3 = Shard::new(2, vec![-1, 0, 1, 1]);

// Simple ordered merge
let merged = merge_sorted(&[s1, s2, s3]);
assert_eq!(merged, vec![1, 0, -1, 1, 0, 0, -1, 1, -1, 0, 1, 1]);
```

## Conflict Resolution

When shards cover overlapping regions and disagree:

```rust
let s1 = Shard::new(0, vec![1, 0, -1, 1]);
let s2 = Shard::new(1, vec![1, -1, -1, 0]);
let s3 = Shard::new(2, vec![-1, 0, -1, 1]);

let (merged, stats) = merge_with_conflict_resolution(
    &[s1, s2, s3],
    4,                        // total output length
    &[(0, 4), (0, 4), (0, 4)] // each shard covers all positions
);

// Check statistics
println!("Conflicts: {} / {} ({:.1}%)",
    stats.conflicts, stats.total_values, stats.conflict_rate() * 100.0);
```

## All-Reduce Simulation

Simulate all-reduce across workers that each produce a full ternary vector:

```rust
let worker_outputs = vec![
    vec![1, -1, 0, 1],
    vec![1, 0, 0, -1],
    vec![-1, -1, 0, 1],
];

let consensus = all_reduce_sim(&worker_outputs, 3);
// Result: [1, -1, 0, 1] — majority vote at each position
```

## Gradient Aggregation

Average ternary gradients from multiple workers:

```rust
let g1 = vec![1, 1, 1, 0];
let g2 = vec![-1, 1, 0, 0];
let g3 = vec![-1, -1, 0, 0];
let g4 = vec![0, 1, -1, 0];

let aggregated = gradient_aggregation(&[g1, g2, g3, g4]);
// Averages: [-0.25, 0.5, 0.0, 0.0] → rounds to [0, 1, 0, 0]
```

## API Reference

### `Shard`

A shard of ternary weights with an index for ordering.

| Method | Description |
|--------|-------------|
| `Shard::new(index, weights)` | Create a shard with validation |
| `Shard::index` | Shard index for ordering |
| `Shard::weights` | The ternary weight values |

### `MergeStatistics`

Statistics from a merge operation with conflict analysis.

| Method | Description |
|--------|-------------|
| `conflict_rate()` | Fraction of positions with conflicts |
| `agreement_rate()` | Fraction of positions that were unanimous |

### Functions

| Function | Description |
|----------|-------------|
| `merge_sorted(shards)` | Merge ordered shards by concatenation |
| `merge_with_conflict_resolution(shards, len, coverage)` | Merge overlapping shards with majority vote |
| `all_reduce_sim(outputs, n)` | Simulate all-reduce via majority consensus |
| `gradient_aggregation(gradients)` | Average ternary gradients with rounding |
| `merge_statistics(merged, shard_count, workers)` | Compute stats on a merged result |

## Design Decisions

- **Tie-breaking toward 0**: When majority vote is tied, we bias toward 0 (pruning) rather than ±1, which is generally safer for model stability
- **No external dependencies**: Pure Rust, no allocators or system crates needed
- **Validation everywhere**: All inputs are validated — invalid ternary values, mismatched lengths, and gap-in-index assertions all panic with clear messages

## License

MIT
