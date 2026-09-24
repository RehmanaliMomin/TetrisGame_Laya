5 games, 8,000 pieces each, safety mask **off**, seeds disjoint from training.

| policy | avg lines | best | topped out | agrees with teacher | p50 / p95 latency per piece |
|---|---|---|---|---|---|
| teacher (El-Tetris script, not a model) | 3198.2 | 3199 | never | 100% | 0.2 / 0.5 ms |
| **laya-tetris** | **3196.2** | **3199** | **never** | **79.5%** | 36.9 / 43.4 ms |
| laya-base (untuned) | 0.0 | 0 | after 25 pieces | 5.2% | 39.0 / 47.5 ms |
| random | 0.0 | 0 | after 26 pieces | 7.0% | – |

The model disagrees with the teacher on one piece in five and still loses 2 lines in 3,198: where it
differs, it has usually found an equally good placement rather than a worse one.
