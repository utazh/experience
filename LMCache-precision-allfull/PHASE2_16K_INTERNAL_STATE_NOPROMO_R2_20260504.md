# Phase2 16K No-Promotion Internal-State Replicate 2026-05-04

| run | harness mean_subsequent_ttft_ms | internal mean_subsequent_ttft_ms | delta % |
|---|---:|---:|---:|
| r1 harness-first | 315.822 | 359.832 | 13.935 |
| r2 internal-first | 333.844 | 382.461 | 14.563 |

## Notes

- r2 reverses the run order to separate implementation overhead from ordering noise.
- Match rate remained 1.0 and final state remained BASE_READY in both modes.
