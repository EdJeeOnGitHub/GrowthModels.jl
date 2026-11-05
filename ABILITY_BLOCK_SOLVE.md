# Block HJB solve: plan and results

## Goal

Make `solve` cheaper as the ability grid grows, without new options for users and
without a fast path that could silently go wrong if models become more complicated.

## Where the cost was

Abilities do not interact in the HJB: the generator is `I ⊗ (Q ⊗ I)`, and ability
only enters through `resources`. The old solver already factored each ability
separately, so cost was linear in the number of abilities. The waste was elsewhere:

- every ability started from a cold guess and ran serially;
- all abilities shared one stopping rule and one failure, so one bad ability marked
  the whole solve as failed.

## Design

`src/BlockSolve.jl` owns all of this. `solve(model, grid)` keeps its signature.

| Decision | Why |
|---|---|
| Blocks come from the generator's stored pattern | Linked states merge automatically, so no model type or axis name is trusted |
| Every model takes the same path; a coupled model is one block | There is no second path to fall out of date |
| Each block owns its workspace and sees only its own values | Coupling outside the operator cannot be added without changing this interface |
| Model code runs once, on the calling task | Custom production does not need to be thread safe |
| Fixed midpoint-first warm-start tree | Results cannot depend on task scheduling |
| A caller's `initial` is never replaced | Warm starts only apply to the default guess, and a neighbour's unusable guess falls back to cold |
| `@spawn`, with no thread setting | Composes with outer threaded loops; `-t` already sets the thread count |
| Results copied back serially | Concurrent `BitArray` writes to neighbouring blocks would race |

### What this does not protect against

The generator check cannot see coupling that bypasses the operator. The block
interface makes that coupling hard to add by accident, not impossible. Anyone adding
per-iteration model code must also revisit the rule that model code stays off worker
threads. `test/test-block-threads.jl` fails if production runs off the calling task.

Separable but interleaved states merge into one block. That is correct but slower.

## Behaviour changes

- Ability models stop at a different iterate, inside tolerance. The baseline moved by
  at most 2.6e-9 in value and 1.6e-11 in consumption. Its `ability` case now has a
  documented 1e-8 tolerance; the fixture itself is unchanged, and other cases keep
  the strict gate.
- Default `fit.initial_value` now holds the values each block actually started from.
  Solving from it reproduces the same result exactly.
- `diagnostics.iterations` and `history` report the worst block.
  `diagnostics.blocks` is new and reports `iterations` and `stop_reason` per block.
- Some solves that failed now converge. With `γ=0.5` and abilities 0.8 to 1.2, the
  old solver failed because ability 0.8 fails from a cold guess. It now starts from a
  neighbour, and the old solver's loop accepted that answer in one iteration when
  checked. This is a side effect, not a guarantee.
- A single-column block now always uses the tridiagonal solver.

## Results

`main` (c7a4abd) against this branch. Julia 1.12.3, one process per case, BLAS set to
one thread. Medians of 3 to 7 runs, except 25 on the smallest grids. Other jobs were
active at times, so treat these as indicative. Reproduce with
`test/benchmark-blocks.jl`.

Single solve, in seconds, with the speedup against `main`:

| k×z×abilities | `main` | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| 50×5×3 | 0.0059 | 0.0056 (1.1×) | 0.0048 (1.2×) | 0.0046 (1.3×) |
| 200×21×3 | 0.237 | 0.145 (1.6×) | 0.127 (1.9×) | 0.143 (1.7×) |
| 200×21×10 | 0.731 | 0.405 (1.8×) | 0.285 (2.6×) | 0.284 (2.6×) |
| 200×21×20 | 1.504 | 0.836 (1.8×) | 0.471 (3.2×) | 0.416 (3.6×) |
| 200×21×40 | 2.859 | 1.503 (1.9×) | 0.777 (3.7×) | 0.747 (3.8×) |
| 500×41×10 | 4.679 | 2.613 (1.8×) | 1.588 (2.9×) | 1.570 (3.0×) |

Other workloads, in seconds:

| Workload | Threads | `main` | Branch | Speedup |
|---|---:|---:|---:|---:|
| 12 solves in an outer loop | 4 | 3.354 | 1.816 | 1.8× |
| 12 solves in an outer loop | 8 | 2.941 | 1.464 | 2.0× |
| 8 AD gradients in an outer loop | 4 | 0.237 | 0.169 | 1.4× |
| 8 AD gradients, serial | 1 | 0.744 | 0.468 | 1.6× |
| 12 solves, each warm from the last | 1 | 5.619 | 5.022 | 1.1× |
| 12 solves, each warm from the last | 4 | 5.619 (1 thread) | 2.781 | 2.0× |
| 4 outer tasks of 3 warm solves | 4 | 2.329 | 1.829 | 1.3× |
| Deterministic Skiba, one block | 1 | 0.0081 | 0.0084 | 4% slower |
| OU Skiba, one block | 1 | 0.0733 | 0.0774 | 6% slower |

Peak RSS, including about 500 MB for Julia itself:

| Case | `main` | Branch |
|---|---:|---:|
| 200×21×40 | 1132 MB | 773 MB (1 thread), 981 MB (4) |
| 500×41×10 | 1477 MB | 817 MB (1 thread), 1214 MB (4) |
| 200×21×3 | 574 MB | 581 MB (1 thread), 627 MB (4) |
| 12 solves in an outer loop | 2004 MB (8 threads) | 1355 MB (8 threads) |

### Reading the results

- Warm starts give most of the one-thread gain, and Julia defaults to one thread.
- Warm starts barely help when `γ<1`, where `Delta=1` makes iteration much slower to
  converge. Separate stopping alone gained about 1.0× to 1.05× in the prototype.
- Warm starts from the previous parameter gain much less, since neighbour starts do
  not apply. Outer-loop estimation with warm starts gained 1.3× here.
- Going from 4 to 8 threads adds little, and adds nothing with few abilities. This
  was not profiled, and other jobs were running.
- Models with one block pay 4 to 6% for copying into a block workspace.

## Tests

- `test/test-block-solve.jl`: block detection, warm-start order, workspace slicing,
  agreement with a forced single-block solve on four model types, a model whose
  generator really does couple abilities, diagnostics, and exact replay.
- `test/test-block-threads.jl`, run at 1 and 4 threads in subprocesses: identical
  results, solves and gradients inside `:dynamic` and `:static` outer loops, custom
  production staying on the calling task, and unwrapped errors in item order.

## Possible follow-ups

- Avoid the workspace copy when there is only one block.
- A `threaded=false` keyword, only if real workloads show nesting costs.
- A batch API that warm starts across both parameters and abilities.
- Profile thread scaling beyond 4 threads on an idle machine.
