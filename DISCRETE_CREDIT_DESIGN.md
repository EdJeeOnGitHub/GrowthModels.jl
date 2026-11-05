# Discrete-time credit market with committed capital

5 November 2026. A design for adding borrowing and lending to the native
discrete-time models, with capital chosen before the next productivity shock.
Implementation status is recorded at the end.

## Why discrete time, and why committed capital

The continuous-time credit market (`CreditModel`, PR #16) rents capital each
instant after productivity is known. That suits slides about traps, but not the
production-function audit in `aggregate-transfers/sims/mainline`, which now uses
the native discrete models. That audit relies on the paper's timing: assets
`k_t` are set before the shock `z_t`, and output is `y_t = η z_t f(k_t)`.

With rented capital, `k_t` would respond to `z_t` and the timing assumption would
fail by construction. With committed capital, a loan-financed asset such as a cow
is bought one period and exposed to the next period's shock, so `k_t` stays
predetermined and the production equation holds in survey assets.

## Borrowing constraint

Use Kiyotaki and Moore's (1997) constraint, equation (3), `R b_t ≤ q_{t+1} k_t`:
repayment cannot exceed the resale value of the capital next period. Here:

```
(1+r) b' ≤ θ (1−δ) k'
```

`θ ≤ 1` is the share of resale value a lender can recover.

| Paper | Constraint | Capital timing | Why debt is safe |
|---|---|---|---|
| Buera and Shin (2013) | `k ≤ λa` | Rented within the period | Borrower can abscond with `1/λ` of rented capital |
| Moll (2014) | `k ≤ λa` | Rented ("solely for simplicity") | Same absconding story |
| Midrigan and Xu (2014) | `B_t ≤ λ(next period's revenue + (1−δ)K_t)` | Owned, chosen after current productivity is known | Next period's assets are known |
| Kiyotaki and Moore (1997) | `R b_t ≤ q_{t+1} k_t` | Owned, committed before production | Output is not pledgeable: the farmer's human capital is inalienable |

Kiyotaki and Moore fit this model best. Output depends on the household's own
work and ability, which a lender cannot seize; the asset can be seized. Because
resale value is `(1−δ)k'`, with a fixed price of capital and deterministic
depreciation, repayment is always feasible and the model needs no default.
Buera (2009), the paper the deck compares against, could not be accessed to check
its constraint.

The constraint is equivalent to `k' ≤ λ s`, where `s` is own funds and

```
λ = (1+r) / (1+r − θ(1−δ))
```

At `θ = 1`, `r = 0.03`, `δ = 0.08`, leverage is about 9, so `θ` should be
disciplined by data rather than set to full recovery.

## Model

State is cash on hand after production and repaying last period's debt, plus the
current productivity `z` and fixed ability `η`:

```
x  = y(k, z, η) + (1−δ)k − (1+r)b
```

Each period the household chooses:

- **Own funds** `s ∈ [0, x]`, so consumption is `c = x − s`.
- **Capital** `k' ∈ [0, λs]`, with debt `b' = k' − s`, before `z'` is known.

Next period's cash on hand is

```
x' = y(k', z', η) − (r+δ)k' + (1+r)s
```

Properties:

- **No default:** `(1+r)b' ≤ θ(1−δ)k'` and `y ≥ 0` imply `x' ≥ 0`.
- **Timing:** `k'` is committed before `z'`; survey assets `k_t` are predetermined.
- **Lending:** `k' < s` lends the difference at `r`. Unsecured borrowing is ruled out
  (`s ≥ 0`).
- **Nesting:** `θ = 0` with a low enough rate gives `k' = s`, the existing model.
- **Singularity:** `x' = 0` is possible at `s = k' = 0`. As in the existing discrete
  models, zero resources is singular for log utility and risk aversion of one or
  more, so the grid must start above zero or the domain must exclude it.

## Solution: two stages

1. **Allocation.** For each `s` and current `z`, choose capital:

   ```
   G(s, z) = max over k' ∈ [0, λs] of  E[ V(y(k',z',η) − (r+δ)k' + (1+r)s, z') | z ]
   ```

   Capital is risky, so this is not a static choice as in continuous time. Search the
   whole interval on a `k'` grid so the Skiba kink and policy jumps are found.

2. **Saving.** Choose `s` given `G`:

   ```
   V(x, z) = max over s ∈ [0, x] of  u(x − s) + β G(s, z)
   ```

   This is the existing one-dimensional problem, so `bellman_choice` (global search
   over piecewise-linear continuation intervals) can be reused with `G` on an `s` grid.

## Work plan

### GrowthModels (estimate: one and a half to two and a half days)

1. **Model type.** `DiscreteCreditModel(base; θ, r)` wrapping `DiscreteSkibaModel`
   or `DiscreteRamseyCassKoopmansModel`, with cash on hand as the first grid axis.
2. **Two-stage Bellman solver.** Compute `G` on an `(s, k')` grid per iteration, then
   the saving stage. Value iteration first, then modified policy iteration.
3. **Transition matrix and simulation.** `x'` depends on `z'`, so interpolate cash on
   hand separately for each next shock. `simulate` records `k`, `b`, `x`, `y` and `c`.
4. **Market clearing.** `∫ b' dμ = 0`, with stationary and finite-horizon options as
   in continuous time. Share the scan and bisection code.
5. **Tests.**
   - `θ = 0` with a low rate reproduces the existing discrete model.
   - Cash on hand is never negative.
   - The capital choice matches brute force.
   - Budget identities hold along simulated paths.
   - Bellman residuals converge.
   - Results are stable under grid refinement.

### aggregate-transfers (after GrowthModels)

6. **Calibration.** Recalibrate to the four targets (median assets, income/assets,
   log-asset IQR, transfer/assets), plus debt targets such as the share with zero
   debt and the debt-to-asset ratio.
7. **Audit.** Add the credit case to the discrete population-moment audit, using
   gross `k` as survey assets. Test the production moments, and check whether the
   consumption and asset rule `g(k)` misses debt: with credit, behaviour depends on
   net worth, not gross assets alone.

## Risks

- **Cost.** The allocation stage evaluates expected values on an `(s, k', z')` grid
  each iteration, far more work than the current solver. It should be manageable at
  audit sizes, but has not been measured.
- **Metastability.** The audit relies on stationary starts. As in continuous time,
  slow escape from the trap can make the stationary distribution uncomputable; a
  finite horizon is the fallback.
- **Calibrating θ.** Full recovery gives large leverage; θ must come from data.

## Sources

- Kiyotaki and Moore (1997), "Credit Cycles", JPE, equation (3).
- Midrigan and Xu (2014), NBER w15647, Section II.A.
- Buera and Shin (2013), NBER w16400, collateral constraint and Appendix B.
- Moll (2014), AER, equation (4).

## Decisions (5 November 2026)

- **Rate:** fixed first, to get the solver right, then market clearing in the same
  piece of work.
- **Lending:** households can both borrow and lend at `r`. This is needed for market
  clearing, since someone inside the model has to lend.
- **θ:** calibrated to debt data, adding targets such as the share with zero debt
  and the debt-to-asset ratio to the recalibration.
- **κ:** pledgeable at the same `θ` as other capital, with one recovery parameter.

## Implementation status

- **GrowthModels (steps 1–5): implemented** in `src/DiscreteCredit.jl`, with tests in
  `test/test-discrete-credit.jl` and a README section. `DiscreteCreditModel`, a two-stage
  solver, `credit_policy`, `simulate`, `Evolution`, `stationary`, `leverage` and
  `discrete_credit_equilibrium` (stationary or finite horizon). Two implementation
  details differ from the sketch above:
  - **Own-funds grid.** Own funds are chosen on zero followed by the cash grid, so a
    household with little cash can save almost nothing. Choosing zero leaves zero cash,
    which is clamped up to the grid's lowest point and counted in the diagnostics.
  - **Policy lottery.** The saving stage treats continuation value as linear between
    own-funds grid points. Policy evaluation, the transition matrix and market aggregates
    therefore use a lottery over neighbouring grid points, each with its own capital, so
    they match the Bellman step. Evaluating capital exactly at the chosen own funds
    instead left modified policy iteration stuck away from the Bellman fixed point.
- **Audit (step 7): implemented for benchmark parameters** in
  `aggregate-transfers/sims/mainline/audit-discrete-credit-moments.jl`, documented in
  `DISCRETE-CREDIT-MOMENTS.md`, with θ ∈ {0, 0.5, 0.9}.
- **Calibration (step 6): not done.** It needs the debt targets below.

## Still open

- **Debt targets:** which exact moments, and from which studies' debt measures.
- **Start date:** after the 9 November seminar.
