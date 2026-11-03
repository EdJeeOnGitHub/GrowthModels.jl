# Pre-redesign tests (historical)

These files use the API at commit f711991 and are retained as review evidence,
not as tests of the current package. Run them only in that historical checkout.
Their relative fixture paths may require adjusting when run separately.

Current tests are listed explicitly in `../runtests.jl`. Probability, residual,
scientific, AD, ownership, class-absorption, and observation contracts are tested
through the new API; low-level generator checks live in `../test-kernels.jl`.
`../test-baseline.jl` compares the new API against saved pre-redesign results on
identical explicit grids. The incomplete two-asset model is unsupported.

Coverage migration is still being audited in API_IMPLEMENTATION.md. Keeping
these files does not mean their individual assertions are all ported yet.
Use `../benchmark-api.jl` for the new/baseline performance comparison.
