# Archived external review experiments

These scripts were supplied with the external review on 2026-11-03 and copied
from its temporary scratch directory before it could disappear. They describe
the original experiments and are not part of CI. Some deliberately exercise
unsupported or non-convergent cases and may stop early after the new safeguards.

The maintained assertions derived from them are in
`test/test-review-correctness.jl`. The regular suite includes that file.
Run an individual archived script from the repository root with
`julia --project=. test/review-reproductions/gamma.jl`, for example.
`suite.jl` additionally requires an absolute checkout path as its first argument.
