# Archived external review experiments

These scripts were supplied with the external review on 2026-11-03 and copied
from its temporary scratch directory before it could disappear. They describe
the original experiments and are not part of CI. Some deliberately exercise
unsupported or non-convergent cases and may stop early after the new safeguards.

These scripts use the pre-redesign API (commit f711991). Run them only with
that historical checkout, not against the new public API. Current scientific
regressions are in `test/test-api.jl` and `test/test-kernels.jl`; the old
correctness test source remains in `test/legacy` for coverage auditing.
