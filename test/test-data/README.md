# Numerical comparison fixtures

`api-baseline.csv` was generated from GrowthModels commit f711991, after the
scientific correctness fixes and before the public API redesign. It contains
value, consumption, and final cell probabilities for six explicit-grid cases.
`test/integration/generate-baseline.jl` regenerates it using that commit's
environment. `test/test-baseline.jl` defines the matching new-API problems.

`diffusion-rbc-matlab-output.csv` is historical. Its MATLAB generator uses a
forward-only drift stencil and is not the monotone generator used now. The
historical comparison lives in `test/legacy`; current acceptance relies on
OU moment/drift tests and the explicit-grid corrected-solver baseline.
