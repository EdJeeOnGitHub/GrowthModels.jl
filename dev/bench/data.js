window.BENCHMARK_DATA = {
  "lastUpdate": 1793839653724,
  "repoUrl": "https://github.com/EdJeeOnGitHub/GrowthModels.jl",
  "entries": {
    "Julia benchmark result": [
      {
        "commit": {
          "author": {
            "email": "edjee96@gmail.com",
            "name": "EdJee",
            "username": "EdJeeOnGitHub"
          },
          "committer": {
            "email": "edjee96@gmail.com",
            "name": "EdJee",
            "username": "EdJeeOnGitHub"
          },
          "distinct": true,
          "id": "c7a4abdb8c09b52160b8b3e76dc2015890560dd8",
          "message": "ci: publish benchmark dashboard with automatic GitHub credentials",
          "timestamp": "2026-11-04T12:36:14-06:00",
          "tree_id": "cb2d3513c543dcca86e423599494596871a0ae4b",
          "url": "https://github.com/EdJeeOnGitHub/GrowthModels.jl/commit/c7a4abdb8c09b52160b8b3e76dc2015890560dd8"
        },
        "date": 1793817441079,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "skiba solve",
            "value": 0.005876626,
            "unit": "seconds"
          },
          {
            "name": "skiba trajectory",
            "value": 0.001207132,
            "unit": "seconds"
          },
          {
            "name": "ou solve",
            "value": 0.061443014,
            "unit": "seconds"
          },
          {
            "name": "ou trajectory",
            "value": 0.014483926,
            "unit": "seconds"
          },
          {
            "name": "ou warm_solve",
            "value": 0.00937902,
            "unit": "seconds"
          },
          {
            "name": "ou moments",
            "value": 0.014661652,
            "unit": "seconds"
          },
          {
            "name": "ou final_only",
            "value": 0.014491447,
            "unit": "seconds"
          },
          {
            "name": "inspection 1000_scalar_reads",
            "value": 0.000002922,
            "unit": "seconds"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "34009555+EdJeeOnGitHub@users.noreply.github.com",
            "name": "Ed Jee",
            "username": "EdJeeOnGitHub"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "05d735e8a3a2078a80adccfbd0cab910745f551f",
          "message": "Merge pull request #9 from EdJeeOnGitHub/fix/stationary-roundoff\n\nFix stationary failure from solver roundoff",
          "timestamp": "2026-11-04T17:46:46-07:00",
          "tree_id": "e9904f4640afa52a1ec1cffdc40a5a7424823559",
          "url": "https://github.com/EdJeeOnGitHub/GrowthModels.jl/commit/05d735e8a3a2078a80adccfbd0cab910745f551f"
        },
        "date": 1793839652912,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "skiba solve",
            "value": 0.00686843,
            "unit": "seconds"
          },
          {
            "name": "skiba trajectory",
            "value": 0.001351304,
            "unit": "seconds"
          },
          {
            "name": "ou solve",
            "value": 0.063569019,
            "unit": "seconds"
          },
          {
            "name": "ou trajectory",
            "value": 0.015159079,
            "unit": "seconds"
          },
          {
            "name": "ou warm_solve",
            "value": 0.009461576,
            "unit": "seconds"
          },
          {
            "name": "ou moments",
            "value": 0.015018862,
            "unit": "seconds"
          },
          {
            "name": "ou final_only",
            "value": 0.015013383,
            "unit": "seconds"
          },
          {
            "name": "inspection 1000_scalar_reads",
            "value": 0.000002464,
            "unit": "seconds"
          }
        ]
      }
    ]
  }
}