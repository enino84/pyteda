# -*- coding: utf-8 -*-
"""
Experiments submodule: Scenario and Benchmark.

* `Scenario` — a frozen twin-experiment setup (truth, observations,
  initial ensemble) shared across compared methods.
* `Benchmark` — runs a grid of (scenario, method, run) cells and
  aggregates results.
"""

from .scenario import Scenario
from .benchmark import Benchmark, BenchmarkResults

__all__ = ["Scenario", "Benchmark", "BenchmarkResults"]
