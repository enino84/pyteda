#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prueba mínima de las métricas por-variable en pyteda (modelo Lorenz96).

Corre un benchmark pequeño con 2 filtros y diagnósticos activados, e
inspecciona:
  1. que cada celda traiga las series por-variable (error_*_by_var, etc.),
  2. que el DataFrame largo tenga la columna 'variable',
  3. que los CSV exportados incluyan las columnas/filas por-variable.

Lorenz96 tiene UN solo bloque de estado ('x'), así que el por-variable
coincide con el global. Para ver el efecto de magnitudes dispares hay que
usar un modelo multivariable (QG: q/psi, SWE: u/v/h) — el código es el
mismo, solo cambia cuántas variables aparecen.

Uso:  python test_per_variable.py
"""

import tempfile

import numpy as np
import pandas as pd

from pyteda.models import Lorenz96
from pyteda.observation.operators import LinearSelection
from pyteda.observation.noise import IsotropicDiagonal
from pyteda.experiments import Scenario, Benchmark


def build_scenario(seed: int) -> Scenario:
    """Un escenario Lorenz96 chico y rápido."""
    n_state = 20
    model = Lorenz96(n=n_state)

    # Verdad de referencia: una condición inicial propagada un rato (spin-up).
    x0 = model.get_initial_condition()
    x0_ref = model.propagate(x0, np.array([0.0, 2.0]))

    return Scenario.generate(
        model=model,
        operator_factory=lambda rng: LinearSelection(m=10, n_state=n_state, rng=rng),
        noise=IsotropicDiagonal(std=0.1, dim=10),
        ensemble_size=20,
        x0_ref=x0_ref,
        pert_xb=0.5, spinup_xb=0.5,
        pert_ensemble=0.5, spinup_ensemble=0.5,
        obs_freq=0.5, end_time=5.0,
        seed=seed,
    )


def main() -> None:
    print("=" * 60)
    print(" Benchmark de prueba — métricas por variable (Lorenz96)")
    print("=" * 60)

    scenarios = [build_scenario(s) for s in (1, 2)]

    bench = Benchmark(
        scenarios=scenarios,
        methods={
            "EnKF":  {"method": "enkf"},
            "LETKF": {"method": "letkf", "r": 2},
        },
        n_runs_per_method=2,
        inflation_factor=1.04,
        store_diagnostics=True,   # activa spread / CRPS / rank histogram
        verbose=False,
    )
    results = bench.run()

    # --- 1. Resumen por método (global, como siempre) -------------------
    print("\n[1] summary_table() — RMSE global por método:")
    print(results.summary_table().to_string(index=False))

    # --- 2. Diagnósticos de calibración (global) ------------------------
    print("\n[2] diagnostics_summary() — spread/CRPS global:")
    print(results.diagnostics_summary().to_string(index=False))

    # --- 3. Series por variable en cada celda ---------------------------
    row = results.rows[0]
    var_names = list(row["error_a_by_var"].keys())
    K = len(row["error_a"])
    print(f"\n[3] Variables detectadas (model.var_blocks): {var_names}")
    print(f"    Cada error_a_by_var[v] es una serie temporal de largo K={K}.")
    for v in var_names:
        serie = row["error_a_by_var"][v]
        print(f"      RMSE_a['{v}']  ->  {serie.shape}, "
              f"primeros 3 pasos = {np.round(serie[:3], 4)}")

    # --- 4. DataFrame largo con columna 'variable' ----------------------
    df = results.to_dataframe()
    print("\n[4] to_dataframe() columnas:", list(df.columns))
    print("    valores de 'variable':", sorted(df['variable'].unique()))

    # --- 5. Exportar CSVs y revisar columnas por variable ---------------
    out_dir = tempfile.mkdtemp(prefix="pyteda_test_")
    written = results.export_csv(out_dir, burn_in_frac=0.2)
    print(f"\n[5] CSVs escritos en: {out_dir}")
    for name, path in written.items():
        print(f"      {name:20s} -> {path.name}")

    summ = pd.read_csv(written["summary"])
    per_var_cols = [c for c in summ.columns if any(c.endswith(f"_{v}") for v in var_names)]
    print("\n    Columnas por-variable en summary.csv:")
    print("     ", per_var_cols)

    curves = pd.read_csv(written["error_curves"])
    print("\n    error_curves.csv tiene columna 'variable':",
          "variable" in curves.columns,
          "| valores:", sorted(curves["variable"].unique()))

    print("\nOK — todo corrió. (Con Lorenz96, 'x' == global por ser monovariable.)")


if __name__ == "__main__":
    main()