# 🧠 TEDA – Toolbox for Ensemble Data Assimilation

> **What's new in v2 (0.3.0)**
> - **Flexible observation operators** (linear selection, generic linear, nonlinear with Jacobian) and noise models (isotropic, heterogeneous diagonal, dense covariance).
> - **`Scenario`** object that freezes the entire random twin experiment under a single seed.
> - **`Benchmark`** runner for `M scenarios × |methods| × K runs` with bit-reproducible results.
> - **`pyteda.io`** module with **netCDF by default** (npz fallback) for portable storage of initial ensembles, truth trajectories, observations, and full scenarios. Granular helpers let you compute X0 once and reuse it across every scenario.
> - **New `QGModel`** — 1.5-layer quasi-geostrophic with pluggable time integrators (Euler, RK4, DP5, AB2/3/4, SSPRK3, Leapfrog+RA/RAW), `dirichlet`/`channel` boundary conditions, and a library of initial conditions (zero, fourier, vortex, dipole, rossby_wave, band_noise, restart). Vendored from the qg-integrators SoftwareX package under its original LICENSE.
> - **New `SWEModel`** — shallow-water equations on the sphere (Driscoll-Healy grid, T-LMAX truncation, vector-invariant form, exponential SH filter + polar sponge, Williamson TC2 + Rossby perturbations). Configurable state vector via `state_vars=['u','v','h']` (or any subset).
> - **Heterogeneous localization radius** — the `r` parameter of LEnKF/LETKF (and of `model.create_decorrelation_matrix`/`get_ngb`/`get_pre`) now accepts three forms:
>   - `int`/`float`: single radius for all components (legacy).
>   - `dict`: per variable block, e.g. `{'q': 2, 'psi': 4}` for QGModel, `{'u': 4, 'v': 4, 'h': 6}` for SWEModel.
>   - `ndarray`: per state component (length `n`).
> - **Calibration diagnostics** in `BenchmarkResults`: ensemble spread, spread/error ratio, CRPS (Continuous Ranked Probability Score), and Talagrand rank histograms. Enable with `Benchmark(..., store_diagnostics=True)`.
>
> See the example notebooks in [`examples/`](examples/) for full walkthroughs of Lorenz96, QGModel, and SWEModel.

We are thrilled to present TEDA, a cutting-edge Python toolbox designed to facilitate the teaching and learning of ensemble-based data assimilation (DA). TEDA caters to the needs of educators and learners alike, offering a comprehensive platform to explore the captivating world of meteorological anomalies, climate change, and DA methodologies.

TEDA stands out as an exceptional resource in the landscape of DA software, as it prioritizes the educational experience. While operational software tends to focus on practical applications, TEDA empowers undergraduate and graduate students by providing an intuitive platform to grasp the intricacies of ensemble-based DA concepts and foster enthusiasm for scientific exploration.

Equipped with a diverse range of features and functionalities, TEDA enriches the learning process by offering powerful visualization tools. Students can delve into error statistics, model errors, observational errors, error distributions, and the dynamic evolution of errors. These visualizations provide multiple perspectives on numerical results, enabling students to develop a deep understanding of ensemble-based DA principles.

TEDA offers numerous ensemble-based DA methods, allowing students to explore an extensive range of techniques. From the stochastic ensemble Kalman filter (EnKF) to dual EnKF formulations, EnKF via Cholesky decomposition, EnKF based on modified Cholesky decomposition, EnKF based on B-localization, and beyond, TEDA equips learners with a comprehensive toolkit to study and compare various DA methodologies.

TEDA's foundation is rooted in the Object-Oriented Programming (OOP) paradigm, enabling effortless integration of new techniques and models. This ensures that TEDA remains at the forefront of the rapidly evolving field of ensemble-based DA, allowing educators and researchers to incorporate the latest advancements into their teaching and experimentation.

Embark on a captivating journey of discovery and knowledge with TEDA as it unveils the secrets of data assimilation. Simulate various DA scenarios, experiment with different model configurations, and witness firsthand the transformative impact of ensemble-based DA methods on forecast accuracy and data analysis.

Unlock your true potential in the realm of data assimilation with TEDA. Whether you are an enthusiastic student or a dedicated educator, TEDA stands as your ultimate companion in unraveling the complexities of ensemble-based DA.

Keywords: Data Assimilation, Ensemble Kalman Filter, Education, Python.

### 🧪 Toy Models for Data Assimilation

To enhance the learning experience, **TEDA** is built around **toy models** that exhibit chaotic behavior under specific configurations. These models are ideal for experimenting with and validating various Data Assimilation (DA) techniques in a hands-on, controlled setting.

Currently, TEDA includes the **Lorenz 96 model** (40 variables), a well-known benchmark in DA research, and a **2D barotropic quasi-geostrophic (QG) model** with spectral dynamics and periodic boundary conditions.

> 📌 *Support for other classic chaotic systems like the Duffing equation and Lorenz 63 may be added in future versions.*

---

## 📦 Installation

Aquí tienes una versión mejor redactada y estructurada para la sección de instalación y uso rápido:

---

## 🚀 Installation and Quick Start

### ✅ From PyPI

To install the latest stable version:

```bash
pip install pyteda
```

### 🧪 From Source (for development)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/pyteda.git
cd pyteda
pip install -e .
```

### 🛠️ Recommended Setup: Virtual Environment

We suggest using a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Example notebooks

The [`examples/`](examples/) directory contains three end-to-end Jupyter notebooks demonstrating the full TEDA workflow on different models:

- **[`01_lorenz96_workflow.ipynb`](examples/01_lorenz96_workflow.ipynb)** — Lorenz96 with the 3-phase ensemble recipe, scenario persistence, three filters (EnKF, EnKF-MC, LETKF), full diagnostics, and bit-reproducibility check.
- **[`02_qg_workflow.ipynb`](examples/02_qg_workflow.ipynb)** — QGModel (1.5-layer quasi-geostrophic) with per-block localization (`r={'q': 2, 'psi': 4}`).
- **[`03_swe_workflow.ipynb`](examples/03_swe_workflow.ipynb)** — Shallow-water on the sphere with per-field localization (`r={'u': 3, 'v': 3, 'h': 5}`) and a configurable state vector.

To run them, clone the repository and open them in Jupyter:

```bash
git clone https://github.com/enino84/TEDA.git
cd TEDA/examples
jupyter notebook
```

Each notebook is self-contained: it builds the reference state `x0_ref`, the ensemble centre `xb`, and the initial ensemble `X_b` once, saves them to disk under `./scenarios/`, and reuses them across multiple scenarios. The first run takes a few minutes; subsequent runs are nearly instantaneous because the expensive artifacts are loaded from disk.

---

### Supported Methods

The `pyteda` package supports a wide range of ensemble-based data assimilation methods including EnKF, LETKF, ETKF, and shrinkage-based filters. All methods are accessible via the `AnalysisFactory`.

📄 See [Supported Methods](docs/index.md) for a full list and references.

---

## 🚀 Quickstart

Here's a minimal scenario-based example using the new v2 API:

```python
import numpy as np
from pyteda.models import Lorenz96
from pyteda.observation import LinearSelection, IsotropicDiagonal
from pyteda.experiments import Scenario, Benchmark

model = Lorenz96(n=40)
n = model.get_number_of_variables()

scen = Scenario.generate(
    model=model,
    operator_factory=lambda rng: LinearSelection(m=32, n_state=n, rng=rng),
    noise=IsotropicDiagonal(std=0.01, dim=32),
    ensemble_size=20,
    spinup_truth=2.0,                          # phase 1: synth_IC -> x0_ref
    pert_xb=0.5, spinup_xb=0.3,                # phase 2: x0_ref -> xb
    pert_ensemble=0.05, spinup_ensemble=0.1,   # phase 3: xb -> X_b
    obs_freq=0.1, end_time=4.0, seed=42,
)

results = Benchmark(
    scenarios=[scen],
    methods={
        'EnKF':  dict(method='enkf'),
        'LETKF': dict(method='letkf', r=2),
    },
    n_runs_per_method=3,
    store_diagnostics=True,
).run()

print(results.summary_table())
print(results.diagnostics_summary())
```

📈 For visualizations and full working examples, see the notebooks in [`examples/`](examples/).

---

## 🎓 Citation

If you use **TEDA** in your teaching or research, please cite:

**Nino-Ruiz, E.D., Valbuena, S.R. (2022)**
*TEDA: A Computational Toolbox for Teaching Ensemble Based Data Assimilation*.
[📖 ICCS 2022, Springer](https://doi.org/10.1007/978-3-031-08760-8_60)

```bibtex
@inproceedings{nino2022teda,
  title={TEDA: A Computational Toolbox for Teaching Ensemble Based Data Assimilation},
  author={Nino-Ruiz, Elias D. and Valbuena, Sebastian Racedo},
  booktitle={Computational Science – ICCS 2022},
  series={Lecture Notes in Computer Science},
  volume={13353},
  pages={787--801},
  year={2022},
  publisher={Springer, Cham},
  doi={10.1007/978-3-031-08760-8_60}
}
```

---

## 📚 More Resources

* 🧪 Explore [examples/](https://github.com/enino84/TEDA/tree/main/examples) to try different filters and models.
* More examples here [Google Colab](https://colab.research.google.com/drive/1Ts67iE19KtMuWd0CF8_GGGqE3wRQjC2G?usp=sharing)
* 📖 Full documentation available in the `docs/` folder.
* ➕ Want to add your own method? TEDA supports [custom DA methods via registry](docs/custom_methods.md).

---

[![PyPI version](https://badge.fury.io/py/pyteda.svg)](https://pypi.org/project/pyteda/)
[![Cite TEDA](https://img.shields.io/badge/Cite-TEDA-blue.svg)](https://doi.org/10.1007/978-3-031-08760-8_60)

---

## Developed by

* Elías D. Niño-Ruiz
* [https://enino84.github.io/](https://enino84.github.io/)
* elias.d.nino@gmail.com

