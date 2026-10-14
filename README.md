# 🧠 TEDA – Toolbox for Ensemble Data Assimilation

> **What's new in 0.6.0: the models moved out**
> - **`QGModel` and `SWEModel` are now adapters, not implementations.** The physics used to be vendored inside pyteda; it now lives in its own published package, and pyteda only maps it onto the `Model` interface. Nothing was lost: the adapters keep the same state vector, variable blocks and localization helpers.
>   - `QGModel` → [`qg-python`](https://pypi.org/project/qg-python/) &nbsp;·&nbsp; `pip install pyteda[qg]`
>   - `SWEModel` → [`swesphere`](https://pypi.org/project/swesphere/) &nbsp;·&nbsp; `pip install pyteda[swe]`
> - **Both are optional extras.** `import pyteda` no longer needs `pyshtools`, and asking for a model you have not installed raises an error that names the command to fix it. `pyteda.models.available_models()` reports what you have.
> - **The shallow-water physics is corrected.** The vendored copy integrated `dh/dt = -h div(V)`, which omits the advection of `h`; `swesphere` uses the flux form `dh/dt = -div(h V)`. Measured on the unforced regime, the global mass now drifts 5e-5 over two days instead of about 1e-3. **Results produced with the vendored model must be regenerated.** The integrator also changes from explicit Euler to RK4.
> - **Flow regimes.** `SWEModel(preset=...)` selects one of the three documented `swesphere` regimes (`waves`, `one_jet`, `two_jets`), each with measured statistics. `preset=None` reproduces the old bare setup. `QGModel(preset=...)` likewise exposes `sakov_oke_stationary`, the regime used in assimilation testbeds.
> - **Two QG options were dropped** because the published `qg-python` does not implement them: the `channel` boundary condition and `raw_beta`. Passing them warns and continues. The scheme list is now `order1, order2, rk4, dp5, leapfrog` and the initial conditions are `zero, fourier, vortex, restart`.
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

### From PyPI

```bash
pip install pyteda            # core: filters, Lorenz96, scenarios, benchmarks
pip install pyteda[qg]        # + QGModel   (quasi-geostrophic, via qg-python)
pip install pyteda[swe]       # + SWEModel  (shallow water on the sphere, via swesphere)
pip install pyteda[all]       # everything
```

pyteda implements the assimilation methods and the toy model. The two
geophysical models are separate published packages that pyteda adapts to its
`Model` interface, so you install only the ones you need:

| Model | Backend package | Extra |
|---|---|---|
| `Lorenz96` | built in | — |
| `QGModel` | [`qg-python`](https://pypi.org/project/qg-python/) | `pyteda[qg]` |
| `SWEModel` | [`swesphere`](https://pypi.org/project/swesphere/) | `pyteda[swe]` |

Check what an installation can build:

```python
from pyteda.models import available_models
available_models()
# {'Lorenz96': True, 'QGModel': 'pip install pyteda[qg]', 'SWEModel': True}
```

### From source (for development)

```bash
git clone https://github.com/enino84/pyteda.git
cd pyteda
pip install -e ".[all,test]"
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

## Choosing what to assimilate

Two independent choices, for a model with more than one variable block
(`SWEModel`: `u`, `v`, `h`; `QGModel`: `q`, `psi`).

**1. Which variables are observed** — the observation operator decides this.
`pyteda.observation.grid_masks` places a network inside the blocks you name:

```python
from pyteda.models import SWEModel
from pyteda.observation.grid_masks import strided_indices, checkerboard_indices
from pyteda.observation import LinearSelection, IsotropicDiagonal

model = SWEModel(preset='two_jets', LMAX=8)

idx, m = strided_indices(model, spacing=4, variables=['h'])        # h only
idx, m = checkerboard_indices(model, variables=['u', 'v'])         # wind only
idx, m = strided_indices(model, spacing=4)                         # everything

op = LinearSelection(m=m, n_state=model.dim, indices=idx)
noise = IsotropicDiagonal(std=1.0, dim=m)
```

Keep the full state in the state vector and select with the operator. Do not
shrink the state itself with `state_vars=['h']`: the model still has to
integrate the wind, so the fields you drop become a fixed reference rather
than something the filter can correct.

**2. Whether an observed variable updates the others** — the `cross` flag on
the localized filters decides this, and the default is *no*:

```python
# univariate: observing h updates h and leaves u, v at their forecast value
dict(method='letkf', r=2)                      # cross=False is the default

# multivariate: the observation of h reaches u and v through the ensemble
# cross covariances
dict(method='letkf', r=2, cross=True)
```

`cross=False` localizes each block on its own, so the Cholesky factor and the
decorrelation matrix come out block diagonal by variable and the increment in
the unobserved blocks is *exactly* zero. `cross=True` puts the same grid cell
of the other blocks into each local domain or predecessor set. `enkf-b-loc`
also takes `cross_scale` to taper the cross-block entries.

The flag is available on `letkf`, `lenkf`, `enkf-b-loc`,
`enkf-modified-cholesky`, `enkf-shrinkage-precision`,
`enkf-obs-modified-cholesky` (with `b_via_cholesky=True`), its local variant
and `enkf-obs-woodbury-cholesky`. The unlocalized filters (`enkf`, `etkf`,
`ensrf`, `enkf-cholesky`, the Ledoit-Wolf family) use the full sample
covariance and are always multivariate, so they take no flag. On `Lorenz96`
the flag is accepted and does nothing: one block, nothing to cross to.

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

