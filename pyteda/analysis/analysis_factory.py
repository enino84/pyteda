# analysis_factory.py

# 🔁 Importa todos los archivos de métodos oficiales para que se auto-registren
from . import (
    analysis_enkf,
    analysis_enkf_bloc,
    analysis_enkf_modified_cholesky,
    analysis_enkf_obs_modified_cholesky,
    analysis_enkf_obs_modified_cholesky_local,
    analysis_enkf_cholesky,
    analysis_enkf_naive,
    analysis_lenkf,
    analysis_ensrf,
    analysis_etkf,
    analysis_letkf,
    analysis_enkf_shrinkage_precision,
    analysis_enkf_lw,
    analysis_enkf_rblw,
)

from .registry import ANALYSIS_REGISTRY


# Some shrinkage-binv registry keys imply a specific α-criterion.
# When the user picks one of these, the factory injects the right
# ``criterion=`` kwarg automatically so they don't have to repeat it.
_BINV_CRITERION_BY_KEY = {
    "enkf-shrinkage-binv-mse":   "mse",
    "enkf-shrinkage-binv-stein": "stein",
    "enkf-shrinkage-binv-da":    "da",
    "enkf-shrinkage-precision":  "mse",   # legacy alias → default criterion
    # "enkf-shrinkage-binv" alone leaves criterion to the kwargs/default
}


class AnalysisFactory:
    def __init__(self, method='enkf', **kwargs):
        if method not in ANALYSIS_REGISTRY:
            raise ValueError(f"Invalid method name: '{method}'\n"
                             f"Available: {list(ANALYSIS_REGISTRY.keys())}")
        self.analysis_type = ANALYSIS_REGISTRY[method]
        # Auto-pick criterion from the registry key for shrinkage-binv
        # variants, unless the user supplied criterion explicitly.
        if method in _BINV_CRITERION_BY_KEY and "criterion" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["criterion"] = _BINV_CRITERION_BY_KEY[method]
        self.analysis_kwargs = kwargs

    def create_analysis(self):
        return self.analysis_type(**self.analysis_kwargs)
