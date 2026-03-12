# causal_mm
Time-series Double Machine Learning for Fuzzy Cognitive Maps (FCMs). Estimates directed edge strengths, bounds them to [-1, +1] for visualization, and writes everything back into a single `fcm_project.json` bundle.

## Why this exists
- Small, time-dependent datasets: forward-chaining DML handles temporal dependence and avoids look-ahead bias.
- Prior knowledge first: uses your drawn FCM as structural prior; estimates "how strong" each edge is instead of discovering a spaghetti graph.
- Stakeholder-friendly outputs: scaled weights in [-1, +1], optional uncertainty, and an adjacency matrix block ready for downstream tools.

## What it does
- Load one `fcm_project.json` that holds meta, model, timeseries, settings, estimates, and optional `results`.
- Per edge: build lagged controls, run forward-fold DML, compute `tau_raw`, then scale with `tanh(alpha_scale * tau_raw)` to [-1, +1].
- Optional block bootstrap: confidence intervals (`ci_low`, `ci_high`), standard errors, and sign-stability.
- Persist results back into the JSON (estimates + `results.adjacency_matrix`) and stamp `meta.weights_computed_at` / `meta.weights_method`.
- Refresh-only helper: `recompute_adjacency` rewrites just the stored adjacency matrix from existing estimates.
- Compare models: `scripts/compare_mental_models.py` reads the stored adjacency matrices and reports similarity + Generalized Distance Ratio (GDR).

## Single-file project format (input/output)
Top-level keys in `fcm_project.json`:
- `meta`: project metadata; will be updated with computation timestamp/method.
- `model`: `concepts` (integer IDs) and `edges` (`source`, `target`, optional `stakeholder_weight`).
- `timeseries`: `index` array; `data` keyed by stringified concept IDs; arrays aligned to `index`.
- `settings`: passthrough config (currently optional, preserved).
- `estimates`: filled after estimation; keys like `"src->tgt"`.
- `results` (optional): `adjacency_matrix` with `concept_ids`, `matrix`, `weight_type`.
Rules: concept IDs are integers everywhere; JSON keys use their string form; all timeseries arrays share length; `results.adjacency_matrix` order must match `concept_ids` so comparison scripts align correctly.

## Installation

### From PyPI (recommended)
```bash
pip install causal-mm
```

With optional econml backends:
```bash
pip install causal-mm[econml]
```

### From GitHub
```bash
pip install git+https://github.com/skp703/causal-mm.git
```

### From source (development)
```bash
git clone https://github.com/skp703/causal-mm.git
cd causal-mm
pip install -e .
```

### Using conda
```bash
conda env create -f environment.yml
conda activate causal-mm
pip install -e .
```

## CLI
```bash
causal-mm-run --input path/to/model.fcm_project.json --output path/to/output.fcm_project.json \
  --max-lag 3 --n-folds 3 --min-train-size 10 --alpha-scale 1.0 \
  --outcome-model ridge --treatment-model ridge \
  [--controls-selection all|connected] \
  [--bootstrap --n-bootstrap 200 --block-size 5 --bs-n-jobs 1 --random-state 123] \
  [--use-econml --econml-estimator linear_dml|causal_forest|ortho_forest]
```
Common variants:
- Smaller data, fewer controls: `--controls-selection connected`
- Uncertainty: add `--bootstrap --n-bootstrap 500 --block-size 5`
- Nonlinear nuisances: `--outcome-model random_forest --treatment-model random_forest`
- EconML backend: `--use-econml --econml-estimator causal_forest`

## Defaults and how changing them affects results
- `max_lag` (default 3): higher captures longer memory but increases features and variance; lower is safer for very short series. Must keep `min_train_size > max_lag`.
- `include_self_lags` (default True, Python-only): turning off removes autoregressive terms for the target; speeds up and reduces variance but risks omitting genuine inertia.
- `drop_initial_na` (default True, Python-only): keeps only rows with full lag history; set False to retain earliest rows at the cost of introducing NaNs you must handle downstream.
- `controls_selection` (default `all`): `all` guards against omitted confounders; `connected` keeps only parents + self lags, reducing variance for tiny samples but assumes your graph is correct.
- `outcome_model` / `treatment_model` (default `ridge`): `ridge` is fast and stable. `random_forest`/`gbm` capture nonlinearity but can overfit small data. `lasso` yields sparse linear models; `linear` is plain OLS.
- `n_folds` (default 3): more folds reduce bias but shrink train windows; with short series prefer 3. If `n_folds` too high relative to length you will drop too much data.
- `min_train_size` (default 10): enforce enough history before the first fold. Increase for more stable nuisance models; decrease only if data are extremely short.
- `alpha_scale` (default 1.0): larger values push `tanh` toward +/-1 faster (good for visualization saturation); smaller keeps weights more linear and comparable across runs.
- `random_state` (default 123): controls fold splits and bootstrap resampling; change for robustness checks or unset in Python to allow full randomness.
- `bootstrap` (default off): enable to get `tau_se`, `ci_low/high`, `sign_stability`. Adds ~`n_bootstrap` times compute cost.
- `n_bootstrap` (default 200): more draws tighten CI at higher cost; for quick smoke tests use 50-100, for final reports 500+ if time allows.
- `block_size` (default 5): should match dependence length (e.g., 5 for annual data with medium persistence). Too small underestimates uncertainty; too large inflates variance.
- `bs_n_jobs` (default 1): increase for parallel bootstrap on multicore machines; beware memory use.
- `use_econml` (default False) and `econml_estimator` (default `linear_dml`): switch to leverage econml backends (`causal_forest`, `ortho_forest`) when you need built-in heterogeneity handling or alternative orthogonalization; compute cost rises notably for forests.

Changing options via CLI: pass the flag shown above. Changing Python-only options (`include_self_lags`, `drop_initial_na`, custom hyperparameters) requires constructing the config objects directly:
```python
from causal_mm.config import LagConfig, MLModelConfig, DMLConfig

lag_cfg = LagConfig(max_lag=2, include_self_lags=False, drop_initial_na=True)
dml_cfg = DMLConfig(
    lag_config=lag_cfg,
    outcome_model=MLModelConfig("random_forest", {"n_estimators": 300, "random_state": 42}),
    treatment_model=MLModelConfig("random_forest", {"n_estimators": 300, "random_state": 42}),
    n_folds=4,
    min_train_size=20,
    alpha_scale=0.8,
    controls_selection="connected",
)
```

## Python API
```python
from pathlib import Path
from causal_mm.config import LagConfig, MLModelConfig, DMLConfig, BootstrapConfig
from causal_mm.pipeline import run_estimation

input_path = Path("data/models/my_model.fcm_project.json")
output_path = Path("data/models/my_model_with_estimates.fcm_project.json")

lag_cfg = LagConfig(max_lag=3)
dml_cfg = DMLConfig(
    lag_config=lag_cfg,
    outcome_model=MLModelConfig("ridge", {"alpha": 1.0}),
    treatment_model=MLModelConfig("ridge", {"alpha": 1.0}),
    n_folds=3,
    min_train_size=10,
    alpha_scale=1.0,
)
bs_cfg = BootstrapConfig(n_bootstrap=200, block_size=5, random_state=123, n_jobs=1)

run_estimation(
    input_path=input_path,
    output_path=output_path,
    dml_config=dml_cfg,
    bootstrap_config=bs_cfg,
)
```

## How the estimator works (and why)
1) Lag all concepts up to `max_lag`; drop initial rows if requested to avoid NA leakage.
2) Select controls:
   - `all` (default) keeps lags of every concept (more robust to omitted variables).
   - `connected` keeps only parent + self lags (lower variance on tiny datasets).
3) Forward-chaining cross-fitting (`n_folds`, `min_train_size`) prevents peeking into the future.
4) Fit outcome and treatment models per fold; residualize; accumulate `tau_raw = sum(Y_tilde * T_tilde) / sum(T_tilde^2)`.
5) Scale with `tanh(alpha_scale * tau_raw)` so weights stay in [-1, +1] for FCM visualization.
6) Uncertainty (optional): block bootstrap resamples contiguous blocks to preserve time dependence; reports `tau_se`, percentile `ci_low/high`, and `sign_stability`.
7) Metadata stamped: `lag_used`, `n_obs`, `computed_at`, `method` (`dml` or `bootstrap-dml`).

## Outputs written to the project file
- `estimates["src->tgt"]`: `tau_raw`, `tau_se`, `ci_low`, `ci_high`, `sign_stability`, `scaled_weight`, `n_obs`, `lag_used`, `status`, `error_message`, `computed_at`, `method`.
- `meta.weights_computed_at` (UTC ISO8601) and `meta.weights_method`.
- `results.adjacency_matrix`: `concept_ids` (order) + `matrix` of `scaled_weight` values (`weight_type` set to `scaled_weight`).

## Comparing multiple mental models
```
python scripts/compare_mental_models.py data/models/a.json data/models/b.json
python scripts/compare_mental_models.py data/models/*.json --format csv --output temp/compare.csv
python scripts/compare_mental_models.py data/models/*.json --format json --precision 6
```
Reads each file's `results.adjacency_matrix`, aligns by concept ID, and reports similarity and GDR. Use CSV/JSON for dashboards.

## Metrics and diagnostics available in code
- `generalized_distance_ratio` (modified GDR) with tunable penalties.
- `graph_complexity_metrics`: N, edge count, density, hierarchy index.
- `concept_centrality_metrics`: in-degree, out-degree, total centrality.
- `score_models` / `best_model`: time-series CV for trying multiple ML models.
- `compare_weights`, `weight_distance`: adjacency-level comparisons.

## Simulation
`simulation.py` builds an adjacency matrix from `scaled_weight` and simulates FCM dynamics using `tanh`, `logistic`, or `identity` activations.

## Data requirements and pitfalls
- Timeseries arrays must align; concept IDs must match between `model.edges` and `timeseries.data` (stringified).
- `min_train_size` must exceed `max_lag` so folds have usable history.
- Block bootstrap is recommended whenever you care about uncertainty in time-dependent data.
- Cycles are fine: time lags break simultaneity (A_t depends on B_{t-1}, B_t on A_{t-1}).

## Interactive GUI

The `gui/` directory contains a Streamlit-based visualizer for inspecting, analyzing, and creating `fcm_project.json` files in the browser.

### Install GUI dependencies
```bash
pip install causal-mm[gui]
# or
pip install -r gui/requirements.txt
```

### Launch
```bash
streamlit run gui/app.py
```

Opens at **http://localhost:8501** with pages for:

| Page | What it shows |
|------|---------------|
| 🕸️ **Network** | Interactive directed graph (colored nodes, weighted edges, adjacency heatmap) |
| ⚖️ **Weights** | Edge-weight table, forest plot with 95% CI, sign-stability bars |
| 📈 **Time Series** | Multi-line Plotly charts, z-score toggle, correlation matrix |
| 📊 **Metrics** | Graph complexity KPIs, concept centrality, node-role distribution |
| ✏️ **Editor** | Create/edit concepts, edges, import CSV time series, export JSON |

See [`gui/README.md`](gui/README.md) for full documentation.

## Repository layout
- `src/causal_mm/`: core package (config, data, fcm, dml, bootstrap, metrics, io, pipeline, simulation).
- `gui/`: Streamlit interactive visualizer (network graph, weights, time series, metrics, editor).
- `scripts/`: utilities (e.g., `compare_mental_models.py`).
- `tests/`: unit tests.
- Packaging: `pyproject.toml`; console entry point `causal-mm-run`.

## FAQ -- Why FCM instead of full causal discovery?
- Data efficiency: testing every possible edge on short time-series yields noise; using the FCM as prior constrains the search.
- Regularization through expertise: arrows you drew act as structural prior; estimator asks "how strong," not "does it exist."
- Trust: outputs stay interpretable and aligned with stakeholder mental models.

## Tests
```bash
pip install -e .
pytest
```
