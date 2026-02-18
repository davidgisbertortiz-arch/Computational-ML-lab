# Computational ML Lab 🔬

**Physics/maths-driven ML engineering**: A comprehensive learning path from fundamentals to production-ready ML systems, with a strong emphasis on numerical methods, uncertainty quantification, simulation, and rigorous engineering practices.

---

## 🎯 Mission

This monorepo serves two goals:
1. **Learning Path**: Structured modules taking you from ML foundations to advanced topics, with physics/computational science flavor
2. **Portfolio Builder**: Production-quality projects demonstrating ML engineering best practices

## 🗺️ Learning Roadmap

Each module includes theory, from-scratch implementations, library-grade code, experiments, tests, and evaluation.

### Phase 1: Foundations
- **[00 - Repo Standards](modules/00_repo_standards/)** - Setup, tooling, workflows, reproducibility patterns
- **[01 - Numerical Toolbox](modules/01_numerical_toolbox/)** - Linear algebra, optimization, stability, conditioning
- **[02 - Statistical Inference & UQ](modules/02_stat_inference_uq/)** - Estimation, hypothesis testing, uncertainty quantification, bootstrapping

### Phase 2: ML Fundamentals  
- **[03 - Tabular ML Foundations](modules/03_ml_tabular_foundations/)** - Linear models, trees, ensembles, gradient boosting
- **[04 - Time Series & State Space](modules/04_time_series_state_space/)** - ARIMA, Kalman filters, HMMs, forecasting
- **[05 - Simulation & Monte Carlo](modules/05_simulation_monte_carlo/)** - MCMC, particle filters, sequential inference

### Phase 3: Deep Learning & Applications
- **[06 - Deep Learning Systems](modules/06_deep_learning_systems/)** - PyTorch, architectures, training dynamics, optimization
- **[07 - Physics-Informed ML](modules/07_physics_informed_ml/)** - PINNs, neural ODEs, scientific ML
- **[08 - NLP & Retrieval (RAG)](modules/08_nlp_retrieval_rag/)** - Embeddings, transformers, vector search, RAG systems
- **[09 - Computer Vision & Inverse Problems](modules/09_cv_inverse_problems/)** - CNNs, segmentation, image reconstruction

### Phase 4: Production & Integration
- **[10 - MLOps & Production](modules/10_mlops_production/)** - Tracking, APIs, Docker, monitoring, deployment
- **[11 - Capstone Projects](modules/11_capstones/)** - End-to-end applications combining multiple techniques

---

## 📊 Module Status

| Module | Status |
|--------|--------|
| **00 - Repo Standards** | Stable |
| **01 - Numerical Toolbox** | Stable |
| **02 - Statistical Inference & UQ** | Stable |
| **03 - ML Tabular Foundations** | Stable |
| **04 - Time Series & State Space** | Stable |
| **05 - Simulation & Monte Carlo** | Stable |
| **06 - Deep Learning Systems** | Stable |
| **07 - Physics-Informed ML** | Stable |
| **08 - NLP & Retrieval (RAG)** | In Progress |
| **09 - CV & Inverse Problems** | Planned |
| **10 - MLOps & Production** | In Progress |
| **11 - Capstones** | Planned |

---

## 🎬 Featured Demos

### 1. 📓 **Kalman Filter Tutorial** (Module 04)
Interactive notebook demonstrating state estimation with uncertainty propagation.

**What it shows**: Tracking a moving object with noisy measurements, posterior estimation, filtering vs smoothing.

```bash
# View notebook
jupyter notebook modules/04_time_series_state_space/notebooks/01_kalman_filter_intro.ipynb

# Run tracking demo
python -m modules.run run --module 04
```

**Key takeaway**: How Kalman filters optimally combine predictions and measurements under Gaussian noise.

---

### 2. 📊 **MCMC Sampler Benchmark** (Module 05)
Reproducible experiment comparing Metropolis-Hastings, HMC, and NUTS samplers.

**What it shows**: Convergence diagnostics (R-hat, ESS), mixing time analysis, multimodal posterior sampling.

```bash
# Run benchmark with config
python -m modules.run run --module 05 --seed 42

# View results
ls modules/05_simulation_monte_carlo/reports/
```

**Key takeaway**: NUTS achieves 10x better ESS than vanilla MH on challenging posteriors.

---

### 3. 🚀 **Physics-Informed Neural Network** (Module 07)
Solve the heat equation PDE with neural networks (no traditional discretization!).

**What it shows**: PINN training to satisfy PDE constraints, boundary conditions, residual loss visualization.

```bash
# Train PINN solver
python -m modules.run run --module 07

# See solution plots
open modules/07_physics_informed_ml/reports/heat_equation_solution.png
```

**Key takeaway**: Neural networks can learn PDE solutions by enforcing physics in the loss function.

---

## 🏆 Portfolio Projects

Production-ready implementations showcasing engineering rigor:

- **Kalman Filter Framework** - State estimation with uncertainty (module 04)
- **MCMC Inference Engine** - Bayesian inference from scratch (module 05)
- **Physics-Informed Neural PDE Solver** - PINN for heat equation (module 07)
- **Document RAG System** - Semantic search + LLM integration (module 08)
- **ML API with Monitoring** - FastAPI + MLflow + Docker (module 10)

Each project includes:
- ✅ Comprehensive tests (unit + integration)
- ✅ Type hints & documentation
- ✅ Reproducible experiments with seeds & logging
- ✅ CLI entry points & configs
- ✅ Performance benchmarks & error analysis

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/davidgisbertortiz-arch/Computational-ML-lab.git
cd Computational-ML-lab

# Install dependencies (using pip)
pip install -e ".[dev,docs]"

# Or using poetry
poetry install --with dev,docs

# Setup pre-commit hooks
pre-commit install

# Run tests
make test

# Run a specific module
make run-module MODULE=01_numerical_toolbox
```

### Running Experiments

Each module can be run using the universal module runner:

```bash
# List all available modules
python -m modules.run list

# Run a specific module
python -m modules.run run --module 00
python -m modules.run run --module 03_ml_tabular_foundations

# Run with specific seed
python -m modules.run demo --module 00 --seed 123
```

**Note on Numeric Module Names & Imports**: Python 3.12+ cannot parse imports like `from modules.03_ml_...` due to lexer limitations with numeric identifiers. Always use the provided import helper:

```python
# ✅ CORRECT - Use safe_import_from
from modules._import_helper import safe_import_from
MyClass = safe_import_from('03_ml_tabular_foundations.src.models', 'MyClass')

# ❌ WRONG - Will cause SyntaxError in Python 3.12+
from modules.03_ml_tabular_foundations.src.models import MyClass
```

See [`docs/getting-started/python312-imports.md`](docs/getting-started/python312-imports.md) for details.

---

## 📚 Documentation

- **Published site**: https://davidgisbertortiz-arch.github.io/Computational-ML-lab/
- **[Getting Started Guide](docs/getting-started/setup.md)** - Installation, environment setup
- **[Contributing](CONTRIBUTING.md)** - Development workflow, style guide, testing
- **[Module Structure](docs/getting-started/module-structure.md)** - Template and conventions

Full documentation: `mkdocs serve` → http://localhost:8000

---

## 🛠️ Technology Stack

| Category | Tools |
|----------|-------|
| **Core** | NumPy, Pandas, SciPy, scikit-learn |
| **DL** | PyTorch |
| **Boosting** | LightGBM, XGBoost |
| **Tracking** | MLflow (optional) |
| **API** | FastAPI, Uvicorn |
| **Testing** | pytest, pytest-cov |
| **Linting** | Ruff, Black, MyPy |
| **Docs** | MkDocs Material |
| **Orchestration** | Typer (CLI), Docker |

---

## 🧪 Quality Standards

Every module adheres to:
- **Reproducibility**: Deterministic seeds, versioned dependencies, logged configs
- **Testing**: Unit tests + integration tests + sanity checks
- **Documentation**: Inline comments for non-obvious math/physics, module READMEs
- **Type Safety**: Type hints for public APIs
- **CI/CD**: Automated linting + testing on push

---

## 📖 Philosophy

- **Physics/Math Identity**: Emphasize numerical stability, conditioning, error analysis, uncertainty propagation
- **From Scratch First**: Implement core algorithms manually before using libraries
- **Rigorous Evaluation**: Baselines, ablations, calibration plots, failure mode analysis
- **Engineering Discipline**: Clean code, tests, configs, CLI tools, tracking

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📬 Contact

**David Gisbert Ortiz** - [GitHub](https://github.com/davidgisbertortiz-arch)

---

**Note**: This is a learning repository. Modules will be developed incrementally. Check individual module READMEs for completion status.
