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

Each module has a CLI entry point:

```bash
# Example: Run module 03 experiment
python -m modules.03_ml_tabular_foundations.src.main --config configs/baseline.yaml --seed 42
```

---

## 📚 Documentation

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
| **Tracking** | MLflow |
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
