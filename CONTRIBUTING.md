# Contributing to Computational ML Lab

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Computational-ML-lab.git`
3. Install dependencies: `make setup`
4. Create a branch: `git checkout -b feature/my-feature`
5. Make changes and add tests
6. Run checks: `make lint && make test`
7. Commit with conventional commits: `git commit -m "feat: add new feature"`
8. Push and open a Pull Request

## Development Workflow

See the detailed [Contributing Guide](docs/contributing.md) in the documentation for:

- Code style guidelines (Ruff, Black, MyPy)
- Testing practices (pytest, coverage)
- Documentation standards (docstrings, MkDocs)
- Module development checklist
- Reproducibility requirements

## Key Commands

```bash
make help          # Show all available commands
make setup         # Initial setup
make test          # Run tests with coverage
make lint          # Check code quality
make format        # Auto-format code
make pre-commit    # Run pre-commit hooks
```

## Code Quality Standards

- **Type hints** on all public functions
- **Docstrings** (Google style) for public APIs
- **Tests** with >80% coverage
- **Reproducibility** with deterministic seeds
- **Linting** passes (Ruff + Black)

## Pull Request Guidelines

- Clear title and description
- Link related issues
- Add/update tests
- Update documentation
- Pass all CI checks

## Questions?

- 📖 Read the [full documentation](https://davidgisbertortiz-arch.github.io/Computational-ML-lab/)
- 🐛 Report bugs via [GitHub Issues](https://github.com/davidgisbertortiz-arch/Computational-ML-lab/issues)
- 💬 Ask questions in [Discussions](https://github.com/davidgisbertortiz-arch/Computational-ML-lab/discussions)

Thank you for contributing! 🚀
