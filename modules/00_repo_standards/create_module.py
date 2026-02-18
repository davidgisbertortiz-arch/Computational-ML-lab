"""Module template generator for creating new modules."""

import json
from pathlib import Path

import typer

app = typer.Typer(help="Generate a new module from template")


MODULE_README_TEMPLATE = """# Module {number}: {title}

**Status**: 📋 Planned

## 📚 What You'll Learn

- TODO: List key learning objectives
- TODO: Skills and concepts covered

## 🎯 Learning Objectives

TODO: Detailed learning objectives

## 📖 Theory

TODO: Minimal theory section (concise, not a textbook)

Key concepts:
- Concept 1
- Concept 2

## 🛠️ Implementation Plan

### Part 1: TODO
- [ ] Task 1
- [ ] Task 2

### Part 2: TODO
- [ ] Task 1
- [ ] Task 2

## 🧪 Experiments & Metrics

### Experiment 1: TODO
- **Goal**: What are we testing?
- **Metrics**: accuracy, loss, etc.
- **Baseline**: What do we compare against?

## ⚠️ Failure Modes

1. **Common pitfall 1**: Description and how to avoid
2. **Common pitfall 2**: Description and how to avoid

## ✅ Definition of Done

- [ ] README with theory and usage
- [ ] Core implementations in src/
- [ ] Tests with >80% coverage
- [ ] Configs for experiments
- [ ] Notebooks for exploration
- [ ] Experiments logged and reproducible
- [ ] Documentation complete

## 🔗 Resources

- Resource 1
- Resource 2

## 🚀 Next Steps

After completing this module:
1. Review outputs in reports/
2. Move to next module
"""

INIT_PY_TEMPLATE = '"""Module {number}: {title}."""\n'

MAIN_PY_TEMPLATE = """\"\"\"CLI entry point for Module {number}.\"\"\"

import typer
from pathlib import Path

app = typer.Typer(help=\"Module {number}: {title}\")


@app.command()
def train(
    config: Path = typer.Option(
        \"configs/experiment.yaml\",
        help=\"Path to config file\",
    ),
    seed: int = typer.Option(42, help=\"Random seed\"),
) -> None:
    \"\"\"Train the model.\"\"\"
    typer.echo(f\"Training with seed={{seed}}\")
    # TODO: Implement training logic


@app.command()
def evaluate(
    model_path: Path = typer.Option(..., help=\"Path to trained model\"),
) -> None:
    \"\"\"Evaluate the model.\"\"\"
    typer.echo(f\"Evaluating model from {{model_path}}\")
    # TODO: Implement evaluation logic


if __name__ == \"__main__\":
    app()
"""

TEST_TEMPLATE = """\"\"\"Tests for Module {number}.\"\"\"

import numpy as np
import pytest
from modules._import_helper import safe_import_from


class TestPlaceholder:
    \"\"\"Placeholder tests.\"\"\"

    def test_example(self):
        \"\"\"Example test.\"\"\"
        assert True

    def test_seeding(self):
        \"\"\"Test reproducibility.\"\"\"
        set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')
        set_seed(42)
        x1 = np.random.randn(10)

        set_seed(42)
        x2 = np.random.randn(10)

        assert np.array_equal(x1, x2)


# TODO: Add more tests
"""

CONFIG_TEMPLATE = """# Configuration for Module {number}

name: \"{module_name}_experiment\"
description: \"TODO: Describe experiment\"
seed: 42

# Output settings
output_dir: \"modules/{module_name}/reports\"
save_artifacts: true
log_level: \"INFO\"

# TODO: Add module-specific parameters
"""


@app.command()
def create(
    name: str = typer.Argument(
        ...,
        help="Module name (e.g., '05_simulation_monte_carlo' or 'simulation_monte_carlo')",
    ),
    description: str = typer.Option(
        "",
        "--description",
        "-d",
        help="Module description",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite if module exists",
    ),
) -> None:
    """
    Create a new module from template.

    Example:
        python modules/00_repo_standards/create_module.py create \
            05_simulation_monte_carlo \
            --description "Monte Carlo methods"
    """
    if not name.startswith("modules/"):
        if not any(name.startswith(f"{i:02d}_") for i in range(20)):
            typer.secho(
                "Module name should start with number (e.g., '05_module_name')",
                fg=typer.colors.YELLOW,
            )
        module_name = name
    else:
        module_name = name.replace("modules/", "")

    parts = module_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        number = parts[0]
        title = parts[1].replace("_", " ").title()
    else:
        number = "XX"
        title = module_name.replace("_", " ").title()

    if not description:
        description = title

    module_path = Path("modules") / module_name

    if module_path.exists() and not force:
        typer.secho(
            f"Module already exists: {module_path}",
            fg=typer.colors.RED,
        )
        typer.echo("Use --force to overwrite")
        raise typer.Exit(1)

    typer.echo(f"Creating module: {module_name}")
    typer.echo(f"  Number: {number}")
    typer.echo(f"  Title: {title}")
    typer.echo(f"  Description: {description}")
    typer.echo("")

    subdirs = ["src", "tests", "notebooks", "configs", "reports"]
    for subdir in subdirs:
        (module_path / subdir).mkdir(parents=True, exist_ok=True)
        typer.echo(f"  Created {module_path / subdir}/")

    readme_path = module_path / "README.md"
    readme_content = MODULE_README_TEMPLATE.format(
        number=number,
        title=title,
        description=description,
    )
    readme_path.write_text(readme_content)
    typer.echo(f"  Created {readme_path}")

    init_path = module_path / "src" / "__init__.py"
    init_content = INIT_PY_TEMPLATE.format(number=number, title=title)
    init_path.write_text(init_content)
    typer.echo(f"  Created {init_path}")

    main_path = module_path / "src" / "main.py"
    main_content = MAIN_PY_TEMPLATE.format(number=number, title=title)
    main_path.write_text(main_content)
    typer.echo(f"  Created {main_path}")

    test_init_path = module_path / "tests" / "__init__.py"
    test_init_path.write_text('"""Tests for module."""\n')
    typer.echo(f"  Created {test_init_path}")

    test_path = module_path / "tests" / "test_module.py"
    test_content = TEST_TEMPLATE.format(number=number, title=title)
    test_path.write_text(test_content)
    typer.echo(f"  Created {test_path}")

    config_path = module_path / "configs" / "experiment.yaml"
    config_content = CONFIG_TEMPLATE.format(
        number=number,
        module_name=module_name,
    )
    config_path.write_text(config_content)
    typer.echo(f"  Created {config_path}")

    notebook_path = module_path / "notebooks" / "01_exploration.ipynb"
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# Module {number}: {title}\\n\\nExploration notebook"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# TODO: Add exploration code"],
                "outputs": [],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(notebook_path, "w") as f:
        json.dump(notebook_content, f, indent=2)
    typer.echo(f"  Created {notebook_path}")

    typer.echo("")
    typer.secho("✓ Module created successfully!", fg=typer.colors.GREEN)
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"  1. Edit {module_path}/README.md")
    typer.echo(f"  2. Implement code in {module_path}/src/")
    typer.echo(f"  3. Add tests in {module_path}/tests/")
    typer.echo(f"  4. Run: pytest {module_path}/tests/ -v")


if __name__ == "__main__":
    app()
