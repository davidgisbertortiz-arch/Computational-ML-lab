"""CLI for running the demo experiment."""

import sys
from pathlib import Path
import typer

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from modules._import_helper import safe_import_from

load_config = safe_import_from('00_repo_standards.src.mlphys_core.config', 'load_config')
DemoConfig, DemoExperiment = safe_import_from('00_repo_standards.src.demo_experiment', 'DemoConfig', 'DemoExperiment')

app = typer.Typer(
    help="Demo experiment showcasing repository standards",
    add_completion=False,
)


@app.command()
def main(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML file. If not provided, uses defaults.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (overrides config)",
    ),
) -> None:
    """
    Run demo classification experiment.
    
    Example:
        python -m modules.00_repo_standards.run_demo --seed 42
        
        python -m modules.00_repo_standards.run_demo \\
            --config configs/demo.yaml \\
            --seed 123
    """
    # Load or create config
    if config is not None:
        if not config.exists():
            typer.secho(f"Config file not found: {config}", fg=typer.colors.RED)
            raise typer.Exit(1)
        exp_config = load_config(config, DemoConfig)
        typer.echo(f"Loaded config from {config}")
    else:
        exp_config = DemoConfig(
            name="demo_experiment",
            description="Binary classification demo",
        )
        typer.echo("Using default configuration")
    
    # Override seed
    exp_config.seed = seed
    
    # Override output_dir if provided
    if output_dir is not None:
        exp_config.output_dir = output_dir
    
    typer.echo(f"Seed: {exp_config.seed}")
    typer.echo(f"Output: {exp_config.output_dir}")
    typer.echo("")
    
    # Run experiment
    try:
        experiment = DemoExperiment(exp_config)
        metrics = experiment.run()
        
        typer.secho("\n✓ Experiment completed successfully!", fg=typer.colors.GREEN)
        typer.echo("\nFinal metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                typer.echo(f"  {key}: {value:.4f}")
            else:
                typer.echo(f"  {key}: {value}")
        
        typer.echo(f"\nOutputs saved to: {experiment.output_dir}")
        
    except Exception as e:
        typer.secho(f"\n✗ Experiment failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
