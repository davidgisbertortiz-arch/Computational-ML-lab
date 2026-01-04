"""Command-line interface for training and evaluation."""

import typer
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from typing import Optional
import logging
from modules._import_helper import safe_import_from

# Import module components
TrainingConfig = safe_import_from('06_deep_learning_systems.src.config', 'TrainingConfig')
create_model = safe_import_from('06_deep_learning_systems.src.models', 'create_model')
get_data_loaders = safe_import_from('06_deep_learning_systems.src.datasets', 'get_data_loaders')
Trainer = safe_import_from('06_deep_learning_systems.src.trainer', 'Trainer')
set_seed, load_config = safe_import_from(
    '00_repo_standards.src.mlphys_core',
    'set_seed', 'load_config'
)

app = typer.Typer(help="Deep Learning Systems - Training Framework")


def get_device(device_str: str) -> str:
    """
    Get appropriate device.
    
    Args:
        device_str: Device string ('auto', 'cpu', 'cuda', 'mps')
    
    Returns:
        Resolved device string
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


def plot_learning_curves(history: dict, output_path: Path) -> None:
    """
    Plot and save learning curves.
    
    Args:
        history: Training history dictionary
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, history['val_accuracy'], 'o-', color='green', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


@app.command()
def train(
    config: Path = typer.Option(
        ...,
        help="Path to training config YAML file",
        exists=True,
    ),
    seed: Optional[int] = typer.Option(
        None,
        help="Random seed (overrides config)",
    ),
    device: Optional[str] = typer.Option(
        None,
        help="Device to use (overrides config): cpu, cuda, mps, auto",
    ),
    resume: Optional[Path] = typer.Option(
        None,
        help="Path to checkpoint to resume from",
        exists=True,
    ),
):
    """
    Train a model with specified configuration.
    
    Example:
        python -m modules.06_deep_learning_systems.src.main train \\
            --config modules/06_deep_learning_systems/configs/mnist_baseline.yaml \\
            --seed 42 \\
            --device cuda
    """
    # Load config
    cfg = load_config(config, TrainingConfig)
    
    # Override with CLI args
    if seed is not None:
        cfg.seed = seed
    if device is not None:
        cfg.device = device
    if resume is not None:
        cfg.resume_from = resume
    
    # Set seed
    set_seed(cfg.seed)
    
    # Get device
    device_str = get_device(cfg.device)
    typer.echo(f"🚀 Starting training on device: {device_str}")
    
    # Create data loaders
    typer.echo("📦 Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    typer.echo(f"   Train batches: {len(train_loader)}")
    typer.echo(f"   Val batches: {len(val_loader)}")
    typer.echo(f"   Test batches: {len(test_loader)}")
    
    # Create model
    typer.echo(f"🏗️  Building model: {cfg.model_type}")
    model = create_model(cfg)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    typer.echo(f"   Parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(cfg, model, device=device_str)
    
    # Resume from checkpoint if specified
    if cfg.resume_from is not None:
        trainer.load_checkpoint(cfg.resume_from)
    
    # Train
    typer.echo(f"🎯 Training for {cfg.num_epochs} epochs...")
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    typer.echo("📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Save metrics
    final_metrics = {
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'best_val_accuracy': max(history['val_accuracy']),
        'best_epoch': int(history['val_accuracy'].index(max(history['val_accuracy'])) + 1),
        'total_epochs': len(history['train_loss']),
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'config': cfg.model_dump() if hasattr(cfg, 'model_dump') else dict(cfg),
    }
    
    metrics_path = cfg.output_dir / "metrics.json"
    trainer.save_metrics(final_metrics, "metrics.json")
    typer.echo(f"💾 Metrics saved to {metrics_path}")
    
    # Plot learning curves
    plot_path = cfg.output_dir / "learning_curves.png"
    plot_learning_curves(history, plot_path)
    typer.echo(f"📈 Learning curves saved to {plot_path}")
    
    # Print summary
    typer.echo("\n" + "=" * 60)
    typer.echo("🎉 Training Complete!")
    typer.echo("=" * 60)
    typer.echo(f"Test Accuracy:  {test_metrics['accuracy']:.2%}")
    typer.echo(f"Test Loss:      {test_metrics['loss']:.4f}")
    typer.echo(f"Best Val Epoch: {final_metrics['best_epoch']}")
    typer.echo(f"Best Val Acc:   {final_metrics['best_val_accuracy']:.2%}")
    typer.echo("=" * 60)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(
        ...,
        help="Path to model checkpoint",
        exists=True,
    ),
    config: Path = typer.Option(
        ...,
        help="Path to config YAML file",
        exists=True,
    ),
    device: str = typer.Option(
        "auto",
        help="Device to use: cpu, cuda, mps, auto",
    ),
    split: str = typer.Option(
        "test",
        help="Which split to evaluate: train, val, test",
    ),
):
    """
    Evaluate a trained model from checkpoint.
    
    Example:
        python -m modules.06_deep_learning_systems.src.main evaluate \\
            --checkpoint modules/06_deep_learning_systems/reports/checkpoints/best_model.pt \\
            --config modules/06_deep_learning_systems/configs/mnist_baseline.yaml \\
            --split test
    """
    # Load config
    cfg = load_config(config, TrainingConfig)
    
    # Get device
    device_str = get_device(device)
    typer.echo(f"🔍 Evaluating on device: {device_str}")
    
    # Load data
    typer.echo(f"📦 Loading {split} data...")
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    
    loader_map = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    eval_loader = loader_map.get(split)
    if eval_loader is None:
        typer.echo(f"❌ Invalid split: {split} (must be train/val/test)")
        raise typer.Exit(1)
    
    # Create model
    typer.echo(f"🏗️  Loading model: {cfg.model_type}")
    model = create_model(cfg)
    
    # Create trainer and load checkpoint
    trainer = Trainer(cfg, model, device=device_str)
    trainer.load_checkpoint(checkpoint)
    
    # Evaluate
    typer.echo(f"📊 Evaluating on {split} set...")
    metrics = trainer.evaluate(eval_loader)
    
    # Print results
    typer.echo("\n" + "=" * 60)
    typer.echo(f"📈 Evaluation Results ({split} set)")
    typer.echo("=" * 60)
    typer.echo(f"Accuracy: {metrics['accuracy']:.2%}")
    typer.echo(f"Loss:     {metrics['loss']:.4f}")
    typer.echo("=" * 60)


@app.command()
def info(
    config: Path = typer.Option(
        ...,
        help="Path to config YAML file",
        exists=True,
    ),
):
    """
    Display information about a config file.
    
    Example:
        python -m modules.06_deep_learning_systems.src.main info \\
            --config modules/06_deep_learning_systems/configs/mnist_baseline.yaml
    """
    cfg = load_config(config, TrainingConfig)
    
    typer.echo("\n" + "=" * 60)
    typer.echo("📋 Configuration Summary")
    typer.echo("=" * 60)
    typer.echo(f"Experiment:     {cfg.name}")
    typer.echo(f"Model:          {cfg.model_type}")
    typer.echo(f"Dataset:        {cfg.dataset}")
    typer.echo(f"Batch Size:     {cfg.batch_size}")
    typer.echo(f"Epochs:         {cfg.num_epochs}")
    typer.echo(f"Learning Rate:  {cfg.learning_rate}")
    typer.echo(f"Optimizer:      {cfg.optimizer}")
    typer.echo(f"Device:         {cfg.device}")
    typer.echo(f"Seed:           {cfg.seed}")
    typer.echo(f"Output Dir:     {cfg.output_dir}")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
