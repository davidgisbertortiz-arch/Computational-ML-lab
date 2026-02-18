"""
Universal module runner for numeric-prefixed modules.

Bypasses Python 3.12+ import limitations by using importlib to load and run modules.

Usage:
    python -m modules.run list
    python -m modules.run run --module 00
    python -m modules.run run --module 03_ml_tabular_foundations
    python -m modules.run demo --module 00 --seed 42
"""

import re
import sys
from pathlib import Path
from typing import Optional

import typer

# Import the safe import helper
from modules._import_helper import safe_import

app = typer.Typer(help="Universal runner for numeric-prefixed modules")


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def list_modules() -> list[tuple[str, str, Path]]:
    """
    List all modules matching the numeric prefix pattern.
    
    Returns:
        List of (prefix, full_name, path) tuples, e.g., ('00', '00_repo_standards', Path(...))
    """
    modules_dir = get_repo_root() / "modules"
    pattern = re.compile(r"^(\d{2})_(.+)$")
    
    modules = []
    for item in sorted(modules_dir.iterdir()):
        if item.is_dir() and pattern.match(item.name):
            match = pattern.match(item.name)
            prefix = match.group(1)
            modules.append((prefix, item.name, item))
    
    return modules


def resolve_module_name(module_id: str) -> Optional[str]:
    """
    Resolve a module ID (e.g., '03' or '03_ml_tabular_foundations') to full name.
    
    Args:
        module_id: Either numeric prefix (e.g., '03') or full name
        
    Returns:
        Full module name (e.g., '03_ml_tabular_foundations') or None if not found
    """
    modules = list_modules()
    
    # Direct match
    for prefix, full_name, _ in modules:
        if module_id == full_name:
            return full_name
    
    # Prefix match
    for prefix, full_name, _ in modules:
        if module_id == prefix:
            return full_name
    
    return None


def find_entrypoint(module_name: str) -> Optional[tuple[str, str]]:
    """
    Find the canonical entrypoint for a module.
    
    Priority:
    1. run_demo.py with main()
    2. quick_test.py with main()
    3. src/main.py with main() or app
    
    Args:
        module_name: Full module name (e.g., '00_repo_standards')
        
    Returns:
        Tuple of (import_path, callable_name) or None
        e.g., ('00_repo_standards.run_demo', 'main')
    """
    module_path = get_repo_root() / "modules" / module_name
    
    # Priority 1: run_demo.py
    if (module_path / "run_demo.py").exists():
        return (f"{module_name}.run_demo", "main")
    
    # Priority 2: quick_test.py
    if (module_path / "quick_test.py").exists():
        return (f"{module_name}.quick_test", "main")
    
    # Priority 3: src/main.py
    if (module_path / "src" / "main.py").exists():
        # Check if it has main() or app
        return (f"{module_name}.src.main", "main_or_app")
    
    return None


@app.command()
def list() -> None:
    """List all available numeric-prefixed modules."""
    modules = list_modules()
    
    if not modules:
        typer.echo("No modules found matching pattern NN_*", err=True)
        raise typer.Exit(1)
    
    typer.echo("\n📚 Available Modules:\n")
    for prefix, full_name, path in modules:
        # Check if module has an entrypoint
        entrypoint = find_entrypoint(full_name)
        status = "✅" if entrypoint else "⚠️"
        
        typer.echo(f"  {status} [{prefix}] {full_name}")
        if entrypoint:
            typer.echo(f"      → {entrypoint[0]}")
        else:
            typer.echo(f"      → No entrypoint found (add run_demo.py or quick_test.py)")
    
    typer.echo("\n💡 Usage:")
    typer.echo(f"  python -m modules.run run --module 00")
    typer.echo(f"  python -m modules.run run --module 03_ml_tabular_foundations\n")


@app.command()
def run(
    module: str = typer.Option(..., "--module", "-m", help="Module ID (e.g., '00' or '00_repo_standards')"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed (if supported by module)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Run a module's canonical entrypoint.
    
    Examples:
        python -m modules.run run --module 00
        python -m modules.run run --module 03 --seed 42
    """
    # Resolve module name
    full_name = resolve_module_name(module)
    if not full_name:
        typer.echo(f"❌ Module '{module}' not found", err=True)
        typer.echo("\n💡 Available modules:", err=True)
        for prefix, name, _ in list_modules():
            typer.echo(f"   [{prefix}] {name}", err=True)
        raise typer.Exit(1)
    
    # Find entrypoint
    entrypoint = find_entrypoint(full_name)
    if not entrypoint:
        typer.echo(f"❌ No entrypoint found for module '{full_name}'", err=True)
        typer.echo("\n💡 Add one of:", err=True)
        typer.echo(f"   - modules/{full_name}/run_demo.py with main()", err=True)
        typer.echo(f"   - modules/{full_name}/quick_test.py with main()", err=True)
        typer.echo(f"   - modules/{full_name}/src/main.py with main() or app", err=True)
        raise typer.Exit(1)
    
    import_path, callable_name = entrypoint
    
    if verbose:
        typer.echo(f"🔧 Loading: {import_path}")
    
    # Import the module using safe_import
    try:
        module_obj = safe_import(import_path)
    except Exception as e:
        typer.echo(f"❌ Failed to import {import_path}: {e}", err=True)
        raise typer.Exit(1)
    
    # Execute the entrypoint
    try:
        if callable_name == "main_or_app":
            # Try main() first, then app
            if hasattr(module_obj, "main") and callable(getattr(module_obj, "main")):
                if verbose:
                    typer.echo(f"▶️  Calling {import_path}.main()")
                
                # Check if main accepts seed parameter
                main_func = getattr(module_obj, "main")
                import inspect
                sig = inspect.signature(main_func)
                if "seed" in sig.parameters and seed is not None:
                    main_func(seed=seed)
                else:
                    main_func()
            elif hasattr(module_obj, "app"):
                # Typer app - invoke it
                if verbose:
                    typer.echo(f"▶️  Invoking Typer app from {import_path}")
                app_obj = getattr(module_obj, "app")
                # For Typer apps, we can't easily pass arguments, so just invoke with defaults
                app_obj()
            else:
                typer.echo(f"❌ No main() or app found in {import_path}", err=True)
                raise typer.Exit(1)
        else:
            # Direct main() call
            if not hasattr(module_obj, callable_name):
                typer.echo(f"❌ No {callable_name}() found in {import_path}", err=True)
                raise typer.Exit(1)
            
            if verbose:
                typer.echo(f"▶️  Calling {import_path}.{callable_name}()")
            
            main_func = getattr(module_obj, callable_name)
            
            # Try to pass seed if supported
            import inspect
            sig = inspect.signature(main_func)
            if "seed" in sig.parameters and seed is not None:
                main_func(seed=seed)
            else:
                main_func()
                
    except Exception as e:
        typer.echo(f"❌ Error running {full_name}: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)
    
    if verbose:
        typer.echo(f"✅ Module {full_name} completed successfully")


@app.command()
def demo(
    module: str = typer.Option(..., "--module", "-m", help="Module ID"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Run a module's demo/quick test with specified seed.
    
    Convenience wrapper around 'run' command.
    """
    run(module=module, seed=seed, verbose=verbose)


if __name__ == "__main__":
    app()
