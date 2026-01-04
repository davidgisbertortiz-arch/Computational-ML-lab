"""Data generation and loading for particle collision classification.

Generates physics-inspired synthetic tabular dataset with:
- 16 kinematic features
- Binary classification (signal vs background)
- 10% class imbalance
- Feature correlations mimicking real particle physics
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_particle_collision_data(
    n_samples: int = 100_000,
    signal_fraction: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic particle collision dataset.
    
    Physics-inspired features with realistic correlations:
    - Signal events: Rare particle decays with specific mass/energy signatures
    - Background: Standard model processes
    
    Args:
        n_samples: Total number of events to generate
        signal_fraction: Fraction of signal events (rest is background)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with features and 'is_signal' target column
    """
    rng = np.random.RandomState(random_state)
    n_signal = int(n_samples * signal_fraction)
    n_background = n_samples - n_signal
    
    # Generate signal events (rare particle decay)
    # Signal has higher invariant mass, missing energy, specific angular distribution
    signal_data = _generate_signal_events(n_signal, rng)
    
    # Generate background events (standard model processes)
    background_data = _generate_background_events(n_background, rng)
    
    # Combine and shuffle
    df = pd.concat([signal_data, background_data], axis=0, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def _generate_signal_events(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Generate signal events with characteristic physics signatures."""
    # Transverse momentum (higher for signal)
    p_T = rng.gamma(shape=3.0, scale=30.0, size=n)
    
    # Pseudorapidity (central for signal)
    eta = rng.normal(loc=0.0, scale=1.2, size=n)
    
    # Azimuthal angle (uniform)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    
    # Total energy (correlated with p_T)
    E_total = p_T * rng.gamma(shape=2.5, scale=1.5, size=n)
    
    # Invariant mass (signal has peak around 125 GeV - Higgs-like)
    m_inv = rng.normal(loc=125.0, scale=5.0, size=n)
    
    # Missing transverse energy (higher for signal - undetected particles)
    missing_E_T = rng.exponential(scale=40.0, size=n)
    
    # Number of jets (signal typically has 2-4 jets)
    n_jets = rng.poisson(lam=3.0, size=n)
    
    # B-jet tagging score (higher for signal)
    b_tag_score = rng.beta(a=5, b=2, size=n)
    
    # Lepton isolation (well-isolated leptons in signal)
    lepton_iso = rng.beta(a=8, b=2, size=n)
    
    # Angular separation (derived feature)
    delta_R = np.sqrt(eta**2 + phi**2)
    
    # Transverse mass (derived from p_T and missing_E_T)
    m_T = np.sqrt(2 * p_T * missing_E_T * (1 - np.cos(phi)))
    
    # Energy ratios
    E_ratio = missing_E_T / (E_total + 1e-6)
    
    # Sphericity (shape variable)
    sphericity = rng.beta(a=2, b=5, size=n)
    
    # Aplanarity (3D shape)
    aplanarity = rng.beta(a=2, b=8, size=n)
    
    # Centrality (energy distribution)
    centrality = rng.beta(a=6, b=3, size=n)
    
    # HT (scalar sum of jet p_T)
    H_T = p_T * rng.gamma(shape=2.0, scale=1.5, size=n)
    
    df = pd.DataFrame({
        'p_T': p_T,
        'eta': eta,
        'phi': phi,
        'E_total': E_total,
        'm_inv': m_inv,
        'missing_E_T': missing_E_T,
        'n_jets': n_jets,
        'b_tag_score': b_tag_score,
        'lepton_iso': lepton_iso,
        'delta_R': delta_R,
        'm_T': m_T,
        'E_ratio': E_ratio,
        'sphericity': sphericity,
        'aplanarity': aplanarity,
        'centrality': centrality,
        'H_T': H_T,
        'is_signal': 1
    })
    
    return df


def _generate_background_events(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Generate background events (standard model processes)."""
    # Transverse momentum (lower for background)
    p_T = rng.gamma(shape=2.0, scale=20.0, size=n)
    
    # Pseudorapidity (more forward/backward)
    eta = rng.normal(loc=0.0, scale=2.0, size=n)
    
    # Azimuthal angle (uniform)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    
    # Total energy
    E_total = p_T * rng.gamma(shape=2.0, scale=1.2, size=n)
    
    # Invariant mass (no specific peak, broader distribution)
    m_inv = rng.gamma(shape=2.5, scale=35.0, size=n)
    
    # Missing transverse energy (lower for background)
    missing_E_T = rng.exponential(scale=20.0, size=n)
    
    # Number of jets
    n_jets = rng.poisson(lam=2.0, size=n)
    
    # B-jet tagging score (lower for background)
    b_tag_score = rng.beta(a=2, b=5, size=n)
    
    # Lepton isolation (less isolated)
    lepton_iso = rng.beta(a=3, b=4, size=n)
    
    # Angular separation
    delta_R = np.sqrt(eta**2 + phi**2)
    
    # Transverse mass
    m_T = np.sqrt(2 * p_T * missing_E_T * (1 - np.cos(phi)))
    
    # Energy ratios
    E_ratio = missing_E_T / (E_total + 1e-6)
    
    # Sphericity
    sphericity = rng.beta(a=3, b=4, size=n)
    
    # Aplanarity
    aplanarity = rng.beta(a=2, b=6, size=n)
    
    # Centrality
    centrality = rng.beta(a=4, b=4, size=n)
    
    # HT
    H_T = p_T * rng.gamma(shape=1.8, scale=1.3, size=n)
    
    df = pd.DataFrame({
        'p_T': p_T,
        'eta': eta,
        'phi': phi,
        'E_total': E_total,
        'm_inv': m_inv,
        'missing_E_T': missing_E_T,
        'n_jets': n_jets,
        'b_tag_score': b_tag_score,
        'lepton_iso': lepton_iso,
        'delta_R': delta_R,
        'm_T': m_T,
        'E_ratio': E_ratio,
        'sphericity': sphericity,
        'aplanarity': aplanarity,
        'centrality': centrality,
        'H_T': H_T,
        'is_signal': 0
    })
    
    return df


def load_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load particle collision dataset from disk.
    
    Args:
        data_path: Path to CSV file. If None, uses default location.
        
    Returns:
        DataFrame with features and target
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if data_path is None:
        # Default to repo root data directory
        repo_root = Path(__file__).resolve().parents[3]
        data_path = repo_root / "data" / "particle_collisions.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Run `python -m modules.03_ml_tabular_foundations.src.data` to generate."
        )
    
    df = pd.read_csv(data_path)
    
    # Validate schema
    expected_cols = {
        'p_T', 'eta', 'phi', 'E_total', 'm_inv', 'missing_E_T',
        'n_jets', 'b_tag_score', 'lepton_iso', 'delta_R', 'm_T',
        'E_ratio', 'sphericity', 'aplanarity', 'centrality', 'H_T', 'is_signal'
    }
    actual_cols = set(df.columns)
    
    if expected_cols != actual_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        raise ValueError(
            f"Schema mismatch. Missing: {missing}, Extra: {extra}"
        )
    
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation/test sets.
    
    Uses stratified splits to maintain class balance in each set.
    
    Args:
        df: Full dataset with 'is_signal' target
        test_size: Fraction for test set
        val_size: Fraction of remaining data for validation
        random_state: Random seed
        stratify: Whether to stratify split by target
        
    Returns:
        (train_df, val_df, test_df) tuple
    """
    stratify_col = df['is_signal'] if stratify else None
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: separate validation from train
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for already removed test set
    stratify_col_train = train_val_df['is_signal'] if stratify else None
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_col_train
    )
    
    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns (excludes target).
    
    Args:
        df: DataFrame with features and target
        
    Returns:
        List of feature column names
    """
    return [col for col in df.columns if col != 'is_signal']


if __name__ == "__main__":
    """Generate and save particle collision dataset."""
    from modules._import_helper import safe_import_from
    
    set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')
    set_seed(42)
    
    # Generate data
    print("Generating particle collision dataset...")
    df = generate_particle_collision_data(
        n_samples=100_000,
        signal_fraction=0.1,
        random_state=42
    )
    
    # Save to repo data directory
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / "particle_collisions.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved dataset to {output_path}")
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Signal events: {df['is_signal'].sum():,} ({df['is_signal'].mean():.1%})")
    print(f"  Background events: {(~df['is_signal'].astype(bool)).sum():,}")
    print(f"  Features: {len(get_feature_columns(df))}")
    print(f"\nFeature columns:")
    for col in get_feature_columns(df):
        print(f"    - {col}")
