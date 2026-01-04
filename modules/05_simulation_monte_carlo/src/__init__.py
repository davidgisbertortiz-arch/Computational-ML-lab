"""Module 05: Simulation & Monte Carlo Methods."""

from .mc_integration import MCIntegrator, estimate_integral
from .variance_reduction import ImportanceSampler, ControlVariates
from .rare_events import RareEventEstimator, tail_probability
from .synthetic_generators import (
    generate_brownian_motion,
    generate_ou_process,
    generate_levy_flight,
    PhysicsDataGenerator,
)

__all__ = [
    "MCIntegrator",
    "estimate_integral",
    "ImportanceSampler",
    "ControlVariates",
    "RareEventEstimator",
    "tail_probability",
    "generate_brownian_motion",
    "generate_ou_process",
    "generate_levy_flight",
    "PhysicsDataGenerator",
]
