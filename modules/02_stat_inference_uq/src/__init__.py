"""Module 02: Statistical Inference & Uncertainty Quantification.

This module implements Bayesian inference, calibration metrics, and MCMC sampling
with uncertainty as a first-class object.
"""

import sys
import importlib

# Workaround for Python's octal literal parsing with module names starting with 0
_mod_regression = importlib.import_module('.bayesian_regression', 'modules.02_stat_inference_uq.src')
_mod_calibration = importlib.import_module('.calibration', 'modules.02_stat_inference_uq.src')
_mod_mcmc = importlib.import_module('.mcmc_basics', 'modules.02_stat_inference_uq.src')

BayesianLinearRegression = _mod_regression.BayesianLinearRegression
posterior_predictive = _mod_regression.posterior_predictive
reliability_diagram = _mod_calibration.reliability_diagram
expected_calibration_error = _mod_calibration.expected_calibration_error
TemperatureScaling = _mod_calibration.TemperatureScaling
MetropolisHastings = _mod_mcmc.MetropolisHastings
MCMCDiagnostics = _mod_mcmc.MCMCDiagnostics

__all__ = [
    "BayesianLinearRegression",
    "posterior_predictive",
    "reliability_diagram",
    "expected_calibration_error",
    "TemperatureScaling",
    "MetropolisHastings",
    "MCMCDiagnostics",
]
