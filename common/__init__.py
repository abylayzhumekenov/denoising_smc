"""Shared project-owned infrastructure (see docs/vision.md, D11).

The ODE baseline (`scripts/`) and the SMC implementation (`smc/`) both import from here; neither
imports the other. This package must not depend on either of them.
"""
