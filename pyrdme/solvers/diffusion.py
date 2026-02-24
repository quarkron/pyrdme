"""
Diffusion-only solver using the MPD (Multiparticle Diffusion) algorithm.

This module implements diffusion without reactions, useful for testing
and for systems where diffusion and reaction steps are separated.

Algorithm
---------
The MPD algorithm uses operator splitting with separate sweeps for each dimension.
For each particle at each site:
    - Calculate transition probability: q = D * dt / λ²
    - Draw random r ∈ [0, 1)
    - If r < q: move in negative direction
    - Elif r ∈ [0.5, 0.5+q): move in positive direction
    - Else: stay

Stability requires q ≤ 0.5, i.e., dt ≤ 0.5 * λ² / D_max
"""

from typing import Optional
import numpy as np
from .base import RDMESolver, DiffusionSpec, diffusion_max_from_spec


class DiffusionSolver(RDMESolver):
    """
    Diffusion-only solver using MPD algorithm.

    Implements 2D diffusion with periodic boundary conditions using
    operator splitting (separate X and Y sweeps).

    Parameters
    ----------
    lattice : Lattice2D
        The spatial domain.
    diffusion : DiffusionSpec
        Diffusion coefficients for each species (m²/s).
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> lattice = Lattice2D(nx=100, ny=100, species=['A'], spacing=1e-6)
    >>> lattice.add_particles('A', count=1000, region=(slice(45, 55), slice(45, 55)))
    >>> solver = DiffusionSolver(lattice, diffusion={'A': 1e-12}, seed=42)
    >>> dt = solver.get_max_timestep() * 0.9  # Use 90% of max for safety
    >>> for _ in range(1000):
    ...     solver.step(dt)
    """

    def get_max_timestep(self) -> float:
        """
        Calculate maximum stable timestep.

        The stability condition is: q = D*dt/λ² ≤ 0.5
        Therefore: dt ≤ 0.5 * λ² / D_max

        Returns
        -------
        float
            Maximum stable timestep in seconds.
        """
        D_max = self._diffusion_max
        if D_max == 0:
            return 1.0
        return 0.5 * self.lattice.spacing**2 / D_max

    def step(self, dt: float) -> None:
        """
        Perform one diffusion timestep.

        Parameters
        ----------
        dt : float
            Timestep duration in seconds.

        Raises
        ------
        ValueError
            If dt exceeds the stability limit.
        """

        # Validate timestep against stability condition
        validate_timestep(dt, self.diffusion, self.lattice.spacing)

        # Diffusion sweeps (inherited from RDMESolver)
        self._diffusion_sweep_x(dt)
        self._diffusion_sweep_y(dt)


def validate_timestep(dt: float, diffusion: DiffusionSpec, spacing: float) -> None:
    """
    Validate that a timestep satisfies stability conditions.

    Parameters
    ----------
    dt : float
        Proposed timestep.
    diffusion : DiffusionSpec
        Diffusion coefficients.
    spacing : float
        Lattice spacing.

    Raises
    ------
    ValueError
        If timestep is too large for stability.
    """
    D_max = diffusion_max_from_spec(diffusion)
    if D_max == 0:
        return  # No diffusion, any timestep is fine

    max_dt = 0.5 * spacing**2 / D_max
    if dt > max_dt:
        raise ValueError(
            f"Timestep {dt:.2e} s exceeds stability limit {max_dt:.2e} s. "
            f"Use dt <= 0.5 * λ² / D_max = 0.5 * {spacing:.2e}² / {D_max:.2e}"
        )
