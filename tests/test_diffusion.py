"""
Tests for diffusion solver.

The key validation is that a Gaussian blob should spread according to:
    σ²(t) = σ²(0) + 2Dt  (in each dimension)

This is the fundamental property of diffusion we need to verify.
"""

import pytest
import numpy as np

from pyrdme import Lattice2D
from pyrdme.solvers import DiffusionSolver, validate_timestep


class TestValidateTimestep:
    """Test timestep validation."""

    def test_valid_timestep(self):
        """Test that valid timesteps pass validation."""
        diffusion = {'A': 1e-12}
        spacing = 1e-6

        # Max dt = 0.5 * (1e-6)^2 / 1e-12 = 0.5e-6 / 1e-12 = 5e-1 = 0.5
        # Use 90% of max
        validate_timestep(dt=0.4, diffusion=diffusion, spacing=spacing)

    def test_invalid_timestep_raises(self):
        """Test that too-large timesteps raise ValueError."""
        diffusion = {'A': 1e-12}
        spacing = 1e-6

        # Max dt = 0.5 s, try 0.6 s
        with pytest.raises(ValueError, match="exceeds stability limit"):
            validate_timestep(dt=0.6, diffusion=diffusion, spacing=spacing)

    def test_zero_diffusion_any_timestep(self):
        """Test that zero diffusion allows any timestep."""
        diffusion = {'A': 0.0}
        spacing = 1e-6

        # Should not raise
        validate_timestep(dt=1000.0, diffusion=diffusion, spacing=spacing)


class TestDiffusionSolverInit:
    """Test DiffusionSolver initialization."""

    def test_basic_creation(self):
        """Test creating a diffusion solver."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'])
        solver = DiffusionSolver(lattice, diffusion={'A': 1e-12})

        assert solver.lattice is lattice
        assert solver.diffusion == {'A': 1e-12}

    def test_missing_diffusion_raises(self):
        """Test that missing diffusion coefficient raises error."""
        lattice = Lattice2D(nx=10, ny=10, species=['A', 'B'])

        with pytest.raises(ValueError, match="Missing diffusion coefficient"):
            DiffusionSolver(lattice, diffusion={'A': 1e-12})  # Missing B

    def test_negative_diffusion_raises(self):
        """Test that negative diffusion coefficient raises error."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'])

        with pytest.raises(ValueError, match="must be non-negative"):
            DiffusionSolver(lattice, diffusion={'A': -1e-12})


class TestDiffusionSolverMaxTimestep:
    """Test max timestep calculation."""

    def test_max_timestep_single_species(self):
        """Test max timestep with single species."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'], spacing=1e-6)
        solver = DiffusionSolver(lattice, diffusion={'A': 1e-12})

        # Max dt = 0.5 * (1e-6)^2 / 1e-12 = 0.5 s
        assert solver.get_max_timestep() == pytest.approx(0.5)

    def test_max_timestep_multiple_species(self):
        """Test max timestep uses fastest diffuser."""
        lattice = Lattice2D(nx=10, ny=10, species=['A', 'B'], spacing=1e-6)
        solver = DiffusionSolver(lattice, diffusion={'A': 1e-12, 'B': 2e-12})

        # Max dt determined by B (faster): 0.5 * (1e-6)^2 / 2e-12 = 0.25 s
        assert solver.get_max_timestep() == pytest.approx(0.25)


class TestDiffusionSolverStep:
    """Test diffusion stepping."""

    def test_step_conserves_particles(self):
        """Test that diffusion conserves total particle count."""
        lattice = Lattice2D(nx=20, ny=20, species=['A'], spacing=1e-6)
        lattice.add_particles('A', count=1000, region=(slice(8, 12), slice(8, 12)))

        solver = DiffusionSolver(lattice, diffusion={'A': 1e-12}, seed=42)
        dt = solver.get_max_timestep() * 0.9

        initial_count = lattice.total_particles('A')

        for _ in range(100):
            solver.step(dt)

        final_count = lattice.total_particles('A')
        assert final_count == initial_count

    def test_step_too_large_raises(self):
        """Test that too-large timestep raises error."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'], spacing=1e-6)
        solver = DiffusionSolver(lattice, diffusion={'A': 1e-12})

        max_dt = solver.get_max_timestep()

        with pytest.raises(ValueError):
            solver.step(max_dt * 1.1)


class TestSiteSpecificDiffusion:
    """Test site-type diffusion maps."""

    def test_membrane_barrier_blocks_transfer(self):
        lattice = Lattice2D(nx=3, ny=1, species=['A'])

        mask = np.zeros((3, 1), dtype=bool)
        mask[1, 0] = True
        lattice.set_site_type(mask, 'membrane')

        lattice.add_particles('A', count=100, region=(slice(0, 1), slice(0, 1)))

        diffusion = {'A': {'default': 1e-12, 'membrane': 0.0}}
        solver = DiffusionSolver(lattice, diffusion=diffusion, seed=42)
        dt = solver.get_max_timestep() * 0.5

        for _ in range(50):
            solver.step(dt)

        assert lattice.get_counts('A')[1, 0] == 0
        assert lattice.total_particles('A') == 100


class TestDiffusionPhysics:
    """
    Test that diffusion obeys expected physics.

    Key test: Gaussian blob spreading follows σ²(t) = σ²(0) + 2Dt
    """

    def test_gaussian_spreading(self):
        """
        Test that variance increases as expected for diffusion.

        Initialize a point source (all particles at center).
        After time t, the variance should be approximately 2Dt in each dimension.
        """
        # Setup
        nx, ny = 100, 100
        spacing = 1e-6  # 1 μm
        D = 1e-12  # m²/s

        lattice = Lattice2D(nx=nx, ny=ny, species=['A'], spacing=spacing)

        # Put all particles at center
        center_x, center_y = nx // 2, ny // 2
        n_particles = 10000

        # Set counts directly at center
        counts = np.zeros((nx, ny), dtype=np.int32)
        counts[center_x, center_y] = n_particles
        lattice.set_counts('A', counts)

        # Create solver
        solver = DiffusionSolver(lattice, diffusion={'A': D}, seed=42)

        # Run simulation
        dt = solver.get_max_timestep() * 0.5  # Conservative timestep
        n_steps = 1000
        t_total = n_steps * dt

        for _ in range(n_steps):
            solver.step(dt)

        # Measure final variance
        final_variance = measure_variance_2d(lattice.get_counts('A'), spacing)

        # Expected variance: σ² = 2Dt (in each dimension, so total is 4Dt for 2D)
        # But we measure sum of x and y variances
        expected_variance_per_dim = 2 * D * t_total
        expected_total_variance = 2 * expected_variance_per_dim  # σ²_x + σ²_y

        # Allow 20% tolerance due to stochastic nature
        assert final_variance == pytest.approx(expected_total_variance, rel=0.2)

    def test_uniform_stays_uniform(self):
        """Test that uniform distribution stays uniform (steady state)."""
        lattice = Lattice2D(nx=20, ny=20, species=['A'], spacing=1e-6)

        # Uniform: 10 particles per site
        counts = np.full((20, 20), 10, dtype=np.int32)
        lattice.set_counts('A', counts)

        solver = DiffusionSolver(lattice, diffusion={'A': 1e-12}, seed=42)
        dt = solver.get_max_timestep() * 0.5

        # Run many steps
        for _ in range(1000):
            solver.step(dt)

        # Should still be approximately uniform
        final_counts = lattice.get_counts('A')
        mean = np.mean(final_counts)
        std = np.std(final_counts)

        # Coefficient of variation should be small
        cv = std / mean
        assert cv < 0.35  # Allow some fluctuation (stochastic test)


def measure_variance_2d(counts: np.ndarray, spacing: float) -> float:
    """
    Measure the spatial variance of a particle distribution.

    Parameters
    ----------
    counts : np.ndarray
        2D array of particle counts.
    spacing : float
        Distance between grid points.

    Returns
    -------
    float
        Sum of variance in x and y directions (σ²_x + σ²_y) in physical units.
    """
    nx, ny = counts.shape
    total = np.sum(counts)

    if total == 0:
        return 0.0

    # Create coordinate arrays (in physical units)
    x = np.arange(nx) * spacing
    y = np.arange(ny) * spacing

    # Calculate mean position
    x_mean = np.sum(counts * x[:, np.newaxis]) / total
    y_mean = np.sum(counts * y[np.newaxis, :]) / total

    # Calculate variance
    var_x = np.sum(counts * (x[:, np.newaxis] - x_mean) ** 2) / total
    var_y = np.sum(counts * (y[np.newaxis, :] - y_mean) ** 2) / total

    return var_x + var_y
