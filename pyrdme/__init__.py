"""
pyrdme - Lightweight spatial stochastic simulation using RDME.

A portable Python package for simulating the Reaction-Diffusion Master Equation
using the MPD (Multiparticle Diffusion) algorithm with operator splitting.

Quick Start
-----------
>>> from pyrdme import Lattice2D, Reaction, simulate_rdme
>>>
>>> # Create lattice
>>> lattice = Lattice2D(nx=50, ny=50, species=['A', 'B', 'C'], spacing=1e-6)
>>> lattice.add_particles('A', count=500)
>>> lattice.add_particles('B', count=500)
>>>
>>> # Define reactions
>>> reactions = [
...     Reaction(['A', 'B'], ['C'], k=1e-4),
...     Reaction(['C'], ['A', 'B'], k=1e-2),
... ]
>>>
>>> # Run simulation
>>> result = simulate_rdme(
...     lattice=lattice,
...     reactions=reactions,
...     diffusion={'A': 1e-12, 'B': 1e-12, 'C': 0.5e-12},
...     t_max=1.0,
...     record_every=0.01,
...     seed=42,
... )
>>>
>>> # Analyze results
>>> print(f"Final A count: {result.total_particles('A')}")

Features
--------
- 2D lattice with arbitrary species
- MPD algorithm with operator splitting
- Chemical reactions with proper stochastic propensities
- Periodic boundary conditions
- Optional GPU acceleration via CuPy (future)
- Integration with pycme for reaction definitions

See Also
--------
- OPENSPEC.md: Full specification document
- pycme: Chemical Master Equation solver (dependency)
"""

__version__ = '0.1.0'

# Core classes
from .lattice import Lattice2D

# Solvers
from .solvers import DiffusionSolver, MPDSolver, validate_timestep

# Simulation functions
from .simulate import simulate_rdme, simulate_diffusion_only, RDMEResult

# Re-export from pycme for convenience
from pycme import Reaction, get_all_species

__all__ = [
    # Version
    '__version__',

    # Core classes
    'Lattice2D',

    # Solvers
    'DiffusionSolver',
    'MPDSolver',

    # Simulation
    'simulate_rdme',
    'simulate_diffusion_only',
    'RDMEResult',

    # Utilities
    'validate_timestep',

    # From pycme
    'Reaction',
    'get_all_species',
]
