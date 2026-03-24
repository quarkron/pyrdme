"""
Main simulation function for RDME.

Provides the high-level simulate_rdme() function that runs a complete
reaction-diffusion simulation and records results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Union
import numpy as np

from .lattice import Lattice2D
from .solvers.mpd import MPDSolver
from .solvers.base import DiffusionSpec

# Import Reaction from pycme
from pycme import Reaction


BackendType = Literal['auto', 'cpu', 'gpu']


@dataclass
class RDMEResult:
    """
    Results from an RDME simulation.

    Stores snapshots of the lattice state at recorded times.

    Attributes
    ----------
    times : np.ndarray
        Array of recorded times (seconds).
    snapshots : List[Lattice2D]
        Lattice states at each recorded time.
    species : List[str]
        Species names (from initial lattice).
    shape : Tuple[int, int]
        Lattice dimensions (nx, ny).
    spacing : float
        Lattice spacing (meters).
    """

    times: np.ndarray
    snapshots: List[Lattice2D]
    species: List[str]
    shape: tuple
    spacing: float
    resource_history: Dict[str, list] = field(default_factory=dict)

    def get_snapshot(self, time: float) -> Lattice2D:
        """
        Get lattice snapshot at (or nearest to) a specific time.

        Parameters
        ----------
        time : float
            Desired time in seconds.

        Returns
        -------
        Lattice2D
            Snapshot of the lattice at the nearest recorded time.
        """
        idx = np.argmin(np.abs(self.times - time))
        return self.snapshots[idx]

    def get_counts(self, species: str, time: Optional[float] = None) -> np.ndarray:
        """
        Get particle counts for a species.

        Parameters
        ----------
        species : str
            Species name.
        time : float, optional
            If provided, return counts at that time. Otherwise, return
            the time series for all recorded times.

        Returns
        -------
        np.ndarray
            If time is provided: shape (nx, ny) with counts.
            Otherwise: shape (n_times, nx, ny) with count history.
        """
        if time is not None:
            snapshot = self.get_snapshot(time)
            return snapshot.get_counts(species)
        else:
            # Return full time series
            n_times = len(self.times)
            nx, ny = self.shape
            counts_history = np.zeros((n_times, nx, ny), dtype=np.int32)
            for i, snapshot in enumerate(self.snapshots):
                counts_history[i, :, :] = snapshot.get_counts(species)
            return counts_history

    def get_concentration(self, species: str, time: float) -> np.ndarray:
        """
        Get concentration field for a species (particles per unit area).

        Parameters
        ----------
        species : str
            Species name.
        time : float
            Time in seconds.

        Returns
        -------
        np.ndarray
            Shape (nx, ny) with concentration values (particles/m²).
        """
        counts = self.get_counts(species, time)
        site_area = self.spacing ** 2
        return counts.astype(np.float64) / site_area

    def get_timeseries(
        self,
        species: str,
        location: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Get time series of particle counts.

        Parameters
        ----------
        species : str
            Species name.
        location : Tuple[int, int], optional
            (x, y) location. If None, returns total count over all sites.

        Returns
        -------
        np.ndarray
            Shape (n_times,) with count at each recorded time.
        """
        counts_history = self.get_counts(species)

        if location is not None:
            x, y = location
            return counts_history[:, x, y]
        else:
            return np.sum(counts_history, axis=(1, 2))

    def total_particles(
        self,
        species: Optional[str] = None,
        time: Optional[float] = None,
    ) -> int:
        """
        Get total particle count.

        Parameters
        ----------
        species : str, optional
            Species to count. If None, count all.
        time : float, optional
            Time to query. If None, returns final count.

        Returns
        -------
        int
            Total particle count.
        """
        if time is None:
            snapshot = self.snapshots[-1]
        else:
            snapshot = self.get_snapshot(time)
        return snapshot.total_particles(species)

    @property
    def n_snapshots(self) -> int:
        """Number of recorded snapshots."""
        return len(self.snapshots)

    @property
    def t_max(self) -> float:
        """Final simulation time."""
        return self.times[-1] if len(self.times) > 0 else 0.0


def simulate_rdme(
    lattice: Lattice2D,
    reactions: List[Reaction],
    diffusion: DiffusionSpec,
    t_max: float,
    timestep: Optional[float] = None,
    record_every: Optional[float] = None,
    seed: Optional[int] = None,
    backend: BackendType = 'auto',
    reaction_sites: Optional[Dict[int, Union[str, List[str]]]] = None,
    global_resources: Optional[Dict[str, int]] = None,
    reaction_costs: Optional[Dict[int, Dict[str, int]]] = None,
) -> RDMEResult:
    """
    Run a reaction-diffusion master equation simulation.

    Parameters
    ----------
    lattice : Lattice2D
        Initial lattice state. Will be copied, not modified in place.
    reactions : List[Reaction]
        List of chemical reactions.
    diffusion : DiffusionSpec
        Diffusion coefficients for each species (m²/s).
    t_max : float
        Simulation end time in seconds.
    timestep : float, optional
        Simulation timestep. If None, uses 90% of max stable timestep.
    record_every : float, optional
        Recording interval. If None, records at every timestep.
    seed : int, optional
        Random seed for reproducibility.
    backend : {'auto', 'cpu', 'gpu'}
        Computation backend. 'gpu' requires CuPy (not yet implemented).

    Returns
    -------
    RDMEResult
        Simulation results with recorded snapshots.

    Raises
    ------
    ValueError
        If timestep exceeds stability limit.
        If species in reactions not found in lattice.
        If diffusion coefficients missing for lattice species.

    Examples
    --------
    >>> from pyrdme import Lattice2D, Reaction, simulate_rdme
    >>>
    >>> # Setup
    >>> lattice = Lattice2D(nx=50, ny=50, species=['A', 'B', 'C'], spacing=1e-6)
    >>> lattice.add_particles('A', count=500)
    >>> lattice.add_particles('B', count=500)
    >>>
    >>> reactions = [
    ...     Reaction(['A', 'B'], ['C'], k=1e-4),
    ...     Reaction(['C'], ['A', 'B'], k=1e-2),
    ... ]
    >>> diffusion = {'A': 1e-12, 'B': 1e-12, 'C': 0.5e-12}
    >>>
    >>> # Run
    >>> result = simulate_rdme(
    ...     lattice=lattice,
    ...     reactions=reactions,
    ...     diffusion=diffusion,
    ...     t_max=1.0,
    ...     record_every=0.01,
    ...     seed=42,
    ... )
    >>>
    >>> # Analyze
    >>> print(f"Final A count: {result.total_particles('A')}")
    >>> print(f"Recorded {result.n_snapshots} snapshots")
    """
    if backend == 'gpu':
        raise NotImplementedError("GPU backend not yet implemented. Use 'cpu' or 'auto'.")

    # Copy lattice to avoid modifying the original
    sim_lattice = lattice.copy()

    # Create solver
    solver = MPDSolver(
        sim_lattice, reactions, diffusion, seed=seed,
        reaction_sites=reaction_sites,
        global_resources=global_resources,
        reaction_costs=reaction_costs,
    )

    # Determine timestep
    max_dt = solver.get_max_timestep()
    if timestep is None:
        dt = max_dt * 0.9
    else:
        dt = timestep
        # Validation happens in solver.step()

    # Determine recording interval
    if record_every is None:
        record_interval = dt
    else:
        record_interval = record_every

    # Initialize recording
    times = [0.0]
    snapshots = [sim_lattice.copy()]

    # Initialize resource history tracking
    resource_history: Dict[str, list] = {}
    if solver.global_resources:
        for name, val in solver.global_resources.items():
            resource_history[name] = [val]

    # Run simulation
    t = 0.0
    next_record_time = record_interval

    while t < t_max:
        # Adjust final step to hit t_max exactly
        step_dt = min(dt, t_max - t)

        # Perform timestep
        solver.step(step_dt)
        t += step_dt

        # Record if needed
        if t >= next_record_time - 1e-12 or t >= t_max - 1e-12:
            times.append(t)
            snapshots.append(sim_lattice.copy())
            for name in resource_history:
                resource_history[name].append(solver.global_resources[name])
            next_record_time += record_interval

    return RDMEResult(
        times=np.array(times),
        snapshots=snapshots,
        species=list(lattice.species),
        shape=lattice.shape,
        spacing=lattice.spacing,
        resource_history=resource_history,
    )


def simulate_diffusion_only(
    lattice: Lattice2D,
    diffusion: DiffusionSpec,
    t_max: float,
    timestep: Optional[float] = None,
    record_every: Optional[float] = None,
    seed: Optional[int] = None,
) -> RDMEResult:
    """
    Run a diffusion-only simulation (no reactions).

    Convenience function that calls simulate_rdme with empty reactions.

    Parameters
    ----------
    lattice : Lattice2D
        Initial lattice state.
    diffusion : DiffusionSpec
        Diffusion coefficients for each species (m²/s).
    t_max : float
        Simulation end time in seconds.
    timestep : float, optional
        Simulation timestep. If None, uses 90% of maximum stable timestep.
    record_every : float, optional
        Recording interval. If None, records at every timestep.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    RDMEResult
        Simulation results.
    """
    return simulate_rdme(
        lattice=lattice,
        reactions=[],
        diffusion=diffusion,
        t_max=t_max,
        timestep=timestep,
        record_every=record_every,
        seed=seed,
        backend='cpu',
    )
