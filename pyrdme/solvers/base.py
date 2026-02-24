"""
Base solver interface for RDME simulations.

Defines the abstract interface that all RDME solvers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Mapping
import numpy as np

from ..lattice import Lattice2D

DiffusionField = Union[float, np.ndarray, Mapping[Union[str, int, Tuple[str, str]], float]]
DiffusionSpec = Dict[str, DiffusionField]


def _is_transition_matrix(field: DiffusionField) -> bool:
    """Return True if field is a transition-matrix style mapping (tuple keys)."""
    if not isinstance(field, Mapping):
        return False
    return any(isinstance(k, tuple) and len(k) == 2 for k in field.keys())


def diffusion_max_from_spec(diffusion: DiffusionSpec) -> float:
    """
    Compute the maximum diffusion coefficient from a diffusion specification.
    """
    d_max = 0.0
    for field in diffusion.values():
        if isinstance(field, (int, float, np.floating)):
            value = float(field)
            if value < 0:
                raise ValueError("Diffusion coefficients must be non-negative")
            d_max = max(d_max, value)
        elif isinstance(field, np.ndarray):
            if field.size == 0:
                continue
            if np.any(field < 0):
                raise ValueError("Diffusion coefficients must be non-negative")
            d_max = max(d_max, float(np.max(field)))
        elif isinstance(field, Mapping):
            if len(field) == 0:
                continue
            values = [float(v) for v in field.values()]
            if any(v < 0 for v in values):
                raise ValueError("Diffusion coefficients must be non-negative")
            d_max = max(d_max, max(values))
        else:
            raise TypeError(
                "Diffusion values must be float, ndarray, or mapping of site types to float"
            )
    return d_max


def _diffusion_maps_from_spec(lattice: Lattice2D, diffusion: DiffusionSpec) -> Dict[str, np.ndarray]:
    """
    Build per-species diffusion maps from a diffusion specification.

    Supports:
    - float: uniform diffusion
    - ndarray (nx, ny): site-specific diffusion
    - mapping: per-site-type diffusion (keys are site type names or IDs)

    Does NOT handle transition-matrix specs (tuple keys). Those are handled
    by _edge_maps_from_transition_matrix() instead.
    """
    maps: Dict[str, np.ndarray] = {}
    nx, ny = lattice.nx, lattice.ny

    for species in lattice.species:
        if species not in diffusion:
            raise ValueError(f"Missing diffusion coefficient for species '{species}'")

        field = diffusion[species]

        # Skip transition-matrix specs — handled separately
        if _is_transition_matrix(field):
            continue

        if isinstance(field, (int, float, np.floating)):
            value = float(field)
            if value < 0:
                raise ValueError(f"Diffusion coefficient for '{species}' must be non-negative")
            maps[species] = np.full((nx, ny), value, dtype=np.float64)
            continue

        if isinstance(field, np.ndarray):
            if field.shape != (nx, ny):
                raise ValueError(
                    f"Diffusion map for '{species}' must have shape {(nx, ny)}, got {field.shape}"
                )
            if np.any(field < 0):
                raise ValueError(f"Diffusion map for '{species}' must be non-negative")
            maps[species] = field.astype(np.float64, copy=False)
            continue

        if isinstance(field, Mapping):
            # Double-check: skip transition-matrix specs (tuple keys)
            if _is_transition_matrix(field):
                continue

            if len(field) == 0:
                raise ValueError(f"Diffusion mapping for '{species}' cannot be empty")

            default_set = False
            default_value = 0.0
            if 'default' in field:
                default_value = float(field['default'])
                default_set = True
            elif 0 in field:
                default_value = float(field[0])
                default_set = True

            if default_set and default_value < 0:
                raise ValueError(f"Diffusion coefficient for '{species}' must be non-negative")

            if lattice.site_types is None:
                if not default_set or len(field) > 1:
                    raise ValueError(
                        "Site types not set on lattice. Call lattice.set_site_type(...) "
                        "before using site-type diffusion maps."
                    )
                maps[species] = np.full((nx, ny), default_value, dtype=np.float64)
                continue

            site_types = lattice.site_types
            if default_set:
                dmap = np.full((nx, ny), default_value, dtype=np.float64)
            else:
                dmap = np.zeros((nx, ny), dtype=np.float64)

            assigned_ids = set()
            for key, value in field.items():
                if key == 'default' or key == 0:
                    continue
                val = float(value)
                if val < 0:
                    raise ValueError(f"Diffusion coefficient for '{species}' must be non-negative")

                if isinstance(key, str):
                    try:
                        type_id = lattice.site_type_id(key)
                    except KeyError as exc:
                        raise ValueError(
                            f"Unknown site type '{key}' in diffusion map for '{species}'."
                        ) from exc
                elif isinstance(key, (int, np.integer)):
                    type_id = int(key)
                else:
                    raise TypeError("Site type keys must be str or int")

                dmap[site_types == type_id] = val
                assigned_ids.add(type_id)

            if not default_set:
                present_ids = set(np.unique(site_types))
                missing = present_ids - assigned_ids
                if missing:
                    raise ValueError(
                        f"Diffusion map for '{species}' missing values for site types: {sorted(missing)}"
                    )

            maps[species] = dmap
            continue

        raise TypeError(
            "Diffusion values must be float, ndarray, or mapping of site types to float"
        )

    return maps


def diffusion_max(diffusion_maps: Dict[str, np.ndarray]) -> float:
    """Compute maximum diffusion coefficient from per-species diffusion maps."""
    d_max = 0.0
    for dmap in diffusion_maps.values():
        if dmap.size == 0:
            continue
        d_max = max(d_max, float(np.max(dmap)))
    return d_max


def _edge_maps_from_site_map(d_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive edge diffusion maps from a per-site diffusion map using the min-rule.

    Returns (edge_x, edge_y) each of shape (nx, ny).
    edge_x[i,j] = min(d_map[i,j], d_map[i+1,j])  (periodic)
    edge_y[i,j] = min(d_map[i,j], d_map[i,j+1])  (periodic)
    """
    neighbor_x = np.roll(d_map, shift=-1, axis=0)
    neighbor_y = np.roll(d_map, shift=-1, axis=1)
    edge_x = np.minimum(d_map, neighbor_x)
    edge_y = np.minimum(d_map, neighbor_y)
    return edge_x, edge_y


def _edge_maps_from_transition_matrix(
    lattice: Lattice2D,
    species: str,
    field: Mapping,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build x-edge and y-edge diffusion maps from a transition matrix.

    The transition matrix is a dict with (type_a, type_b) tuple keys
    mapping to diffusion rates. It is treated as symmetric.
    Unspecified pairs default to 0 (no diffusion).

    Returns (edge_x, edge_y) each of shape (nx, ny).
    edge_x[i,j] = D for edge between site (i,j) and (i+1,j) (periodic).
    edge_y[i,j] = D for edge between site (i,j) and (i,j+1) (periodic).
    """
    if lattice.site_types is None:
        raise ValueError(
            "Site types not set on lattice. Call lattice.set_site_type(...) "
            "before using transition-matrix diffusion."
        )

    site_types = lattice.site_types
    max_id = int(np.max(site_types))
    n_types = max_id + 1
    rate_table = np.zeros((n_types, n_types), dtype=np.float64)

    for key, value in field.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(
                f"Transition matrix keys must be 2-tuples of site type names, got {key!r}"
            )
        val = float(value)
        if val < 0:
            raise ValueError(f"Diffusion coefficient for '{species}' must be non-negative")

        try:
            id_a = lattice.site_type_id(key[0])
        except KeyError as exc:
            raise ValueError(
                f"Unknown site type '{key[0]}' in transition matrix for '{species}'."
            ) from exc
        try:
            id_b = lattice.site_type_id(key[1])
        except KeyError as exc:
            raise ValueError(
                f"Unknown site type '{key[1]}' in transition matrix for '{species}'."
            ) from exc

        rate_table[id_a, id_b] = val
        rate_table[id_b, id_a] = val  # symmetric

    # Build edge maps via fancy indexing
    neighbor_x = np.roll(site_types, shift=-1, axis=0)  # type at (i+1, j)
    neighbor_y = np.roll(site_types, shift=-1, axis=1)  # type at (i, j+1)

    edge_x = rate_table[site_types, neighbor_x]
    edge_y = rate_table[site_types, neighbor_y]

    return edge_x, edge_y


def _build_edge_maps(
    lattice: Lattice2D,
    diffusion: DiffusionSpec,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Build precomputed edge diffusion maps for all species.

    For transition-matrix specs, builds edge maps directly from the matrix.
    For other specs (float, ndarray, site-type mapping), builds per-site maps
    first, then derives edge maps using the min-rule.

    Returns (edge_maps_x, edge_maps_y) where each is a dict mapping
    species name to an (nx, ny) array of edge diffusion rates.
    """
    edge_maps_x: Dict[str, np.ndarray] = {}
    edge_maps_y: Dict[str, np.ndarray] = {}

    # Build per-site maps for non-transition-matrix species
    site_maps = _diffusion_maps_from_spec(lattice, diffusion)

    for species in lattice.species:
        if species not in diffusion:
            raise ValueError(f"Missing diffusion coefficient for species '{species}'")

        field = diffusion[species]

        if _is_transition_matrix(field):
            ex, ey = _edge_maps_from_transition_matrix(lattice, species, field)
        else:
            ex, ey = _edge_maps_from_site_map(site_maps[species])

        edge_maps_x[species] = ex
        edge_maps_y[species] = ey

    return edge_maps_x, edge_maps_y


class RDMESolver(ABC):
    """
    Abstract base class for RDME solvers.

    Subclasses implement specific algorithms (e.g., MPD, NSM).

    Parameters
    ----------
    lattice : Lattice2D
        The spatial domain.
    diffusion : DiffusionSpec
        Diffusion coefficients for each species (m²/s).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        lattice: Lattice2D,
        diffusion: DiffusionSpec,
        seed: Optional[int] = None,
    ):
        self.lattice = lattice
        self.diffusion = diffusion
        self.rng = np.random.default_rng(seed)

        # Validate diffusion coefficients
        self._validate_diffusion()

    def _validate_diffusion(self) -> None:
        """Check that all species have diffusion coefficients and precompute edge maps."""
        self._diffusion_max = diffusion_max_from_spec(self.diffusion)
        self._edge_maps_x, self._edge_maps_y = _build_edge_maps(self.lattice, self.diffusion)

    def _diffusion_sweep_x(self, dt: float) -> None:
        """
        Perform diffusion sweep in x-direction using precomputed edge maps.

        For each species, for each site:
        - Calculate q from precomputed edge rates
        - Sample particles moving left/right using binomial distribution
        - Apply periodic boundary conditions via np.roll
        """
        for species in self.lattice.species:
            species_idx = self.lattice.species_index(species)

            edge_map = self._edge_maps_x[species]
            if np.max(edge_map) == 0:
                continue

            counts = self.lattice.counts[species_idx, :, :]

            # edge_map[i,j] = D for edge between (i,j) and (i+1,j)
            q_right = edge_map * dt / self.lattice.spacing**2
            q_left = np.roll(edge_map, shift=1, axis=0) * dt / self.lattice.spacing**2

            if np.any(q_left + q_right > 1.0 + 1e-12):
                raise ValueError("Timestep too large for diffusion stability in x-sweep.")

            n_left = self.rng.binomial(counts, q_left)
            remaining = counts - n_left
            denom = 1.0 - q_left
            q_right_cond = np.zeros_like(q_right, dtype=np.float64)
            np.divide(q_right, denom, out=q_right_cond, where=denom > 0)
            n_right = self.rng.binomial(remaining, q_right_cond)

            # Apply with periodic boundaries
            # n_left[i] particles leave site i going to i-1 → arrive at i-1
            # n_right[i] particles leave site i going to i+1 → arrive at i+1
            counts_new = remaining - n_right
            counts_new += np.roll(n_left, shift=-1, axis=0)
            counts_new += np.roll(n_right, shift=1, axis=0)

            self.lattice.counts[species_idx, :, :] = counts_new

    def _diffusion_sweep_y(self, dt: float) -> None:
        """
        Perform diffusion sweep in y-direction using precomputed edge maps.

        Same algorithm as x-sweep but for y-direction.
        """
        for species in self.lattice.species:
            species_idx = self.lattice.species_index(species)

            edge_map = self._edge_maps_y[species]
            if np.max(edge_map) == 0:
                continue

            counts = self.lattice.counts[species_idx, :, :]

            # edge_map[i,j] = D for edge between (i,j) and (i,j+1)
            q_up = edge_map * dt / self.lattice.spacing**2
            q_down = np.roll(edge_map, shift=1, axis=1) * dt / self.lattice.spacing**2

            if np.any(q_down + q_up > 1.0 + 1e-12):
                raise ValueError("Timestep too large for diffusion stability in y-sweep.")

            n_down = self.rng.binomial(counts, q_down)
            remaining = counts - n_down
            denom = 1.0 - q_down
            q_up_cond = np.zeros_like(q_up, dtype=np.float64)
            np.divide(q_up, denom, out=q_up_cond, where=denom > 0)
            n_up = self.rng.binomial(remaining, q_up_cond)

            # Apply with periodic boundaries
            # n_down[i,j] particles leave (i,j) going to (i,j-1) → arrive at j-1
            # n_up[i,j] particles leave (i,j) going to (i,j+1) → arrive at j+1
            counts_new = remaining - n_up
            counts_new += np.roll(n_down, shift=-1, axis=1)
            counts_new += np.roll(n_up, shift=1, axis=1)

            self.lattice.counts[species_idx, :, :] = counts_new

    @abstractmethod
    def step(self, dt: float) -> None:
        """
        Advance the simulation by one timestep.

        Parameters
        ----------
        dt : float
            Timestep duration (seconds).
        """
        pass

    @abstractmethod
    def get_max_timestep(self) -> float:
        """
        Get the maximum stable timestep for this solver.

        Returns
        -------
        float
            Maximum dt satisfying stability constraints.
        """
        pass
