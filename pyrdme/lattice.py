"""
2D Lattice representation for RDME simulations.

This module provides the Lattice2D class for representing spatial domains
as discrete grids with particle counts per site.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Literal, Dict
import numpy as np


DistributionType = Literal['uniform', 'random']


@dataclass
class Lattice2D:
    """
    A 2D lattice for spatial stochastic simulation.

    The lattice is a rectangular grid where each site can hold particles
    of multiple species. Particle counts are stored as integers.

    Parameters
    ----------
    nx : int
        Number of grid points in x-direction.
    ny : int
        Number of grid points in y-direction.
    species : List[str]
        Names of chemical species.
    spacing : float, optional
        Physical distance between adjacent sites (default: 1e-6 meters = 1 μm).

    Attributes
    ----------
    shape : Tuple[int, int]
        Grid dimensions (nx, ny).
    counts : np.ndarray
        Particle counts with shape (n_species, nx, ny), dtype int32.
    site_types : np.ndarray or None
        Optional site type labels with shape (nx, ny), dtype int16.

    Examples
    --------
    >>> lattice = Lattice2D(nx=100, ny=100, species=['A', 'B', 'C'])
    >>> lattice.add_particles('A', count=1000, region=(slice(40, 60), slice(40, 60)))
    >>> print(lattice.total_particles('A'))
    1000
    """

    nx: int
    ny: int
    species: List[str]
    spacing: float = 1e-6

    # These are initialized in __post_init__
    counts: np.ndarray = field(init=False, repr=False)
    site_types: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _species_to_idx: dict = field(init=False, repr=False)
    _site_type_to_id: Dict[str, int] = field(init=False, repr=False)
    _site_id_to_type: Dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize arrays after dataclass fields are set."""
        self.counts = np.zeros((len(self.species), self.nx, self.ny), dtype=np.int32)
        self._species_to_idx = {species: i for i, species in enumerate(self.species)}
        self._site_type_to_id = {'default': 0}
        self._site_id_to_type = {0: 'default'}

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid dimensions (nx, ny)."""
        return (self.nx, self.ny)

    @property
    def n_species(self) -> int:
        """Number of species."""
        return len(self.species)

    def species_index(self, species: str) -> int:
        """
        Get the index of a species.

        Parameters
        ----------
        species : str
            Species name.

        Returns
        -------
        int
            Index into the counts array.

        Raises
        ------
        KeyError
            If species is not in the lattice.
        """
        # TODO: Return index from _species_to_idx, raise KeyError if not found
        return self._species_to_idx[species]
        # raise NotImplementedError

    def add_particles(
        self,
        species: str,
        count: int,
        region: Optional[Tuple[slice, slice]] = None,
        distribution: DistributionType = 'uniform',
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Add particles to the lattice.

        Parameters
        ----------
        species : str
            Species to add.
        count : int
            Total number of particles to add.
        region : Tuple[slice, slice], optional
            Region to add particles (x_slice, y_slice). Default is entire lattice.
        distribution : {'uniform', 'random'}
            How to distribute particles:
            - 'uniform': Spread evenly across sites (may have remainder)
            - 'random': Place each particle at a random site in region
        rng : np.random.Generator, optional
            Random generator for 'random' distribution.

        Examples
        --------
        >>> lattice.add_particles('A', count=1000)  # Whole lattice, uniform
        >>> lattice.add_particles('B', count=500, region=(slice(0, 50), slice(0, 50)))
        >>> lattice.add_particles('C', count=100, distribution='random')
        """
        # TODO: Implement particle addition
        # 1. Get species index
        # 2. Determine region (default to full lattice)
        # 3. If uniform: divide count by number of sites, distribute evenly
        # 4. If random: place particles one-by-one at random sites

        # Get species index 
        species_idx = self.species_index(species)

        # Determine region (default to full lattice)
        if region is None:
            region = (slice(0, self.nx), slice(0, self.ny))

        # Calculate region dimensions
        x_indices = np.arange(self.nx)[region[0]]
        y_indices = np.arange(self.ny)[region[1]]
        region_nx = len(x_indices)
        region_ny = len(y_indices)
        n_sites = region_nx * region_ny

        if distribution == 'uniform':
            # Distribute evenly, putting remainder particles one per site
            count_per_site = count // n_sites
            remainder = count % n_sites
            self.counts[species_idx, region[0], region[1]] += count_per_site
            # Distribute remainder across first `remainder` sites
            if remainder > 0:
                flat_idx = 0
                for xi in x_indices:
                    for yi in y_indices:
                        if flat_idx >= remainder:
                            break
                        self.counts[species_idx, xi, yi] += 1
                        flat_idx += 1
                    if flat_idx >= remainder:
                        break

        elif distribution == 'random':
            if rng is None:
                rng = np.random.default_rng()

            # Place particles one-by-one at random sites within region (with replacement)
            for _ in range(count):
                xi = rng.choice(x_indices)
                yi = rng.choice(y_indices)
                self.counts[species_idx, xi, yi] += 1
            

        #raise NotImplementedError

    def set_counts(
        self,
        species: str,
        counts: np.ndarray,
        region: Optional[Tuple[slice, slice]] = None,
    ) -> None:
        """
        Directly set particle counts for a species.

        Parameters
        ----------
        species : str
            Species name.
        counts : np.ndarray
            2D array of counts to set.
        region : Tuple[slice, slice], optional
            Region to set. Must match counts shape.
        """
        # TODO: Set counts directly for a species in a region
        
        # Get species index
        species_idx = self.species_index(species)

        # Determine region (default to full lattice)
        if region is None:
            region = (slice(0, self.nx), slice(0, self.ny))

        # Set counts directly for a species in a region
        self.counts[species_idx, region[0], region[1]] = counts

        # raise NotImplementedError

    def get_counts(self, species: str) -> np.ndarray:
        """
        Get the count array for a species.

        Parameters
        ----------
        species : str
            Species name.

        Returns
        -------
        np.ndarray
            2D array of shape (nx, ny) with particle counts.
        """
        # TODO: Return counts[species_idx, :, :]

        # Get species index
        species_idx = self.species_index(species)

        # Return counts for a species in a region
        return self.counts[species_idx, :, :]
        # raise NotImplementedError

    def total_particles(self, species: Optional[str] = None) -> int:
        """
        Get total particle count.

        Parameters
        ----------
        species : str, optional
            Species to count. If None, count all species.

        Returns
        -------
        int
            Total particle count.
        """
        # TODO: Sum counts for one species or all species 

        # Sum counts for one species or all species 
        if species is None:
            return np.sum(self.counts)
        else:
            species_idx = self.species_index(species)
            return np.sum(self.counts[species_idx, :, :])
        # raise NotImplementedError

    def copy(self) -> 'Lattice2D':
        """
        Create a deep copy of the lattice.

        Returns
        -------
        Lattice2D
            Independent copy with same state.
        """
        # TODO: Create a new Lattice2D with copied arrays

        new_lattice = Lattice2D(nx=self.nx, ny=self.ny, species=list(self.species), spacing=self.spacing)
        new_lattice.counts = self.counts.copy()
        if self.site_types is not None:
            new_lattice.site_types = self.site_types.copy()
        new_lattice._site_type_to_id = dict(self._site_type_to_id)
        new_lattice._site_id_to_type = dict(self._site_id_to_type)
        return new_lattice
        # raise NotImplementedError

    def site_type_map(self) -> Dict[str, int]:
        """
        Get the mapping of site type names to integer IDs.

        Returns
        -------
        Dict[str, int]
            Mapping from site type name to numeric ID.
        """
        return dict(self._site_type_to_id)

    def site_type_id(self, site_type: Union[str, int]) -> int:
        """
        Resolve a site type name or ID to an integer ID.

        Parameters
        ----------
        site_type : str or int
            Site type name or numeric ID.

        Returns
        -------
        int
            Site type ID.
        """
        if isinstance(site_type, str):
            if site_type not in self._site_type_to_id:
                raise KeyError(f"Unknown site type '{site_type}'. Define it via set_site_type().")
            return self._site_type_to_id[site_type]
        if isinstance(site_type, (int, np.integer)):
            return int(site_type)
        raise TypeError(f"site_type must be str or int, got {type(site_type).__name__}")

    def _get_or_create_site_type_id(self, site_type: str) -> int:
        """Get an existing site type ID or create a new one."""
        if site_type in self._site_type_to_id:
            return self._site_type_to_id[site_type]
        new_id = max(self._site_type_to_id.values(), default=-1) + 1
        self._site_type_to_id[site_type] = new_id
        self._site_id_to_type[new_id] = site_type
        return new_id

    def set_site_type(
        self,
        region: Union[Tuple[slice, slice], np.ndarray],
        site_type: Union[str, int],
    ) -> None:
        """
        Set site types for a region (for future region-specific behavior).

        Parameters
        ----------
        region : Tuple[slice, slice] or np.ndarray
            Region to set (x_slice, y_slice) or boolean mask of shape (nx, ny).
        site_type : str or int
            Site type identifier.
        """
        if self.site_types is None:
            self.site_types = np.zeros((self.nx, self.ny), dtype=np.int16)

        if isinstance(site_type, str):
            type_id = self._get_or_create_site_type_id(site_type)
        elif isinstance(site_type, (int, np.integer)):
            type_id = int(site_type)
            if type_id < 0:
                raise ValueError("site_type ID must be non-negative")
        else:
            raise TypeError(f"site_type must be str or int, got {type(site_type).__name__}")

        if isinstance(region, tuple):
            if len(region) != 2:
                raise ValueError("region must be a (x_slice, y_slice) tuple")
            self.site_types[region[0], region[1]] = type_id
            return

        if isinstance(region, np.ndarray):
            if region.shape != (self.nx, self.ny):
                raise ValueError(f"region mask must have shape {(self.nx, self.ny)}")
            if region.dtype != np.bool_:
                raise ValueError("region mask must be boolean")
            self.site_types[region] = type_id
            return

        raise TypeError("region must be a (slice, slice) tuple or a boolean mask array")

    def __str__(self) -> str:
        total = self.total_particles()
        return f"Lattice2D({self.nx}×{self.ny}, {self.n_species} species, {total} particles)"
