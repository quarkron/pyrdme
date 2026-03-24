"""
Full MPD-RDME solver with diffusion and reactions.

This module implements the complete MPD (Multiparticle Diffusion) algorithm
with operator splitting for both diffusion and chemical reactions.

Algorithm
---------
Each timestep consists of:
1. X-direction diffusion sweep (from DiffusionSolver)
2. Y-direction diffusion sweep (from DiffusionSolver)
3. Reaction step at each site independently

The reaction step uses the tau-leaping/Poisson method:
- Compute propensity for each reaction at each site
- Sample number of reaction firings from Poisson distribution
- Apply stoichiometry changes

Propensity Formulas
-------------------
| Reaction Type      | Formula              |
|--------------------|----------------------|
| ∅ → A              | a = k                |
| A → products       | a = k·nₐ             |
| A + B → products   | a = k·nₐ·n_b         |
| A + A → products   | a = k·nₐ·(nₐ-1)/2    |
"""

from typing import Dict, List, Optional, Union
import numpy as np

from ..lattice import Lattice2D
from .base import RDMESolver, DiffusionSpec
from .diffusion import validate_timestep

# Import Reaction from pycme
from pycme import Reaction

from ._numba_kernels import HAS_NUMBA, _apply_stoichiometry_numba


class MPDSolver(RDMESolver):
    """
    Full RDME solver using MPD algorithm with reactions.

    Combines diffusion (via operator splitting) with chemical reactions
    at each lattice site.

    Parameters
    ----------
    lattice : Lattice2D
        The spatial domain.
    reactions : List[Reaction]
        List of chemical reactions.
    diffusion : DiffusionSpec
        Diffusion coefficients for each species (m²/s).
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from pyrdme import Lattice2D, Reaction
    >>> from pyrdme.solvers import MPDSolver
    >>>
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
    >>> solver = MPDSolver(lattice, reactions, diffusion, seed=42)
    >>> dt = solver.get_max_timestep() * 0.9
    >>> for _ in range(1000):
    ...     solver.step(dt)
    """

    def __init__(
        self,
        lattice: Lattice2D,
        reactions: List[Reaction],
        diffusion: DiffusionSpec,
        seed: Optional[int] = None,
        reaction_sites: Optional[Dict[int, Union[str, List[str]]]] = None,
        global_resources: Optional[Dict[str, int]] = None,
        reaction_costs: Optional[Dict[int, Dict[str, int]]] = None,
    ):
        super().__init__(lattice, diffusion, seed)
        self.reactions = reactions

        # Validate that all species in reactions exist in lattice
        self._validate_reactions()

        # Precompute stoichiometry matrix for efficiency
        # Shape: (n_reactions, n_species)
        self._stoichiometry = self._build_stoichiometry_matrix()

        # Site-gated reactions: precompute masks
        self._reaction_masks = self._build_reaction_masks(reaction_sites)

        # Global resource counters
        self.global_resources = dict(global_resources) if global_resources else {}
        self._reaction_costs = dict(reaction_costs) if reaction_costs else {}

        # Precompute reactant index arrays for Numba kernels
        self._reactant_indices_flat, self._reactant_stoich_flat, self._reactant_offsets = \
            self._build_reactant_arrays()

    def _validate_reactions(self) -> None:
        """Validate that all reaction species exist in the lattice."""
        lattice_species = set(self.lattice.species)
        for rxn in self.reactions:
            for species in rxn.reactants + rxn.products:
                if species not in lattice_species:
                    raise ValueError(
                        f"Species '{species}' in reaction '{rxn}' "
                        f"not found in lattice species: {self.lattice.species}"
                    )

    def _build_reaction_masks(
        self,
        reaction_sites: Optional[Dict[int, Union[str, List[str]]]] = None,
    ) -> np.ndarray:
        """
        Build site-type masks for each reaction.

        Returns
        -------
        np.ndarray
            Shape (n_reactions, nx, ny) float64 masks (1.0 or 0.0).
        """
        nx, ny = self.lattice.nx, self.lattice.ny
        n_reactions = len(self.reactions)
        masks = np.ones((n_reactions, nx, ny), dtype=np.float64)

        if not reaction_sites:
            return masks

        # Validate: site types must be configured on lattice
        if self.lattice.site_types is None:
            raise ValueError(
                "reaction_sites requires lattice site types to be configured "
                "via lattice.set_site_type()"
            )

        for r_idx, site_constraint in reaction_sites.items():
            if isinstance(site_constraint, str):
                site_constraint = [site_constraint]

            mask = np.zeros((nx, ny), dtype=np.float64)
            for st in site_constraint:
                type_id = self.lattice.site_type_id(st)
                mask[self.lattice.site_types == type_id] = 1.0
            masks[r_idx] = mask

        return masks

    def _build_stoichiometry_matrix(self) -> np.ndarray:
        """
        Build stoichiometry matrix for efficient reaction application.

        Returns
        -------
        np.ndarray
            Shape (n_reactions, n_species) where S[r, s] is the net change
            in species s when reaction r fires once.
        """
        n_reactions = len(self.reactions)
        n_species = len(self.lattice.species)
        S = np.zeros((n_reactions, n_species), dtype=np.int32)

        for r, rxn in enumerate(self.reactions):
            for species in rxn.reactants:
                s = self.lattice.species_index(species)
                S[r, s] -= 1
            for species in rxn.products:
                s = self.lattice.species_index(species)
                S[r, s] += 1

        return S

    def _build_reactant_arrays(self):
        """Build flattened reactant index/stoich arrays for Numba kernels."""
        from collections import Counter
        indices = []
        stoichs = []
        offsets = [0]

        for rxn in self.reactions:
            reactant_counts = Counter(rxn.reactants)
            for species, stoich in reactant_counts.items():
                indices.append(self.lattice.species_index(species))
                stoichs.append(stoich)
            offsets.append(len(indices))

        return (
            np.array(indices, dtype=np.int32),
            np.array(stoichs, dtype=np.int32),
            np.array(offsets, dtype=np.int32),
        )

    def warmup(self) -> None:
        """Trigger Numba JIT compilation on small dummy arrays."""
        if not HAS_NUMBA or not self.reactions:
            return
        n_species = len(self.lattice.species)
        n_reactions = len(self.reactions)
        dummy_counts = np.ones((n_species, 2, 2), dtype=np.int32)
        dummy_firings = np.ones((n_reactions, 2, 2), dtype=np.int32)
        dummy_stoich = self._stoichiometry.copy()
        _apply_stoichiometry_numba(
            dummy_counts, dummy_stoich, dummy_firings,
            self._reactant_indices_flat, self._reactant_stoich_flat,
            self._reactant_offsets,
        )

    def get_max_timestep(self) -> float:
        """
        Calculate maximum stable timestep.

        The stability condition for diffusion is: q = D*dt/λ² ≤ 0.5
        Therefore: dt ≤ 0.5 * λ² / D_max

        For reactions, we want the expected number of reactions per site
        to be small enough for the Poisson approximation to be valid.
        This is generally satisfied if a₀·dt << 1, but we don't enforce
        this strictly here (user should monitor).

        Returns
        -------
        float
            Maximum stable timestep in seconds (based on diffusion).
        """
        D_max = self._diffusion_max
        if D_max == 0:
            # No diffusion - return a large timestep, reactions will limit
            return 1.0
        return 0.5 * self.lattice.spacing**2 / D_max

    def step(self, dt: float) -> None:
        """
        Perform one full RDME timestep.

        Applies operator splitting:
        1. X-direction diffusion sweep
        2. Y-direction diffusion sweep
        3. Reaction step at each site

        Parameters
        ----------
        dt : float
            Timestep duration in seconds.

        Raises
        ------
        ValueError
            If dt exceeds the diffusion stability limit.
        """
        # Validate timestep against diffusion stability
        validate_timestep(dt, self.diffusion, self.lattice.spacing)

        # Diffusion sweeps
        self._diffusion_sweep_x(dt)
        self._diffusion_sweep_y(dt)

        # Reaction step
        self._reaction_step(dt)

    def _reaction_step(self, dt: float) -> None:
        """
        Perform reaction step at each site using tau-leaping.

        For each site:
        1. Compute propensity for each reaction
        2. Sample number of firings from Poisson(a_j * dt)
        3. Limit firings to available reactants (conservation)
        4. Apply stoichiometry changes

        This is a vectorized implementation that processes all sites at once.
        """
        if not self.reactions:
            return

        nx, ny = self.lattice.nx, self.lattice.ny
        n_reactions = len(self.reactions)

        # Compute propensities for all reactions at all sites
        # Shape: (n_reactions, nx, ny)
        propensities = self._compute_propensities()

        # Gate by depleted global resources: zero out propensities for
        # reactions whose required resource is at zero
        if self._reaction_costs and self.global_resources:
            for r_idx, costs in self._reaction_costs.items():
                for resource, _cost in costs.items():
                    if self.global_resources.get(resource, 0) <= 0:
                        propensities[r_idx] = 0.0

        # Sample number of reaction firings at each site
        # Shape: (n_reactions, nx, ny)
        n_firings = self.rng.poisson(propensities * dt).astype(np.int32)

        if HAS_NUMBA and n_reactions > 0:
            # JIT path: limit firings and apply stoichiometry in compiled code
            _apply_stoichiometry_numba(
                self.lattice.counts, self._stoichiometry, n_firings,
                self._reactant_indices_flat, self._reactant_stoich_flat,
                self._reactant_offsets,
            )
        else:
            # Pure-NumPy path
            for r in range(n_reactions):
                firings_r = n_firings[r, :, :].copy()

                if np.sum(firings_r) == 0:
                    continue

                firings_r = self._limit_firings(r, firings_r)
                n_firings[r, :, :] = firings_r

                for s, delta in enumerate(self._stoichiometry[r, :]):
                    if delta != 0:
                        self.lattice.counts[s, :, :] += delta * firings_r

        # Consume global resources with proportional limiting
        if self._reaction_costs and self.global_resources:
            self._consume_resources(n_firings)

    def _consume_resources(self, n_firings: np.ndarray) -> None:
        """
        Consume global resources based on reaction firings.

        If total cost exceeds available resource, scale firings proportionally
        (floor) and revert excess stoichiometry changes.
        """
        # Group reactions by resource
        resource_reactions: Dict[str, list] = {}
        for r_idx, costs in self._reaction_costs.items():
            for resource, cost in costs.items():
                if resource not in resource_reactions:
                    resource_reactions[resource] = []
                resource_reactions[resource].append((r_idx, cost))

        for resource, rxn_costs in resource_reactions.items():
            available = self.global_resources.get(resource, 0)
            if available <= 0:
                continue

            # Compute total cost
            total_cost = 0
            for r_idx, cost in rxn_costs:
                total_cost += int(np.sum(n_firings[r_idx])) * cost

            if total_cost <= available:
                # Enough resource — just decrement
                self.global_resources[resource] = available - total_cost
            else:
                # Proportional limiting: scale down firings
                ratio = available / total_cost if total_cost > 0 else 0.0

                # Revert stoichiometry for affected reactions, then re-apply scaled
                actual_cost = 0
                for r_idx, cost in rxn_costs:
                    old_firings = n_firings[r_idx].copy()
                    new_firings = np.floor(old_firings * ratio).astype(np.int32)
                    delta_firings = new_firings - old_firings  # negative

                    for s, delta in enumerate(self._stoichiometry[r_idx, :]):
                        if delta != 0:
                            self.lattice.counts[s, :, :] += delta * delta_firings

                    actual_cost += int(np.sum(new_firings)) * cost

                self.global_resources[resource] = max(0, available - actual_cost)

    def _limit_firings(self, reaction_idx: int, firings: np.ndarray) -> np.ndarray:
        """
        Limit reaction firings to available reactants.

        This ensures we don't consume more reactants than available,
        preserving particle conservation.

        Parameters
        ----------
        reaction_idx : int
            Index of the reaction.
        firings : np.ndarray
            Proposed number of firings at each site.

        Returns
        -------
        np.ndarray
            Limited number of firings.
        """
        rxn = self.reactions[reaction_idx]

        if rxn.order == 0:
            # Production reaction - no reactants to limit
            return firings

        # Count how many of each reactant species are consumed per firing
        from collections import Counter
        reactant_counts = Counter(rxn.reactants)

        # For each reactant, compute max possible firings
        for species, stoich in reactant_counts.items():
            s_idx = self.lattice.species_index(species)
            available = self.lattice.counts[s_idx, :, :]
            max_firings = available // stoich  # Integer division
            firings = np.minimum(firings, max_firings)

        return firings

    def _compute_propensities(self) -> np.ndarray:
        """
        Compute propensity for each reaction at each site.

        Returns
        -------
        np.ndarray
            Shape (n_reactions, nx, ny) with propensity values.
        """
        nx, ny = self.lattice.nx, self.lattice.ny
        n_reactions = len(self.reactions)
        propensities = np.zeros((n_reactions, nx, ny), dtype=np.float64)

        for r, rxn in enumerate(self.reactions):
            propensities[r, :, :] = self._propensity_for_reaction(rxn)
            # Apply site-type mask
            propensities[r] *= self._reaction_masks[r]

        return propensities

    def _propensity_for_reaction(self, rxn: Reaction) -> np.ndarray:
        """
        Compute propensity for a single reaction at all sites.

        Implements the standard stochastic propensity formulas:
        - Order 0 (∅ → A):         a = k
        - Order 1 (A → ...):       a = k * n_A
        - Order 2 (A + B → ...):   a = k * n_A * n_B
        - Order 2 (A + A → ...):   a = k * n_A * (n_A - 1) / 2

        Parameters
        ----------
        rxn : Reaction
            The reaction to compute propensity for.

        Returns
        -------
        np.ndarray
            Shape (nx, ny) with propensity at each site.
        """
        nx, ny = self.lattice.nx, self.lattice.ny

        if rxn.order == 0:
            # Zeroth order: ∅ → products
            return np.full((nx, ny), rxn.k, dtype=np.float64)

        elif rxn.order == 1:
            # First order: A → products
            species = rxn.reactants[0]
            s_idx = self.lattice.species_index(species)
            n_A = self.lattice.counts[s_idx, :, :].astype(np.float64)
            return rxn.k * n_A

        elif rxn.order == 2:
            species1 = rxn.reactants[0]
            species2 = rxn.reactants[1]

            if species1 == species2:
                # Self-reaction: A + A → products
                # Propensity = k * n_A * (n_A - 1) / 2
                s_idx = self.lattice.species_index(species1)
                n_A = self.lattice.counts[s_idx, :, :].astype(np.float64)
                return rxn.k * n_A * (n_A - 1) / 2
            else:
                # Different species: A + B → products
                # Propensity = k * n_A * n_B
                s1_idx = self.lattice.species_index(species1)
                s2_idx = self.lattice.species_index(species2)
                n_A = self.lattice.counts[s1_idx, :, :].astype(np.float64)
                n_B = self.lattice.counts[s2_idx, :, :].astype(np.float64)
                return rxn.k * n_A * n_B

        else:
            # Higher order reactions (rare in practice)
            # General formula: k * product of (n_i choose r_i) * r_i!
            # For simplicity, we support up to order 2 with the efficient formulas
            # For higher orders, use combinatorial formula
            propensity = np.full((nx, ny), rxn.k, dtype=np.float64)

            # Count occurrences of each reactant
            from collections import Counter
            reactant_counts = Counter(rxn.reactants)

            for species, count in reactant_counts.items():
                s_idx = self.lattice.species_index(species)
                n = self.lattice.counts[s_idx, :, :].astype(np.float64)

                if count == 1:
                    propensity *= n
                elif count == 2:
                    propensity *= n * (n - 1) / 2
                else:
                    # General: n! / (n - count)! / count!
                    # = n * (n-1) * ... * (n-count+1) / count!
                    from math import factorial
                    term = np.ones_like(n)
                    for i in range(count):
                        term *= (n - i)
                    term /= factorial(count)
                    propensity *= term

            return propensity

    def step_diffusion_only(self, dt: float) -> None:
        """
        Perform only the diffusion step (useful for testing).

        Parameters
        ----------
        dt : float
            Timestep duration in seconds.
        """
        validate_timestep(dt, self.diffusion, self.lattice.spacing)
        self._diffusion_sweep_x(dt)
        self._diffusion_sweep_y(dt)

    def step_reaction_only(self, dt: float) -> None:
        """
        Perform only the reaction step (useful for testing).

        Parameters
        ----------
        dt : float
            Timestep duration in seconds.
        """
        self._reaction_step(dt)
