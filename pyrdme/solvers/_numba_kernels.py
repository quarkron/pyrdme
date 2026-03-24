"""
Numba JIT-compiled kernels for the RDME solver hot path.

These functions handle the deterministic array manipulation (firing limitation,
stoichiometry application) while random variate generation stays in NumPy
(preserving Generator/PCG64 RNG quality).

If Numba is not installed, HAS_NUMBA is False and the pure-NumPy path is used.
"""

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # No-op decorator fallback so functions are still callable
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

import numpy as np


@njit(cache=True)
def _limit_firings_numba(
    counts: np.ndarray,
    reactant_indices: np.ndarray,
    reactant_stoich: np.ndarray,
    firings: np.ndarray,
) -> np.ndarray:
    """
    Limit reaction firings to available reactants (per-site).

    Parameters
    ----------
    counts : ndarray, shape (n_species, nx, ny)
        Current particle counts.
    reactant_indices : ndarray, shape (n_reactants,)
        Species indices of reactants for this reaction.
    reactant_stoich : ndarray, shape (n_reactants,)
        Stoichiometric coefficient (consumption) for each reactant.
    firings : ndarray, shape (nx, ny)
        Proposed firings at each site.

    Returns
    -------
    ndarray, shape (nx, ny)
        Limited firings.
    """
    nx = firings.shape[0]
    ny = firings.shape[1]
    result = firings.copy()
    for ri in range(len(reactant_indices)):
        s_idx = reactant_indices[ri]
        stoich = reactant_stoich[ri]
        for i in range(nx):
            for j in range(ny):
                max_f = counts[s_idx, i, j] // stoich
                if result[i, j] > max_f:
                    result[i, j] = max_f
    return result


@njit(cache=True)
def _apply_stoichiometry_numba(
    counts: np.ndarray,
    stoichiometry: np.ndarray,
    all_firings: np.ndarray,
    reactant_indices_flat: np.ndarray,
    reactant_stoich_flat: np.ndarray,
    reactant_offsets: np.ndarray,
) -> np.ndarray:
    """
    Apply firing limitation and stoichiometry updates for all reactions.

    Parameters
    ----------
    counts : ndarray, shape (n_species, nx, ny)
        Current particle counts (modified in-place).
    stoichiometry : ndarray, shape (n_reactions, n_species)
        Net stoichiometry matrix.
    all_firings : ndarray, shape (n_reactions, nx, ny)
        Proposed firings (modified in-place with limited values).
    reactant_indices_flat : ndarray
        Flattened array of reactant species indices for all reactions.
    reactant_stoich_flat : ndarray
        Flattened array of reactant stoichiometric coefficients.
    reactant_offsets : ndarray, shape (n_reactions + 1,)
        Offsets into the flat arrays for each reaction.

    Returns
    -------
    ndarray, shape (n_reactions, nx, ny)
        The limited firings (same array as all_firings, modified in-place).
    """
    n_reactions = stoichiometry.shape[0]
    n_species = stoichiometry.shape[1]
    nx = counts.shape[1]
    ny = counts.shape[2]

    for r in range(n_reactions):
        # Check if any firings for this reaction
        has_firings = False
        for i in range(nx):
            for j in range(ny):
                if all_firings[r, i, j] > 0:
                    has_firings = True
                    break
            if has_firings:
                break

        if not has_firings:
            continue

        # Limit firings based on available reactants
        start = reactant_offsets[r]
        end = reactant_offsets[r + 1]
        for ri in range(start, end):
            s_idx = reactant_indices_flat[ri]
            stoich = reactant_stoich_flat[ri]
            for i in range(nx):
                for j in range(ny):
                    max_f = counts[s_idx, i, j] // stoich
                    if all_firings[r, i, j] > max_f:
                        all_firings[r, i, j] = max_f

        # Apply stoichiometry
        for s in range(n_species):
            delta = stoichiometry[r, s]
            if delta != 0:
                for i in range(nx):
                    for j in range(ny):
                        counts[s, i, j] += delta * all_firings[r, i, j]

    return all_firings
