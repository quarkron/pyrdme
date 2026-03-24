"""
Ensemble simulation runner for parallel parameter sweeps.

Provides run_ensemble() for distributing independent simulation runs
across CPU cores, and make_param_grid() for generating parameter grids.
"""

import sys
import itertools
from typing import Callable, Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed


def make_param_grid(seeds=None, **axes) -> List[Dict[str, Any]]:
    """
    Generate a cartesian product parameter grid.

    Parameters
    ----------
    seeds : iterable, optional
        Range of random seeds. Each parameter combination is repeated
        for every seed.
    **axes : lists
        Named parameter axes. Each key maps to a list of values.

    Returns
    -------
    List[Dict[str, Any]]
        One dict per simulation run.

    Examples
    --------
    >>> grid = make_param_grid(
    ...     N_RNAP=[2, 5, 10],
    ...     N_Ribo=[2, 5, 10],
    ...     N_ATP=[1000],
    ...     seeds=range(200),
    ... )
    >>> len(grid)  # 3 * 3 * 1 * 200 = 1800
    1800
    """
    if seeds is None:
        seeds = [None]

    axis_names = list(axes.keys())
    axis_values = list(axes.values())

    grid = []
    for combo in itertools.product(*axis_values):
        params = dict(zip(axis_names, combo))
        for seed in seeds:
            entry = dict(params)
            entry['seed'] = seed
            grid.append(entry)

    return grid


def run_ensemble(
    sim_fn: Callable[..., dict],
    param_grid: List[dict],
    n_workers: Optional[int] = None,
    progress: bool = True,
) -> List[dict]:
    """
    Run an ensemble of simulations in parallel.

    Parameters
    ----------
    sim_fn : Callable[..., dict]
        Function that runs one simulation and returns a summary dict.
        Called as sim_fn(**params) for each entry in param_grid.
    param_grid : List[dict]
        List of keyword argument dicts for sim_fn.
    n_workers : int, optional
        Number of worker processes. Defaults to os.cpu_count().
    progress : bool
        If True, print progress to stderr.

    Returns
    -------
    List[dict]
        Results in the same order as param_grid.
    """
    n_total = len(param_grid)
    if n_total == 0:
        return []

    results = [None] * n_total

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs, tracking their index
        future_to_idx = {}
        for idx, params in enumerate(param_grid):
            future = executor.submit(sim_fn, **params)
            future_to_idx[future] = idx

        n_done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            n_done += 1
            if progress and n_done % max(1, n_total // 20) == 0:
                pct = 100.0 * n_done / n_total
                print(f"\rEnsemble progress: {n_done}/{n_total} ({pct:.0f}%)",
                      end='', file=sys.stderr, flush=True)

    if progress and n_total > 0:
        print(f"\rEnsemble progress: {n_total}/{n_total} (100%)",
              file=sys.stderr, flush=True)

    return results
