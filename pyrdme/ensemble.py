"""
Ensemble simulation runner for parallel parameter sweeps.

Provides run_ensemble() for distributing independent simulation runs
across CPU cores, make_param_grid() for generating parameter grids,
and SweepState for incremental checkpointing and crash recovery.
"""

import json
import os
import sys
import itertools
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


def _result_key(params: dict) -> str:
    """
    Build a deterministic filename-safe key from simulation parameters.

    Sorts parameter names alphabetically and joins as key=value pairs.
    Internal keys (starting with '_') are excluded.
    """
    items = sorted(
        (k, v) for k, v in params.items()
        if not k.startswith('_')
    )
    return '_'.join(f'{k}={v}' for k, v in items)


class SweepState:
    """
    Manages incremental persistence for parameter sweeps.

    Saves each simulation result to disk immediately upon completion,
    enabling crash recovery by skipping already-completed runs on resume.

    Parameters
    ----------
    output_dir : str or Path
        Directory for sweep outputs. Created if it doesn't exist.
    """

    METADATA_FILE = 'sweep_meta.json'
    RUNS_DIR = 'runs'

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.runs_dir = self.output_dir / self.RUNS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

    def save_metadata(self, **config):
        """
        Write sweep configuration to sweep_meta.json.

        Parameters
        ----------
        **config : Any
            Arbitrary configuration (axes, seeds, t_max, etc.).
            Values must be JSON-serializable.
        """
        meta_path = self.output_dir / self.METADATA_FILE
        with open(meta_path, 'w') as f:
            json.dump(config, f, indent=2, default=_json_default)

    def load_metadata(self) -> dict:
        """Read and return sweep_meta.json."""
        meta_path = self.output_dir / self.METADATA_FILE
        with open(meta_path, 'r') as f:
            return json.load(f)

    def save_result(self, params: dict, result: dict):
        """
        Atomically write one simulation result to the runs/ directory.

        Writes to a .tmp file first, then renames for crash safety.
        """
        key = _result_key(params)
        final_path = self.runs_dir / f'{key}.json'
        tmp_path = self.runs_dir / f'{key}.json.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(result, f, default=_json_default)
        os.replace(tmp_path, final_path)

    def load_completed(self) -> Tuple[List[dict], Set[str]]:
        """
        Scan runs/ directory for completed results.

        Returns
        -------
        results : List[dict]
            All completed result dicts.
        completed_keys : Set[str]
            Set of result keys (filename stems) for fast lookup.
        """
        results = []
        completed_keys = set()
        for path in sorted(self.runs_dir.glob('*.json')):
            try:
                with open(path, 'r') as f:
                    result = json.load(f)
                results.append(result)
                completed_keys.add(path.stem)
            except (json.JSONDecodeError, OSError):
                # Skip corrupt/partial files
                continue
        return results, completed_keys

    def remaining(self, param_grid: List[dict]) -> List[dict]:
        """
        Filter param_grid to only entries not yet completed.

        Parameters
        ----------
        param_grid : List[dict]
            Full parameter grid.

        Returns
        -------
        List[dict]
            Subset of param_grid with no corresponding result file.
        """
        _, completed_keys = self.load_completed()
        return [
            p for p in param_grid
            if _result_key(p) not in completed_keys
        ]


def _json_default(obj):
    """JSON serializer for numpy types and other non-standard objects."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
    sweep_state: Optional[SweepState] = None,
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
    sweep_state : SweepState, optional
        If provided, each result is saved to disk immediately upon
        completion for crash recovery.

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
            if sweep_state is not None:
                sweep_state.save_result(param_grid[idx], results[idx])
            n_done += 1
            if progress and n_done % max(1, n_total // 20) == 0:
                pct = 100.0 * n_done / n_total
                print(f"\rEnsemble progress: {n_done}/{n_total} ({pct:.0f}%)",
                      end='', file=sys.stderr, flush=True)

    if progress and n_total > 0:
        print(f"\rEnsemble progress: {n_total}/{n_total} (100%)",
              file=sys.stderr, flush=True)

    return results
