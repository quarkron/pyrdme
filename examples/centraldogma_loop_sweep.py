#!/usr/bin/env python3
"""

Runs parameter sweeps for the autocatalytic loop collapse experiments
and saves all results to the results/ folder.

Usage:
    python centraldogma_loop_sweep.py [--n-grid N] [--n-seeds M] [--output-dir DIR]
"""

import argparse
import pickle
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from pyrdme import Lattice2D, Reaction, MPDSolver
from pyrdme.ensemble import make_param_grid


# --- Dimensionless units ---
# Length: λ = lattice spacing = 1
# Time:   τ = λ² / D_mRNA = 1  (one mRNA diffusion time across a site)
#
# Physical reference (for converting back):
#   λ_phys = 1 μm,  D_mRNA_phys = 5e-13 m²/s  →  τ_phys = 2.0 s

# --- Species ---
SPECIES = ['P', 'R', 'mP', 'mR']

# --- Diffusion (dimensionless, relative to D_mRNA = 1) ---
DIFFUSION = {
    'P':  0.2,   # RNAP:     D_phys / D_mRNA = 1e-13 / 5e-13
    'R':  0.1,   # Ribosome: D_phys / D_mRNA = 0.5e-13 / 5e-13
    'mP': 1.0,   # mRNA (reference)
    'mR': 1.0,
}

# --- Rate constants (dimensionless, in units of 1/τ) ---
k_tx    = 0.1     # transcription rate per RNAP at gene locus
k_tl    = 0.004   # translation rate per ribosome-mRNA pair
k_deg_m = 0.01    # mRNA degradation
k_deg_p = 0.001   # protein degradation

# --- Reactions ---
R0 = Reaction(['P'], ['P', 'mP'], k=k_tx)
R1 = Reaction(['P'], ['P', 'mR'], k=k_tx)
R2 = Reaction(['R', 'mP'], ['R', 'mP', 'P'], k=k_tl)
R3 = Reaction(['R', 'mR'], ['R', 'mR', 'R'], k=k_tl)
R4 = Reaction(['mP'], [], k=k_deg_m)
R5 = Reaction(['mR'], [], k=k_deg_m)
R6 = Reaction(['P'], [], k=k_deg_p)
R7 = Reaction(['R'], [], k=k_deg_p)

REACTIONS = [R0, R1, R2, R3, R4, R5, R6, R7]

REACTION_SITES = {
    0: 'rnap_gene',
    1: 'ribo_gene',
}

REACTION_COSTS = {
    0: {'ATP': 1},
    1: {'ATP': 1},
    2: {'ATP': 1},
    3: {'ATP': 1},
}


def make_lattice(N_RNAP=20, N_Ribo=20):
    """Create a 10×10 lattice with gene loci and initial particles."""
    lattice = Lattice2D(nx=10, ny=10, species=SPECIES, spacing=1.0)
    lattice.set_site_type((slice(1, 3), slice(1, 3)), 'rnap_gene')
    lattice.set_site_type((slice(7, 9), slice(7, 9)), 'ribo_gene')
    lattice.add_particles('P', count=N_RNAP)
    lattice.add_particles('R', count=N_Ribo)
    return lattice


def _check_cycles(lattice, solver, idx_P, idx_R, idx_mP, idx_mR):
    """Return (cycle1_active, cycle2_active, counts tuple)."""
    P = int(np.sum(lattice.counts[idx_P]))
    R = int(np.sum(lattice.counts[idx_R]))
    mP = int(np.sum(lattice.counts[idx_mP]))
    mR = int(np.sum(lattice.counts[idx_mR]))
    ATP = solver.global_resources['ATP']

    cycle1 = (P > 0) and (mP > 0) and (R > 0) and (ATP > 0)
    cycle2 = (R > 0) and (mR > 0) and (ATP > 0)
    return cycle1, cycle2, (P, R, mP, mR, ATP)


def run_single_sim(N_RNAP=20, N_Ribo=20, N_ATP=1000, seed=None,
                    t_max=50000.0, dt=None, return_trajectory=False):
    """
    Run one spore dormancy simulation with deterministic PNR detection.

    Three deterministic PNRs (points of no return):

    - PNR-1 (ribo_depletion):  R = 0
      Ribosome absorbing wall — autocatalytic, no recovery possible.
      Walking dead: RNAP continues futile transcription until P→0 or ATP→0.

    - PNR-2 (rnap_deadlock):   P = 0 ∧ mP = 0
      Mutual dependency deadlock — neither P nor mP can be produced.
      Walking dead: ribosomes translate remaining mR until cycle 2 breaks.

    - ATP = 0: immediate functional death (not a PNR — no transient activity).

    The first PNR reached is recorded. Simulation continues through the
    walking-dead phase until functional death.
    """
    lattice = make_lattice(N_RNAP, N_Ribo)
    solver = MPDSolver(
        lattice, REACTIONS, DIFFUSION, seed=seed,
        reaction_sites=REACTION_SITES,
        global_resources={'ATP': N_ATP},
        reaction_costs=REACTION_COSTS,
    )

    if dt is None:
        dt = solver.get_max_timestep() * 0.9

    idx_P = lattice.species_index('P')
    idx_R = lattice.species_index('R')
    idx_mP = lattice.species_index('mP')
    idx_mR = lattice.species_index('mR')

    def _make_result(t, death_mode, t_pnr, pnr_type, trajectory):
        result = {
            't_death': t,
            'death_mode': death_mode,
            'pnr_type': pnr_type,
            't_pnr': t_pnr,
            'dt_walking_dead': (t - t_pnr) if t_pnr is not None else 0.0,
            'N_RNAP': N_RNAP, 'N_Ribo': N_Ribo, 'N_ATP': N_ATP, 'seed': seed,
        }
        if return_trajectory:
            result['trajectory'] = trajectory
        return result

    t = 0.0
    t_pnr = None
    pnr_type = None
    trajectory = [] if return_trajectory else None

    # Record initial state
    if return_trajectory:
        _, _, counts = _check_cycles(lattice, solver, idx_P, idx_R, idx_mP, idx_mR)
        P, R, mP, mR, ATP = counts
        trajectory.append({'t': 0.0, 'P': P, 'R': R, 'mP': mP, 'mR': mR, 'ATP': ATP})

    # Immediate death: if R=0 or P=0 or ATP=0 at t=0
    _, _, counts = _check_cycles(lattice, solver, idx_P, idx_R, idx_mP, idx_mR)
    P, R, mP, mR, ATP = counts
    if R == 0:
        return _make_result(0.0, 'ribo_collapse', 0.0, 'ribo_depletion', trajectory)
    if P == 0:
        return _make_result(0.0, 'cycle_collapse', 0.0, 'rnap_deadlock', trajectory)
    if ATP <= 0:
        return _make_result(0.0, 'atp_depletion', None, None, trajectory)

    while t < t_max:
        solver.step(dt)
        t += dt

        c1, c2, counts = _check_cycles(lattice, solver, idx_P, idx_R, idx_mP, idx_mR)
        P, R, mP, mR, ATP = counts

        if return_trajectory and (len(trajectory) == 0 or t - trajectory[-1]['t'] >= dt * 10):
            trajectory.append({'t': t, 'P': P, 'R': R, 'mP': mP, 'mR': mR, 'ATP': ATP})

        # --- ATP = 0: immediate functional death ---
        if ATP <= 0:
            return _make_result(t, 'atp_depletion', t_pnr, pnr_type, trajectory)

        # --- Deterministic PNRs ---
        if R == 0 and pnr_type is None:
            t_pnr = t
            pnr_type = 'ribo_depletion'

        if (P == 0) and (mP == 0) and pnr_type is None:
            t_pnr = t
            pnr_type = 'rnap_deadlock'

        # --- Functional death ---
        if pnr_type == 'ribo_depletion' and P == 0:
            return _make_result(t, 'ribo_collapse', t_pnr, pnr_type, trajectory)

        if pnr_type == 'rnap_deadlock' and not c2:
            return _make_result(t, 'cycle_collapse', t_pnr, pnr_type, trajectory)

    return _make_result(t_max, 'survived', t_pnr, pnr_type, trajectory)


def run_ensemble_with_progress(func, param_grid):
    """Run ensemble with detailed progress reporting."""
    results = []
    total = len(param_grid)

    print(f"\nRunning {total} simulations...")
    print("-" * 60)

    # Track statistics for progress reporting
    mode_counts = {'ribo_collapse': 0, 'cycle_collapse': 0, 'atp_depletion': 0, 'survived': 0}

    for i, params in enumerate(param_grid, 1):
        result = func(**params)
        results.append(result)

        # Update mode counts
        mode = result['death_mode']
        if mode in mode_counts:
            mode_counts[mode] += 1

        # Print progress every 10% or for first/last few
        if i % max(1, total // 10) == 0 or i <= 5 or i >= total - 5:
            pct = 100 * i / total
            print(f"[{i:>5}/{total}] {pct:>5.1f}% | "
                  f"Last: P={params['N_RNAP']:>3} R={params['N_Ribo']:>3} → {mode:15s} "
                  f"t={result['t_death']:>6.0f}τ", flush=True)
        elif i % max(1, total // 50) == 0:
            # More frequent but less verbose progress
            pct = 100 * i / total
            sys.stdout.write(f"\r[{i:>5}/{total}] {pct:>5.1f}%")
            sys.stdout.flush()

    print()
    print("-" * 60)
    print("Simulation complete!")
    print("\nDeath mode distribution:")
    for mode, count in mode_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {mode:20s}: {count:>5} ({pct:>5.1f}%)")
    print()

    return results


def aggregate_results(results, x_key, y_key, x_axis, y_axis):
    """Aggregate ensemble results into 2D grids for plotting."""
    nx, ny = len(x_axis), len(y_axis)
    x_map = {v: i for i, v in enumerate(x_axis)}
    y_map = {v: i for i, v in enumerate(y_axis)}

    mode_names = ['ribo_collapse', 'cycle_collapse', 'atp_depletion', 'survived']
    pnr_names = ['ribo_depletion', 'rnap_deadlock', 'none']
    death_counts = {mode: np.zeros((nx, ny)) for mode in mode_names}
    pnr_counts = {p: np.zeros((nx, ny)) for p in pnr_names}
    t_death_sum = np.zeros((nx, ny))
    t_death_n = np.zeros((nx, ny))
    wd_sum = np.zeros((nx, ny))
    wd_n = np.zeros((nx, ny))

    for r in results:
        xi = x_map.get(r[x_key])
        yi = y_map.get(r[y_key])
        if xi is None or yi is None:
            continue
        mode = r['death_mode']
        if mode in death_counts:
            death_counts[mode][xi, yi] += 1
        pnr = r.get('pnr_type') or 'none'
        if pnr in pnr_counts:
            pnr_counts[pnr][xi, yi] += 1
        t_death_sum[xi, yi] += r['t_death']
        t_death_n[xi, yi] += 1
        if r['dt_walking_dead'] > 0:
            wd_sum[xi, yi] += r['dt_walking_dead']
            wd_n[xi, yi] += 1

    # Mode fractions
    total = t_death_n.copy()
    total[total == 0] = 1
    frac_ribo = death_counts['ribo_collapse'] / total
    frac_cycle = death_counts['cycle_collapse'] / total
    frac_atp = death_counts['atp_depletion'] / total
    frac_survived = death_counts['survived'] / total

    rgb = np.zeros((nx, ny, 3))
    c_ribo    = np.array([0x44, 0x77, 0xAA]) / 255.0
    c_cycle   = np.array([0xCC, 0x33, 0x11]) / 255.0
    c_atp     = np.array([0x22, 0x88, 0x33]) / 255.0
    c_survived = np.array([0.0, 0.0, 0.0])  # Black for survived
    for ch in range(3):
        rgb[:, :, ch] = (frac_ribo * c_ribo[ch]
                       + frac_cycle * c_cycle[ch]
                       + frac_atp * c_atp[ch]
                       + frac_survived * c_survived[ch])

    mean_t_death = np.divide(t_death_sum, t_death_n, where=t_death_n > 0,
                             out=np.zeros_like(t_death_sum))
    mean_wd = np.divide(wd_sum, wd_n, where=wd_n > 0,
                        out=np.zeros_like(wd_sum))

    mode_grid = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            counts = [death_counts[m][i, j] for m in mode_names]
            mode_grid[i, j] = np.argmax(counts)

    pnr_grid = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            counts = [pnr_counts[p][i, j] for p in pnr_names]
            pnr_grid[i, j] = np.argmax(counts)

    return rgb, mode_grid, mean_t_death, mean_wd, mode_names, pnr_grid, pnr_names


def plot_phase_diagrams(rgb, mode_grid, t_death, wd, mode_names, pnr_grid, pnr_names,
                        x_axis, y_axis, output_dir):
    """Generate and save phase diagram plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer 1: Continuous RGB Phase Diagram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower', aspect='auto',
              interpolation='nearest')
    ax.set_xticks(range(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_yticks(range(len(y_axis)))
    ax.set_yticklabels(y_axis)
    ax.set_xlabel('Initial RNAP')
    ax.set_ylabel('Initial Ribosome')
    ax.set_title('Loop Collapse Mode (N_RNAP × N_Ribo)')

    c_ribo   = (0x44/255, 0x77/255, 0xAA/255)
    c_cycle  = (0xCC/255, 0x33/255, 0x11/255)
    c_atp    = (0x22/255, 0x88/255, 0x33/255)
    c_surv   = (0.0, 0.0, 0.0)  # Black for survived
    legend_elements = [
        Patch(facecolor=c_ribo,  label='Ribo collapse'),
        Patch(facecolor=c_cycle, label='Cycle collapse'),
        Patch(facecolor=c_atp,   label='ATP depletion'),
        Patch(facecolor=c_surv,  label='Survived', edgecolor='white'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_diagram_rgb.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'phase_diagram_rgb.png'}")

    # Layer 2: Mean Cessation Time Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(t_death.T, origin='lower', aspect='auto', cmap='viridis',
                   interpolation='nearest')
    ax.set_xticks(range(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_yticks(range(len(y_axis)))
    ax.set_yticklabels(y_axis)
    ax.set_xlabel('Initial RNAP')
    ax.set_ylabel('Initial Ribosome')
    ax.set_title('Mean Cessation Time (N_RNAP × N_Ribo)')
    cbar = plt.colorbar(im, ax=ax, label=r'Mean $t_{death}$ ($\tau$)')
    plt.tight_layout()
    plt.savefig(output_dir / 'cessation_time_heatmap.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'cessation_time_heatmap.png'}")

    # Layer 3: Walking Dead Duration Overlay
    wd_masked = np.ma.masked_where(wd <= 0, wd)

    fig, ax = plt.subplots(figsize=(8, 6))
    bg = rgb.copy()
    bg = 0.3 * bg + 0.7
    ax.imshow(np.transpose(bg, (1, 0, 2)), origin='lower', aspect='auto',
              interpolation='nearest')
    im = ax.imshow(wd_masked.T, origin='lower', aspect='auto', cmap='Oranges',
                   interpolation='nearest')

    for i in range(wd_masked.shape[0]):
        for j in range(wd_masked.shape[1]):
            if not wd_masked.mask[i, j] if np.ndim(wd_masked.mask) > 0 else wd[i, j] > 0:
                marker = 'R' if pnr_grid[i, j] == 0 else 'P'
                ax.text(i, j, marker, ha='center', va='center', fontsize=7,
                        color='black', alpha=0.6)

    ax.set_xticks(range(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_yticks(range(len(y_axis)))
    ax.set_yticklabels(y_axis)
    ax.set_xlabel('Initial RNAP')
    ax.set_ylabel('Initial Ribosome')
    ax.set_title('Walking Dead Duration (both PNR pathways)')
    cbar = plt.colorbar(im, ax=ax, label=r'Mean $\Delta t_{wd}$ ($\tau$)')

    pnr_legend = [
        Line2D([0], [0], marker='$R$', color='w', markerfacecolor='black',
               markersize=10, label='PNR: ribo depletion (R=0)'),
        Line2D([0], [0], marker='$P$', color='w', markerfacecolor='black',
               markersize=10, label='PNR: RNAP deadlock (P=0,mP=0)'),
    ]
    ax.legend(handles=pnr_legend, loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'walking_dead_duration.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'walking_dead_duration.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Run spore dormancy parameter sweep and save results'
    )
    parser.add_argument('--n-grid', type=int, default=8,
                        help='Grid points per axis (default: 8)')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Seeds per parameter point (default: 10)')
    parser.add_argument('--t-max', type=float, default=50000.0,
                        help='Max simulation time in τ (default: 50000)')
    parser.add_argument('--output-dir', type=str, default='../results',
                        help='Output directory for results (default: ../results)')
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'sweep_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Spore Dormancy Phase 1 - Parameter Sweep")
    print("=" * 60)
    print(f"Grid size: {args.n_grid} × {args.n_grid}")
    print(f"Seeds per point: {args.n_seeds}")
    print(f"T_MAX: {args.t_max:.0f} τ  ({args.t_max * 2:.0f} s physical)")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Generate axes
    RNAP_AXIS = np.unique(np.logspace(np.log10(2), np.log10(100), args.n_grid).astype(int))
    RIBO_AXIS = np.unique(np.logspace(np.log10(2), np.log10(100), args.n_grid).astype(int))
    ATP_AXIS = [1000]

    print(f"RNAP axis: {RNAP_AXIS.tolist()}")
    print(f"Ribo axis: {RIBO_AXIS.tolist()}")
    print(f"Total runs: {len(RNAP_AXIS) * len(RIBO_AXIS) * args.n_seeds}")
    print()

    # Run parameter sweep
    grid = make_param_grid(
        N_RNAP=RNAP_AXIS.tolist(),
        N_Ribo=RIBO_AXIS.tolist(),
        N_ATP=ATP_AXIS,
        seeds=range(args.n_seeds),
        t_max=[args.t_max],
    )

    results = run_ensemble_with_progress(run_single_sim, grid)

    # Save raw results
    results_file = output_dir / 'results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'RNAP_AXIS': RNAP_AXIS,
            'RIBO_AXIS': RIBO_AXIS,
            'ATP_AXIS': ATP_AXIS,
            'n_seeds': args.n_seeds,
            't_max': args.t_max,
            'timestamp': timestamp,
        }, f)
    print(f"Saved raw results: {results_file}")

    # Aggregate and plot
    print("Generating phase diagrams...")
    rgb, mode_grid, t_death, wd, mode_names, pnr_grid, pnr_names = aggregate_results(
        results, 'N_RNAP', 'N_Ribo', RNAP_AXIS, RIBO_AXIS
    )

    plot_phase_diagrams(rgb, mode_grid, t_death, wd, mode_names, pnr_grid, pnr_names,
                        RNAP_AXIS, RIBO_AXIS, output_dir)

    # Save aggregated data
    aggregate_file = output_dir / 'aggregated.npz'
    np.savez(aggregate_file,
             rgb=rgb, mode_grid=mode_grid, t_death=t_death, wd=wd,
             pnr_grid=pnr_grid,
             RNAP_AXIS=RNAP_AXIS, RIBO_AXIS=RIBO_AXIS)
    print(f"Saved aggregated data: {aggregate_file}")

    # Save summary statistics
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Spore Dormancy Phase 1 - Parameter Sweep Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Grid size: {args.n_grid} × {args.n_grid}\n")
        f.write(f"Seeds per point: {args.n_seeds}\n")
        f.write(f"T_MAX: {args.t_max:.0f} τ\n")
        f.write(f"Total simulations: {len(results)}\n\n")

        f.write("Death mode counts:\n")
        for mode in ['ribo_collapse', 'cycle_collapse', 'atp_depletion', 'survived']:
            count = sum(1 for r in results if r['death_mode'] == mode)
            f.write(f"  {mode}: {count} ({100*count/len(results):.1f}%)\n")

        f.write("\nPNR type counts:\n")
        for pnr in ['ribo_depletion', 'rnap_deadlock', None]:
            count = sum(1 for r in results if r.get('pnr_type') == pnr)
            f.write(f"  {pnr}: {count} ({100*count/len(results):.1f}%)\n")

        f.write(f"\nMean cessation time: {np.mean([r['t_death'] for r in results]):.1f} τ\n")
        f.write(f"Median cessation time: {np.median([r['t_death'] for r in results]):.1f} τ\n")

        wd_times = [r['dt_walking_dead'] for r in results if r['dt_walking_dead'] > 0]
        if wd_times:
            f.write(f"\nWalking dead statistics ({len(wd_times)} trajectories):\n")
            f.write(f"  Mean duration: {np.mean(wd_times):.1f} τ\n")
            f.write(f"  Median duration: {np.median(wd_times):.1f} τ\n")
            f.write(f"  Max duration: {np.max(wd_times):.1f} τ\n")

    print(f"Saved summary: {summary_file}")
    print()
    print("=" * 60)
    print("Parameter sweep complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
