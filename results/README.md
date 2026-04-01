# Spore Dormancy Phase 1 - Results Directory

This directory contains results from parameter sweep simulations.

## Directory Structure

Each sweep run creates a timestamped subdirectory: `sweep_YYYYMMDD_HHMMSS/`

## Files Saved Per Run

### 1. Raw Data Files

#### `results.pkl` (Python pickle format)
Complete raw simulation results containing:
- `results`: List of all simulation outcomes with full details
- `RNAP_AXIS`: Array of N_RNAP values tested
- `RIBO_AXIS`: Array of N_Ribo values tested
- `ATP_AXIS`: Array of N_ATP values tested
- `n_seeds`: Number of random seeds per parameter point
- `t_max`: Maximum simulation time
- `timestamp`: Run timestamp

**Each result entry contains:**
- `t_death`: Time of functional death (τ)
- `death_mode`: 'ribo_collapse', 'cycle_collapse', 'atp_depletion', or 'survived'
- `pnr_type`: Point-of-no-return type ('ribo_depletion', 'rnap_deadlock', or None)
- `t_pnr`: Time when PNR was reached
- `dt_walking_dead`: Duration of walking-dead phase (t_death - t_pnr)
- `N_RNAP`, `N_Ribo`, `N_ATP`: Initial conditions
- `seed`: Random seed used

**Load with:**
```python
import pickle
with open('results.pkl', 'rb') as f:
    data = pickle.load(f)
results = data['results']
RNAP_AXIS = data['RNAP_AXIS']
```

#### `aggregated.npz` (NumPy compressed format)
Processed 2D grids for analysis and plotting:
- `rgb`: (nx, ny, 3) RGB phase diagram (weighted mode fractions)
- `mode_grid`: (nx, ny) Dominant death mode at each point
- `t_death`: (nx, ny) Mean cessation time at each point
- `wd`: (nx, ny) Mean walking-dead duration at each point
- `pnr_grid`: (nx, ny) Dominant PNR type at each point
- `RNAP_AXIS`: Parameter axis values
- `RIBO_AXIS`: Parameter axis values

**Load with:**
```python
import numpy as np
data = np.load('aggregated.npz')
rgb = data['rgb']
t_death = data['t_death']
RNAP_AXIS = data['RNAP_AXIS']
```

### 2. Visualizations (PNG, 300 DPI)

#### `phase_diagram_rgb.png`
Continuous RGB phase diagram showing:
- **Blue** (#4477AA): Ribo collapse pathway
- **Red** (#CC3311): Cycle collapse pathway
- **Green** (#228833): ATP depletion pathway
- **Black** (#000000): Survived (no death)

Mixed colors indicate parameter points with multiple death modes.

#### `cessation_time_heatmap.png`
Viridis heatmap showing mean time to functional death (t_death) in τ units.

#### `walking_dead_duration.png`
Orange heatmap overlay showing mean walking-dead phase duration (Δt_wd).
- **R** markers: PNR via ribosome depletion (R=0)
- **P** markers: PNR via RNAP deadlock (P=0 ∧ mP=0)

### 3. Summary Statistics

#### `summary.txt`
Human-readable text summary including:
- Run parameters (grid size, seeds, t_max)
- Death mode distribution (counts and percentages)
- PNR type distribution
- Mean/median cessation times
- Walking-dead statistics (if applicable)

## Parameter Sweep Details

### Command-line Options

```bash
python centraldogma_loop_sweep.py [OPTIONS]

--n-grid N        Grid points per axis (default: 8)
--n-seeds M       Random seeds per point (default: 10)
--t-max T         Max simulation time in τ (default: 50000)
--output-dir DIR  Output directory (default: ../results)
```

### What is `--n-seeds`?

Controls the number of **independent stochastic realizations** per parameter point.

Example: `--n-grid 8 --n-seeds 10`
- Creates 8×8 = 64 parameter combinations (N_RNAP × N_Ribo)
- Runs 10 simulations with different random seeds for each combination
- **Total: 640 simulations**

Multiple seeds enable:
- Statistical averaging (mean, median, variance)
- Capturing stochastic variability
- Robust phase boundary identification

### Dimensionless Units

- **Length**: λ = 1 (lattice spacing)
- **Time**: τ = λ²/D_mRNA = 1 (mRNA diffusion time)

**Physical conversion:**
- λ_phys = 1 μm
- D_mRNA_phys = 5×10⁻¹³ m²/s
- τ_phys = 2.0 s

Example: t_death = 5000 τ → 10,000 s (2.78 hours)

## Data Reuse Examples

### Replot Phase Diagram
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('aggregated.npz')
rgb = data['rgb']
RNAP_AXIS = data['RNAP_AXIS']
RIBO_AXIS = data['RIBO_AXIS']

fig, ax = plt.subplots()
ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower', aspect='auto')
ax.set_xticks(range(len(RNAP_AXIS)))
ax.set_xticklabels(RNAP_AXIS)
ax.set_yticks(range(len(RIBO_AXIS)))
ax.set_yticklabels(RIBO_AXIS)
ax.set_xlabel('Initial RNAP')
ax.set_ylabel('Initial Ribosome')
plt.show()
```

### Extract Specific Parameter Point
```python
import pickle

with open('results.pkl', 'rb') as f:
    data = pickle.load(f)

# Get all results for N_RNAP=20, N_Ribo=10
subset = [r for r in data['results']
          if r['N_RNAP'] == 20 and r['N_Ribo'] == 10]

print(f"Found {len(subset)} simulations")
lifetimes = [r['t_death'] for r in subset]
print(f"Mean lifetime: {np.mean(lifetimes):.1f} τ")
```

### Analyze Walking-Dead Phase
```python
import numpy as np
import pickle

with open('results.pkl', 'rb') as f:
    results = pickle.load(f)['results']

# Filter for trajectories with walking-dead phase
wd_trajs = [r for r in results if r['dt_walking_dead'] > 0]

# Group by PNR type
ribo_wd = [r['dt_walking_dead'] for r in wd_trajs
           if r['pnr_type'] == 'ribo_depletion']
rnap_wd = [r['dt_walking_dead'] for r in wd_trajs
           if r['pnr_type'] == 'rnap_deadlock']

print(f"Ribo pathway - mean Δt_wd: {np.mean(ribo_wd):.1f} τ")
print(f"RNAP pathway - mean Δt_wd: {np.mean(rnap_wd):.1f} τ")
```
