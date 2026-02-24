"""
Tests for reaction propensity calculation and stoichiometry.

These tests verify that:
1. Propensity formulas are correct for all reaction orders
2. Stoichiometry changes are applied correctly
3. Edge cases (zero counts, self-reactions) are handled
"""

import pytest
import numpy as np
from pyrdme import Lattice2D, Reaction
from pyrdme.solvers import MPDSolver


class TestPropensityCalculation:
    """Test propensity formulas for different reaction types."""

    def test_zeroth_order_propensity(self):
        """Zeroth order: ∅ → A, propensity = k."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'])
        reaction = Reaction([], ['A'], k=5.0)
        solver = MPDSolver(lattice, [reaction], {'A': 0}, seed=42)

        propensity = solver._propensity_for_reaction(reaction)

        # Should be constant k at all sites
        assert propensity.shape == (10, 10)
        assert np.allclose(propensity, 5.0)

    def test_first_order_propensity(self):
        """First order: A → products, propensity = k * n_A."""
        lattice = Lattice2D(nx=5, ny=5, species=['A', 'B'])
        lattice.counts[0, :, :] = np.arange(25).reshape(5, 5)  # A counts: 0-24

        reaction = Reaction(['A'], ['B'], k=0.1)
        solver = MPDSolver(lattice, [reaction], {'A': 0, 'B': 0}, seed=42)

        propensity = solver._propensity_for_reaction(reaction)

        expected = 0.1 * np.arange(25).reshape(5, 5)
        assert np.allclose(propensity, expected)

    def test_second_order_different_species(self):
        """Second order (A + B): propensity = k * n_A * n_B."""
        lattice = Lattice2D(nx=3, ny=3, species=['A', 'B', 'C'])

        # Set counts
        lattice.counts[0, :, :] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # A
        lattice.counts[1, :, :] = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]  # B

        reaction = Reaction(['A', 'B'], ['C'], k=0.01)
        solver = MPDSolver(lattice, [reaction], {'A': 0, 'B': 0, 'C': 0}, seed=42)

        propensity = solver._propensity_for_reaction(reaction)

        n_A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        n_B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        expected = 0.01 * n_A * n_B
        assert np.allclose(propensity, expected)

    def test_second_order_self_reaction(self):
        """Self reaction (A + A): propensity = k * n_A * (n_A - 1) / 2."""
        lattice = Lattice2D(nx=3, ny=3, species=['A', 'A2'])

        # Set counts: 0, 1, 2, 3, ...
        lattice.counts[0, :, :] = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        reaction = Reaction(['A', 'A'], ['A2'], k=0.1)
        solver = MPDSolver(lattice, [reaction], {'A': 0, 'A2': 0}, seed=42)

        propensity = solver._propensity_for_reaction(reaction)

        n_A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float64)
        expected = 0.1 * n_A * (n_A - 1) / 2
        assert np.allclose(propensity, expected)

    def test_propensity_zero_counts(self):
        """Propensity should be zero when reactant count is zero."""
        lattice = Lattice2D(nx=3, ny=3, species=['A', 'B'])
        # All zeros
        reaction = Reaction(['A'], ['B'], k=1.0)
        solver = MPDSolver(lattice, [reaction], {'A': 0, 'B': 0}, seed=42)

        propensity = solver._propensity_for_reaction(reaction)
        assert np.allclose(propensity, 0)

    def test_propensity_self_reaction_single_particle(self):
        """Self-reaction with 1 particle should have zero propensity."""
        lattice = Lattice2D(nx=3, ny=3, species=['A', 'A2'])
        lattice.counts[0, :, :] = 1  # One particle everywhere

        reaction = Reaction(['A', 'A'], ['A2'], k=1.0)
        solver = MPDSolver(lattice, [reaction], {'A': 0, 'A2': 0}, seed=42)

        propensity = solver._propensity_for_reaction(reaction)
        # n*(n-1)/2 = 1*0/2 = 0
        assert np.allclose(propensity, 0)


class TestStoichiometryMatrix:
    """Test stoichiometry matrix construction."""

    def test_simple_reaction(self):
        """A + B -> C: A-1, B-1, C+1."""
        lattice = Lattice2D(nx=5, ny=5, species=['A', 'B', 'C'])
        reaction = Reaction(['A', 'B'], ['C'], k=1.0)
        solver = MPDSolver(lattice, [reaction], {'A': 0, 'B': 0, 'C': 0}, seed=42)

        S = solver._stoichiometry
        assert S.shape == (1, 3)
        assert S[0, 0] == -1  # A consumed
        assert S[0, 1] == -1  # B consumed
        assert S[0, 2] == 1   # C produced

    def test_unbinding_reaction(self):
        """C -> A + B: C-1, A+1, B+1."""
        lattice = Lattice2D(nx=5, ny=5, species=['A', 'B', 'C'])
        reaction = Reaction(['C'], ['A', 'B'], k=1.0)
        solver = MPDSolver(lattice, [reaction], {'A': 0, 'B': 0, 'C': 0}, seed=42)

        S = solver._stoichiometry
        assert S[0, 0] == 1   # A produced
        assert S[0, 1] == 1   # B produced
        assert S[0, 2] == -1  # C consumed

    def test_dimerization(self):
        """A + A -> A2: A-2, A2+1."""
        lattice = Lattice2D(nx=5, ny=5, species=['A', 'A2'])
        reaction = Reaction(['A', 'A'], ['A2'], k=1.0)
        solver = MPDSolver(lattice, [reaction], {'A': 0, 'A2': 0}, seed=42)

        S = solver._stoichiometry
        assert S[0, 0] == -2  # A consumed (2 molecules)
        assert S[0, 1] == 1   # A2 produced

    def test_production(self):
        """∅ -> A: A+1."""
        lattice = Lattice2D(nx=5, ny=5, species=['A'])
        reaction = Reaction([], ['A'], k=1.0)
        solver = MPDSolver(lattice, [reaction], {'A': 0}, seed=42)

        S = solver._stoichiometry
        assert S[0, 0] == 1  # A produced

    def test_degradation(self):
        """A -> ∅: A-1."""
        lattice = Lattice2D(nx=5, ny=5, species=['A'])
        reaction = Reaction(['A'], [], k=1.0)
        solver = MPDSolver(lattice, [reaction], {'A': 0}, seed=42)

        S = solver._stoichiometry
        assert S[0, 0] == -1  # A consumed

    def test_multiple_reactions(self):
        """Multiple reactions should have correct stoichiometry."""
        lattice = Lattice2D(nx=5, ny=5, species=['A', 'B', 'C'])
        reactions = [
            Reaction(['A', 'B'], ['C'], k=1.0),
            Reaction(['C'], ['A', 'B'], k=0.5),
        ]
        solver = MPDSolver(
            lattice, reactions,
            {'A': 0, 'B': 0, 'C': 0},
            seed=42
        )

        S = solver._stoichiometry
        assert S.shape == (2, 3)

        # Reaction 0: A + B -> C
        assert S[0, 0] == -1
        assert S[0, 1] == -1
        assert S[0, 2] == 1

        # Reaction 1: C -> A + B
        assert S[1, 0] == 1
        assert S[1, 1] == 1
        assert S[1, 2] == -1


class TestReactionApplication:
    """Test that reactions are applied correctly."""

    def test_single_reaction_step(self):
        """Test that reaction firing updates counts correctly."""
        lattice = Lattice2D(nx=1, ny=1, species=['A', 'B', 'C'])
        lattice.counts[0, 0, 0] = 100  # A
        lattice.counts[1, 0, 0] = 100  # B
        lattice.counts[2, 0, 0] = 0    # C

        # Moderate rate reaction
        reaction = Reaction(['A', 'B'], ['C'], k=0.01)
        solver = MPDSolver(
            lattice, [reaction],
            {'A': 0, 'B': 0, 'C': 0},
            seed=42
        )

        # Store initial values
        initial_A = int(lattice.counts[0, 0, 0])
        initial_B = int(lattice.counts[1, 0, 0])
        initial_C = int(lattice.counts[2, 0, 0])

        # Run many reaction steps to accumulate firings
        for _ in range(100):
            solver.step_reaction_only(dt=0.1)

        # A and B should decrease, C should increase
        final_A = int(lattice.counts[0, 0, 0])
        final_B = int(lattice.counts[1, 0, 0])
        final_C = int(lattice.counts[2, 0, 0])

        assert final_A < initial_A
        assert final_B < initial_B
        assert final_C > 0

        # Conservation: A_lost = B_lost = C_gained
        # Note: Due to clamping at zero, we check that A+C = initial_A+initial_C
        # and B+C = initial_B+initial_C (conservation laws for this reaction)
        assert (final_A + final_C) == (initial_A + initial_C)
        assert (final_B + final_C) == (initial_B + initial_C)

    def test_conservation_with_reversible_reactions(self):
        """A + B <-> C should conserve A+C and B+C."""
        lattice = Lattice2D(nx=10, ny=10, species=['A', 'B', 'C'])
        lattice.add_particles('A', count=500)
        lattice.add_particles('B', count=500)

        reactions = [
            Reaction(['A', 'B'], ['C'], k=0.001),
            Reaction(['C'], ['A', 'B'], k=0.1),
        ]
        solver = MPDSolver(
            lattice, reactions,
            {'A': 0, 'B': 0, 'C': 0},
            seed=42
        )

        initial_A = lattice.total_particles('A')
        initial_B = lattice.total_particles('B')
        initial_C = lattice.total_particles('C')

        # Run for many steps
        for _ in range(1000):
            solver.step_reaction_only(dt=0.01)

        final_A = lattice.total_particles('A')
        final_B = lattice.total_particles('B')
        final_C = lattice.total_particles('C')

        # Conservation: A + C = initial_A + initial_C
        assert (final_A + final_C) == (initial_A + initial_C)
        # Conservation: B + C = initial_B + initial_C
        assert (final_B + final_C) == (initial_B + initial_C)

    def test_production_reaction(self):
        """∅ -> A should increase A count."""
        lattice = Lattice2D(nx=5, ny=5, species=['A'])

        reaction = Reaction([], ['A'], k=1.0)
        solver = MPDSolver(lattice, [reaction], {'A': 0}, seed=42)

        initial = lattice.total_particles('A')

        # Run production
        for _ in range(100):
            solver.step_reaction_only(dt=0.1)

        final = lattice.total_particles('A')
        assert final > initial

    def test_degradation_reaction(self):
        """A -> ∅ should decrease A count."""
        lattice = Lattice2D(nx=5, ny=5, species=['A'])
        lattice.add_particles('A', count=1000)

        reaction = Reaction(['A'], [], k=0.1)
        solver = MPDSolver(lattice, [reaction], {'A': 0}, seed=42)

        initial = lattice.total_particles('A')

        # Run degradation
        for _ in range(100):
            solver.step_reaction_only(dt=0.1)

        final = lattice.total_particles('A')
        assert final < initial


class TestMPDSolverValidation:
    """Test MPDSolver validation and error handling."""

    def test_missing_species_in_reaction(self):
        """Reactions with unknown species should raise error."""
        lattice = Lattice2D(nx=5, ny=5, species=['A', 'B'])
        reaction = Reaction(['A', 'X'], ['B'], k=1.0)  # X not in lattice

        with pytest.raises(ValueError, match="not found in lattice"):
            MPDSolver(lattice, [reaction], {'A': 0, 'B': 0}, seed=42)

    def test_empty_reactions_list(self):
        """Empty reactions list should work (diffusion-only)."""
        lattice = Lattice2D(nx=5, ny=5, species=['A'])
        lattice.add_particles('A', count=100)

        solver = MPDSolver(lattice, [], {'A': 1e-12}, seed=42)
        dt = solver.get_max_timestep() * 0.9

        # Should not raise
        for _ in range(10):
            solver.step(dt)

        # Particles conserved
        assert lattice.total_particles('A') == 100

    def test_zero_diffusion_with_reactions(self):
        """Zero diffusion should still allow reactions."""
        lattice = Lattice2D(nx=5, ny=5, species=['A', 'B'])
        lattice.add_particles('A', count=100)

        reaction = Reaction(['A'], ['B'], k=0.1)
        solver = MPDSolver(lattice, [reaction], {'A': 0, 'B': 0}, seed=42)

        initial_A = lattice.total_particles('A')
        for _ in range(100):
            solver.step(dt=0.01)

        # Reactions should have occurred
        final_A = lattice.total_particles('A')
        final_B = lattice.total_particles('B')
        assert final_A < initial_A
        assert final_B > 0
        # Conservation
        assert final_A + final_B == initial_A
