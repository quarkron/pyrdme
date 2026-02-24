"""Tests for Lattice2D class."""

import pytest
import numpy as np

from pyrdme import Lattice2D


class TestLattice2DInit:
    """Test Lattice2D initialization."""

    def test_basic_creation(self):
        """Test creating a simple lattice."""
        lattice = Lattice2D(nx=10, ny=20, species=['A', 'B'])

        assert lattice.nx == 10
        assert lattice.ny == 20
        assert lattice.shape == (10, 20)
        assert lattice.species == ['A', 'B']
        assert lattice.n_species == 2
        assert lattice.counts.shape == (2, 10, 20)
        assert lattice.counts.dtype == np.int32

    def test_default_spacing(self):
        """Test default spacing is 1 μm."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'])
        assert lattice.spacing == 1e-6

    def test_custom_spacing(self):
        """Test custom spacing."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'], spacing=2.5e-6)
        assert lattice.spacing == 2.5e-6

    def test_initial_counts_zero(self):
        """Test that initial counts are all zero."""
        lattice = Lattice2D(nx=5, ny=5, species=['A', 'B', 'C'])
        assert np.all(lattice.counts == 0)
        assert lattice.total_particles() == 0


class TestLattice2DSpecies:
    """Test species handling."""

    def test_species_index(self):
        """Test species_index returns correct index."""
        lattice = Lattice2D(nx=10, ny=10, species=['A', 'B', 'C'])

        assert lattice.species_index('A') == 0
        assert lattice.species_index('B') == 1
        assert lattice.species_index('C') == 2

    def test_species_index_invalid(self):
        """Test species_index raises KeyError for unknown species."""
        lattice = Lattice2D(nx=10, ny=10, species=['A', 'B'])

        with pytest.raises(KeyError):
            lattice.species_index('X')


class TestLattice2DParticles:
    """Test particle manipulation."""

    def test_add_particles_uniform_whole_lattice(self):
        """Test adding particles uniformly to whole lattice."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'])
        lattice.add_particles('A', count=100)

        assert lattice.total_particles('A') == 100
        # Uniform distribution: 100 particles / 100 sites = 1 per site
        assert np.all(lattice.get_counts('A') == 1)

    def test_add_particles_uniform_region(self):
        """Test adding particles to a specific region."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'])
        lattice.add_particles('A', count=100, region=(slice(0, 5), slice(0, 5)))

        assert lattice.total_particles('A') == 100
        # Only the 5x5 region should have particles
        counts = lattice.get_counts('A')
        assert np.sum(counts[0:5, 0:5]) == 100
        assert np.sum(counts[5:, :]) == 0
        assert np.sum(counts[:, 5:]) == 0

    def test_add_particles_random(self):
        """Test adding particles randomly."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'])
        rng = np.random.default_rng(42)
        lattice.add_particles('A', count=100, distribution='random', rng=rng)

        assert lattice.total_particles('A') == 100

    def test_get_counts(self):
        """Test get_counts returns correct array."""
        lattice = Lattice2D(nx=10, ny=10, species=['A', 'B'])
        lattice.add_particles('A', count=50)
        lattice.add_particles('B', count=30)

        counts_a = lattice.get_counts('A')
        counts_b = lattice.get_counts('B')

        assert counts_a.shape == (10, 10)
        assert counts_b.shape == (10, 10)
        assert np.sum(counts_a) == 50
        assert np.sum(counts_b) == 30

    def test_set_counts(self):
        """Test set_counts directly sets values."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'])

        # Create a pattern
        pattern = np.zeros((10, 10), dtype=np.int32)
        pattern[5, 5] = 100

        lattice.set_counts('A', pattern)

        assert lattice.get_counts('A')[5, 5] == 100
        assert lattice.total_particles('A') == 100

    def test_total_particles_single_species(self):
        """Test total_particles for single species."""
        lattice = Lattice2D(nx=10, ny=10, species=['A', 'B'])
        lattice.add_particles('A', count=100)
        lattice.add_particles('B', count=50)

        assert lattice.total_particles('A') == 100
        assert lattice.total_particles('B') == 50

    def test_total_particles_all_species(self):
        """Test total_particles for all species."""
        lattice = Lattice2D(nx=10, ny=10, species=['A', 'B'])
        lattice.add_particles('A', count=100)
        lattice.add_particles('B', count=50)

        assert lattice.total_particles() == 150


class TestLattice2DCopy:
    """Test lattice copying."""

    def test_copy_is_independent(self):
        """Test that copy creates independent lattice."""
        lattice = Lattice2D(nx=10, ny=10, species=['A'])
        lattice.add_particles('A', count=100)

        copy = lattice.copy()

        # Modify original
        lattice.add_particles('A', count=50)

        # Copy should be unchanged
        assert lattice.total_particles('A') == 150
        assert copy.total_particles('A') == 100

    def test_copy_preserves_state(self):
        """Test that copy preserves all state."""
        lattice = Lattice2D(nx=10, ny=10, species=['A', 'B'], spacing=2e-6)
        lattice.add_particles('A', count=100)
        lattice.add_particles('B', count=50)

        copy = lattice.copy()

        assert copy.nx == lattice.nx
        assert copy.ny == lattice.ny
        assert copy.species == lattice.species
        assert copy.spacing == lattice.spacing
        assert np.array_equal(copy.counts, lattice.counts)


class TestLattice2DSiteTypes:
    """Test site type assignment."""

    def test_set_site_type_slice(self):
        lattice = Lattice2D(nx=10, ny=10, species=['A'])
        lattice.set_site_type((slice(0, 5), slice(0, 5)), 'membrane')

        assert lattice.site_types is not None
        type_map = lattice.site_type_map()
        assert type_map['default'] == 0
        assert type_map['membrane'] == 1
        assert np.all(lattice.site_types[0:5, 0:5] == 1)
        assert np.all(lattice.site_types[5:, :] == 0)
        assert np.all(lattice.site_types[:, 5:] == 0)

    def test_set_site_type_mask(self):
        lattice = Lattice2D(nx=6, ny=6, species=['A'])
        mask = np.zeros((6, 6), dtype=bool)
        mask[2, 3] = True
        lattice.set_site_type(mask, 'cytoplasm')

        type_id = lattice.site_type_map()['cytoplasm']
        assert lattice.site_types[2, 3] == type_id
