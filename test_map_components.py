import pytest
from Map_Components import Planetary_System, Star


class TestPlanetarySystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.star = Star(
            id=1, name="Sun", x=0, y=0, z=0, spectral_class="G-Type", luminosity=1
        )
        self.system = Planetary_System(star=self.star)

    def test_generate_orbits(self):
        self.system.generate_orbits(include_habitable_zone=True, num_orbits=3)
        assert len(self.system.orbits) == 3
        assert all(orbit >= 0 for orbit in self.system.orbits)
