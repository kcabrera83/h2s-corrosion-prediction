import numpy as np
import pandas as pd


class CorrosionDataGenerator:
    """Generates realistic synthetic corrosion data for H2S environments."""

    MATERIALS = ["carbon_steel", "stainless_steel", "coated"]

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate(self, n_samples=2000):
        """Generate synthetic corrosion dataset.

        Returns DataFrame with columns:
        h2s_concentration_ppm, co2_concentration_pct, temperature_c,
        pressure_mpa, ph, flow_velocity_ms, pipe_material,
        wall_loss_mm, corrosion_rate_mpy, pitting_depth_mm
        """
        h2s = self.rng.uniform(50, 5000, n_samples)
        co2 = self.rng.uniform(0.1, 15.0, n_samples)
        temp = self.rng.uniform(20, 200, n_samples)
        pressure = self.rng.uniform(0.1, 25.0, n_samples)
        ph = self.rng.uniform(2.5, 8.5, n_samples)
        velocity = self.rng.uniform(0.1, 15.0, n_samples)
        material_idx = self.rng.integers(0, len(self.MATERIALS), n_samples)
        material = np.array([self.MATERIALS[i] for i in material_idx])

        base_rate = (
            0.5
            + 0.003 * h2s
            + 0.8 * co2
            + 0.02 * temp
            + 0.3 * pressure
            + 0.4 * np.abs(5.5 - ph)
            + 0.15 * velocity
        )

        material_factor = np.where(
            material == "carbon_steel", 1.2,
            np.where(material == "stainless_steel", 0.3, 0.15),
        )

        corrosion_rate = base_rate * material_factor + self.rng.normal(0, 0.5, n_samples)
        corrosion_rate = np.clip(corrosion_rate, 0.01, None)

        wall_loss = corrosion_rate * self.rng.uniform(1, 10, n_samples) * 0.001
        pitting_depth = wall_loss * self.rng.uniform(1.2, 3.5, n_samples)

        df = pd.DataFrame({
            "h2s_concentration_ppm": np.round(h2s, 2),
            "co2_concentration_pct": np.round(co2, 2),
            "temperature_c": np.round(temp, 1),
            "pressure_mpa": np.round(pressure, 2),
            "ph": np.round(ph, 2),
            "flow_velocity_ms": np.round(velocity, 2),
            "pipe_material": material,
            "wall_loss_mm": np.round(wall_loss, 4),
            "corrosion_rate_mpy": np.round(corrosion_rate, 2),
            "pitting_depth_mm": np.round(pitting_depth, 4),
        })
        return df
