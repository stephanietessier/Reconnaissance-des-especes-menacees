"""Generate an imagined endangered species dataset.

The data is fictional and educational. It is not suitable for real conservation decisions.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "imagined_endangered_species.csv"

SPECIES_PROFILES = {
    "Panda roux": {
        "region": "Himalaya", "habitat": "forêt tempérée", "activity": "crépusculaire",
        "pattern": "roux et blanc", "length": (50, 70), "weight": (3, 7), "altitude": (1800, 4000),
        "threat": "en danger"
    },
    "Tigre de Sumatra": {
        "region": "Sumatra", "habitat": "forêt tropicale", "activity": "nocturne",
        "pattern": "rayures noires", "length": (220, 260), "weight": (75, 140), "altitude": (0, 1200),
        "threat": "critique"
    },
    "Gorille des montagnes": {
        "region": "Afrique centrale", "habitat": "forêt montagneuse", "activity": "diurne",
        "pattern": "pelage sombre", "length": (140, 180), "weight": (70, 180), "altitude": (2200, 4300),
        "threat": "en danger"
    },
    "Tortue imbriquée": {
        "region": "océans tropicaux", "habitat": "récif corallien", "activity": "diurne",
        "pattern": "carapace écailleuse", "length": (60, 100), "weight": (40, 80), "altitude": (0, 5),
        "threat": "critique"
    },
    "Vaquita": {
        "region": "Golfe de Californie", "habitat": "mer côtière", "activity": "diurne",
        "pattern": "gris avec taches", "length": (120, 150), "weight": (30, 55), "altitude": (0, 2),
        "threat": "critique"
    },
    "Rhinocéros de Java": {
        "region": "Java", "habitat": "forêt humide", "activity": "crépusculaire",
        "pattern": "peau grise", "length": (300, 340), "weight": (900, 2300), "altitude": (0, 600),
        "threat": "critique"
    },
    "Léopard des neiges": {
        "region": "Asie centrale", "habitat": "montagne rocheuse", "activity": "crépusculaire",
        "pattern": "rosettes grises", "length": (160, 230), "weight": (25, 55), "altitude": (3000, 5500),
        "threat": "vulnérable"
    },
    "Orang-outan de Bornéo": {
        "region": "Bornéo", "habitat": "forêt tropicale", "activity": "diurne",
        "pattern": "pelage orangé", "length": (110, 150), "weight": (30, 100), "altitude": (0, 1500),
        "threat": "critique"
    },
}


def sample_range(rng: np.random.Generator, low_high: tuple[float, float]) -> float:
    low, high = low_high
    return float(rng.uniform(low, high))


def generate_dataset(samples_per_species: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for species, p in SPECIES_PROFILES.items():
        for _ in range(samples_per_species):
            rows.append({
                "species": species,
                "region": p["region"],
                "habitat": p["habitat"],
                "activity": p["activity"],
                "visual_pattern": p["pattern"],
                "threat_level": p["threat"],
                "body_length_cm": round(sample_range(rng, p["length"]) + rng.normal(0, 4), 2),
                "weight_kg": round(max(0.5, sample_range(rng, p["weight"]) + rng.normal(0, 2)), 2),
                "altitude_m": round(max(0, sample_range(rng, p["altitude"]) + rng.normal(0, 80)), 2),
                "temperature_c": round(rng.uniform(5, 34), 2),
                "observation_quality": rng.choice(["faible", "moyenne", "bonne"], p=[0.15, 0.35, 0.50]),
            })
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset created: {DATA_PATH}")
    print(df.head())
