"""Run one example prediction."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "species_model.joblib"


def predict_species(observation: dict) -> tuple[str, float]:
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([observation])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(probabilities.max())
    return prediction, confidence


if __name__ == "__main__":
    example = {
        "body_length_cm": 185,
        "weight_kg": 42,
        "altitude_m": 4200,
        "temperature_c": 8,
        "region": "Asie centrale",
        "habitat": "montagne rocheuse",
        "activity": "crépusculaire",
        "visual_pattern": "rosettes grises",
        "threat_level": "vulnérable",
        "observation_quality": "bonne",
    }
    species, confidence = predict_species(example)
    print(f"Espèce prédite : {species}")
    print(f"Confiance : {confidence:.1%}")
