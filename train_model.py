"""Train a species recognition model on fictional data."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "imagined_endangered_species.csv"
MODEL_PATH = ROOT / "models" / "species_model.joblib"
REPORT_PATH = ROOT / "outputs" / "metrics.json"

NUMERIC_FEATURES = ["body_length_cm", "weight_kg", "altitude_m", "temperature_c"]
CATEGORICAL_FEATURES = ["region", "habitat", "activity", "visual_pattern", "threat_level", "observation_quality"]
TARGET = "species"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train() -> Pipeline:
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    return pipeline


if __name__ == "__main__":
    train()
