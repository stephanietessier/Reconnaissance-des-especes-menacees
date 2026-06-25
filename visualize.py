"""Create simple visualizations for the fictional dataset."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "imagined_endangered_species.csv"
OUTPUT_PATH = ROOT / "outputs" / "species_distribution.png"

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = df["species"].value_counts().sort_values()
    plt.figure(figsize=(10, 6))
    counts.plot(kind="barh")
    plt.title("Nombre d'observations fictives par espèce")
    plt.xlabel("Observations")
    plt.ylabel("Espèce")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    print(f"Chart saved: {OUTPUT_PATH}")
