"""Predict one delivery from the command line."""

from pathlib import Path
import pandas as pd
from delivery import load_model


def main():
    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "work" / "model.joblib"
    model = load_model(str(model_path))

    order = pd.DataFrame([{
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])

    minutes = model.predict(order)[0]
    print(f"PREDICTION: {minutes:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
