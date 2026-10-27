from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "historical_energy.csv"
MODEL = BASE / "energy_model.joblib"
METRICS = BASE / "model_metrics.json"

def train():
    df = pd.read_csv(DATA)
    features = ["computers_used", "operating_hours", "avg_power_watts"]
    target = "energy_kwh"
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=250, random_state=42, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae_kwh": round(float(mean_absolute_error(y_test, predictions)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": features,
        "target": target,
        "dataset_note": "The bundled dataset is synthetic demonstration data."
    }
    joblib.dump(model, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved to: {MODEL}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    train()
