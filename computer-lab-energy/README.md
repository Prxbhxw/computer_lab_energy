# Computer Lab Energy Consumption Prediction

A beginner-friendly college project using **React + Vite** for the frontend and **Python + FastAPI + scikit-learn** for predictions. It uses historical CSV data and does **not** require real-time sensors.

> Dataset note: `backend/data/historical_energy.csv` contains synthetic sample data created for demonstrating the workflow. Replace it with verified historical readings before reporting results as real measurements.

## Features
- Dashboard overview with estimated energy and cost
- Interactive consumption trend chart
- Prediction form for computers used, operating hours, and average power per computer
- FastAPI endpoint for machine-learning predictions
- Model training and evaluation script
- CSV dataset for experimentation

## Streamlit App (Fastest & Easiest — Cloud Deployable)

A complete, self-contained **Streamlit** dashboard is included, ideal for **one-click deployment to Streamlit Community Cloud (share.streamlit.io)** without needing separate backend servers.

```powershell
# Run Streamlit locally
streamlit run streamlit_app.py
```
> See [`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md) for step-by-step instructions on deploying for free to Streamlit Cloud.

---

## Alternative: React + FastAPI Stack
If you prefer running the decoupled React + FastAPI development setup:
Open a terminal in the project folder:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

API docs: http://127.0.0.1:8000/docs

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `.\.venv\Scripts\python.exe train_model.py` and `.\.venv\Scripts\python.exe -m uvicorn main:app --reload`.

## 2. Start the React frontend
Open a **second** terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL Vite prints (usually http://localhost:5173).

## How the model works
- Features: number of computers used, operating hours, average power rating in watts.
- Target: estimated energy consumed in kWh.
- Model: Random Forest Regressor.
- Evaluation: MAE and R² on a held-out test split.

The sample data follows a simplified energy formula, so the model demonstrates the pipeline rather than proving real-world predictive accuracy. Use real historical meter readings for a valid evaluation. The cost shown in the UI uses an editable default of ₹8 per kWh as an illustrative assumption.

## API
- `GET /api/summary` — historical summary and trend data
- `POST /api/predict` — predicts kWh and estimated cost
- `GET /api/health` — backend status

## Suggested next steps
1. Replace synthetic CSV data with actual historical records or a suitable public dataset.
2. Add CSV upload and validation.
3. Compare Random Forest with Linear Regression and Decision Tree.
4. Add user authentication and a database only if required by the project scope.
