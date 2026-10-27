# Deploying LabEnergy to Streamlit Community Cloud

This guide provides simple, step-by-step instructions to run the application locally and deploy it online for free using **Streamlit Community Cloud**.

---

## 🚀 1. Run Locally

Open a PowerShell terminal in your project directory:

```powershell
# Navigate to the project directory
cd computer-lab-energy

# Activate your virtual environment (if using venv)
.\backend\venv\Scripts\Activate.ps1

# Start the Streamlit application
streamlit run streamlit_app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 🌐 2. Deploy to Streamlit Community Cloud (Free)

Streamlit Community Cloud lets you host and share your app with a public URL for free.

### Step 1: Push Code to GitHub

1. Open PowerShell in your project folder (`computer-lab-energy`):
   ```powershell
   git init
   git add .
   git commit -m "Initial commit of LabEnergy Streamlit app"
   ```

2. Create a new repository on [GitHub](https://github.com/new) (e.g. `computer-lab-energy`).

3. Link your local repository and push:
   ```powershell
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/computer-lab-energy.git
   git branch -M main
   git push -u origin main
   ```

---

### Step 2: Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Fill in the deployment form:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/computer-lab-energy`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py` (or `app.py`)
4. *(Optional)* Click **"Advanced settings"** to choose Python 3.10, 3.11, or 3.12 if prompted.
5. Click **"Deploy!"** 🎈

Streamlit Cloud will read `requirements.txt`, install the necessary packages (`streamlit`, `pandas`, `scikit-learn`, `joblib`, `plotly`), and launch your live app with a shareable URL (e.g., `https://your-app-name.streamlit.app`).

---

## 📁 Key Files Created for Deployment

| File | Purpose |
| :--- | :--- |
| `streamlit_app.py` | Standalone Streamlit application containing the dashboard, predictor, and charts |
| `requirements.txt` | Package dependencies required for Streamlit Cloud deployment |
| `.streamlit/config.toml` | Theme configuration (purple accent, modern typography, clean layout) |
| `data/historical_energy.csv` | Bundled dataset for immediate demonstration |
| `energy_model.joblib` | Pre-trained scikit-learn Random Forest model |

---

## 💡 Troubleshooting Tips

- **Model file missing?** No worries! The app includes an automatic fallback that trains a fresh Random Forest model directly from `historical_energy.csv` on the fly.
- **Custom Dataset:** You can upload your own college lab CSV readings directly inside the **"Historical Data & Upload"** tab in the app, and retrain the model with one click.
