# Smart Industry Anomaly Detection Dataset (SIADD)

This project detects cyber-physical threats in Industrial IoT network traffic using machine learning models trained on the NSL-KDD dataset.

## Attack Categories

- Normal
- Probing
- Denial of Service (DoS)
- User to Root (U2R)
- Remote to Local (R2L)

## Project Structure

- `CODE/FRONTEND/` - Flask web application for login, dataset preview, model information, and prediction.
- `CODE/BACKEND/` - Training notebook and saved research model artifacts.
- `CODE/FRONTEND/data/KDDTrain+.txt` - Dataset used by the web app.
- `CODE/FRONTEND/catboost_model.pkl` - Deployed CatBoost model.
- `CODE/FRONTEND/scaler.pkl` and `CODE/FRONTEND/top_features.pkl` - Preprocessing files required for prediction.

## Setup

Use Python 3.10. Render is pinned to Python 3.10.13 with `.python-version`.

```bash
cd CODE/FRONTEND
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Database

On Render, create a Render Postgres database and add its internal database URL to the web service as `DATABASE_URL`.

For local MySQL development, create the database using:

```bash
mysql -u root -p < db.sql
```

Optional environment variables:

- `DATABASE_URL`
- `SIADD_SECRET_KEY`
- `SIADD_DB_HOST`
- `SIADD_DB_PORT`
- `SIADD_DB_USER`
- `SIADD_DB_PASSWORD`
- `SIADD_DB_NAME`

## Run

```bash
cd CODE/FRONTEND
python app.py
```

Open `http://127.0.0.1:5000`.

## Dataset

Dataset source: https://www.kaggle.com/datasets/hassan06/nslkdd/data
