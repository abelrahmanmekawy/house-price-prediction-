# House Price Prediction

Predicts Indian residential property prices from a Kaggle listings dataset.
Raw data → cleaning & feature engineering → trained sklearn Pipeline →
FastAPI backend → React + TypeScript frontend.

## Overview

A user fills in property details (location, area, floor, bathrooms,
furnishing, etc.) in a web form. The React frontend sends those details to
a FastAPI backend, which runs them through a scikit-learn `Pipeline`
(preprocessing + regressor) trained in `notebooks/house_price_model.ipynb`,
and returns a predicted price.

## Architecture

```
┌──────────────┐      POST /predict       ┌──────────────────┐      .predict()     ┌────────────────────┐
│ React (Vite) │ ───────────────────────▶ │  FastAPI backend │ ──────────────────▶ │ sklearn Pipeline    │
│ :5173        │ ◀─────────────────────── │  :8000           │ ◀────────────────── │ (house_price.pkl)   │
└──────────────┘   {predicted_price}      └──────────────────┘    predicted price  └────────────────────┘
```

The pipeline bundles `ColumnTransformer` (imputation, scaling, one-hot
encoding) + the regressor, so the backend never encodes anything manually —
it only builds a one-row DataFrame with the right column names.

## Tech stack

| Layer      | Tech |
|------------|------|
| Modeling   | pandas, scikit-learn, joblib, matplotlib/seaborn |
| Backend    | FastAPI, Pydantic v2, uvicorn |
| Frontend   | React 18, TypeScript, Vite, React Router |
| Packaging  | Docker (backend) |

## Project structure

```
house-price-app/
├── notebooks/
│   ├── house_price_model.ipynb   # data cleaning, EDA, training, export
│   └── data/                     # put house_prices.csv here (gitignored)
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app, CORS, startup model load
│   │   ├── api/routes/prediction.py   # GET /health, POST /predict, GET /locations
│   │   ├── core/config.py             # settings from .env
│   │   ├── schemas/prediction.py      # PredictionRequest / PredictionResponse
│   │   ├── services/
│   │   │   ├── preprocessing.py       # request → one-row DataFrame
│   │   │   └── inference.py           # loads .pkl, runs predict
│   │   └── utils/logging_config.py
│   ├── models/                        # house_price.pkl + locations.json go here
│   ├── tests/test_prediction.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/predictionClient.ts    # fetch wrapper
│   │   ├── components/PredictionForm.tsx
│   │   ├── pages/HomePage.tsx | ResultPage.tsx | NotFoundPage.tsx
│   │   ├── types/prediction.ts        # mirrors backend schema
│   │   └── App.tsx                    # routes: / , /result , * (404)
│   ├── public/locations.json          # placeholder — replace with notebook export
│   └── .env.example
└── README.md
```

## Dataset

[House Price by Juhi Bhojani](https://www.kaggle.com/datasets/juhibhojani/house-price)
on Kaggle (~187,000 Indian property listings).

```bash
pip install kaggle
# Get your API token: Kaggle → Settings → API → "Create New Token"
# Place kaggle.json in ~/.kaggle/ (or C:\Users\<you>\.kaggle\ on Windows)
kaggle datasets download -d juhibhojani/house-price -p notebooks/data --unzip
```

## 1. Run the notebook

```bash
cd notebooks
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install jupyter pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook house_price_model.ipynb
```

Run all cells top to bottom. It cleans the data, trains and compares
`LinearRegression`, `Ridge` and `RandomForestRegressor`, and exports:

- `house_price.pkl` — the winning model's full Pipeline
- `locations.json` — the top-50 locations used during training

Copy both into `backend/models/` and `locations.json` into
`frontend/public/locations.json` (replacing the placeholder there).

## 2. Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs to try `/predict` from the Swagger UI.

### Environment variables (`backend/.env`)

| Variable       | Default                     | Description                          |
|----------------|------------------------------|---------------------------------------|
| `CORS_ORIGINS` | `http://localhost:5173`      | Comma-separated allowed origins       |
| `MODEL_PATH`   | `./models/house_price.pkl`   | Path to the exported pipeline         |
| `LOCATIONS_PATH` | `./models/locations.json`  | Path to the known-locations list      |

### API reference

**`GET /health`**
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

**`POST /predict`**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "location": "Whitefield, Bangalore",
        "carpet_area_sqft": 1200,
        "floor_num": 3,
        "bathroom": 2,
        "balcony": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East"
      }'
# {"predicted_price": 8215000.0}
```

**`GET /locations`** — bonus convenience endpoint returning the known
location list (the frontend defaults to the committed `locations.json`
instead, but falls back to this if that file is missing).

### Tests

```bash
cd backend
pytest
```

## 3. Run the frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open http://localhost:5173, fill in the form, submit, and see the
predicted price on the result page.

### Environment variables (`frontend/.env`)

| Variable              | Default                 | Description                  |
|-----------------------|--------------------------|-------------------------------|
| `VITE_API_BASE_URL`   | `http://localhost:8000`  | Base URL of the FastAPI backend |

## Model metrics

*(Fill in with your own run's numbers from notebook section 2.5 —
`results_df` — after training on the full dataset.)*

| Model             | MAE | RMSE | R² |
|-------------------|-----|------|----|
| LinearRegression  |     |      |    |
| Ridge             |     |      |    |
| RandomForest      |     |      |    |

## Screenshots
<img width="1606" height="975" alt="Screenshot 2026-08-28 183937" src="https://github.com/user-attachments/assets/f03c6a87-ad9b-46f4-83e1-ad49bdc850dd" />


## Common pitfalls

- **scikit-learn version mismatch** — pin `backend/requirements.txt`'s
  `scikit-learn` version to match `sklearn.__version__` printed at the end
  of the notebook, or the pickle may fail to load.
- **Hard-coded API URL** — the frontend must read `VITE_API_BASE_URL` from
  `.env`, never hard-code `http://localhost:8000`.
- **Committing secrets or the raw CSV** — both are excluded in
  `.gitignore`; only the trained `.pkl` (if under 50 MB) should be committed.
