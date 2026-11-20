# CardioPredict-X deployment

## Requirements
- Python 3.11
- Node.js 22
- Docker with Compose (optional)
- Trained artifacts in `artifacts/` (not tracked by Git)

## Train locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.train --trials 20
```

The model and evaluation metrics are generated locally. Never present sample results as actual model output.

## Run API and React
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
In another terminal:
```bash
cd frontend
npm ci
npm run dev
```
The Vite development proxy forwards `/api` and `/health` to port 8000.

## Docker
```bash
docker compose up --build
```
Compose currently starts FastAPI and Streamlit. React is run separately.

## History security
History is disabled by default. To enable it, set `ENABLE_HISTORY=1` and `HISTORY_API_TOKEN` to a long secret. Reading or deleting history requires `Authorization: Bearer <token>`. Never embed this token in client-side React code. A publicly deployed application needs per-user authentication, authorization, HTTPS, rate limits, and appropriate data retention controls. Do not use the current shared-token approach as a substitute for per-user identity.

## Research limitations
This is retrospective binary classification on a public heart disease dataset, not clinical diagnosis or prospective risk estimation. Validation on independent cohorts is required before any clinical application.
