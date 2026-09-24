# Network Security — Phishing URL Detection (MLOps Pipeline)

An end-to-end machine learning system that detects phishing websites from URL/network-derived features. The project is built as a modular ML pipeline with experiment tracking, a REST API for training and inference, and a Streamlit UI — containerized for deployment.

## Overview

Raw phishing/legitimate site data is pushed to MongoDB, then pulled through a structured pipeline (ingestion → validation → transformation → training) that produces a serialized preprocessing object and a trained classification model. The model is served through a FastAPI backend and a Streamlit frontend, with training runs tracked in MLflow (backed by DagsHub).

## Architecture

```
CSV data → push_data.py → MongoDB
                              │
                              ▼
                     ┌─────────────────┐
                     │  Data Ingestion  │  Pulls data from MongoDB, splits train/test
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │ Data Validation  │  Schema checks, data drift detection
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │Data Transformation│  Imputation, feature engineering
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │  Model Trainer   │  Trains model, logs metrics/params to MLflow
                     └────────┬────────┘
                              ▼
                  final_model/ (model.pkl, preprocessor.pkl)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              FastAPI (app.py)   Streamlit (frontend.py)
              /train  /predict     interactive UI
```

## Features

- **Modular training pipeline** — separate, testable components for data ingestion, validation, transformation, and model training, orchestrated via a single `TrainingPipeline` entry point (`main.py`).
- **MongoDB as the data source** — raw CSV data is converted to JSON and pushed to MongoDB (`push_data.py`); the ingestion component reads from there instead of static files.
- **Experiment tracking** — model parameters, metrics, and artifacts are logged with MLflow, backed by DagsHub for remote tracking.
- **Model serving API** — a FastAPI app exposes:
  - `GET /train` — triggers the full training pipeline on demand
  - `POST /predict` — accepts a CSV upload, runs it through the saved preprocessor + model, and returns predictions
- **Interactive frontend** — a Streamlit app for uploading data and viewing predictions without calling the API directly.
- **Containerized deployment** — a Docker image installs dependencies (including the AWS CLI for artifact/model storage) and runs the Streamlit app on port 8501.
- **CI-ready** — includes a GitHub Actions workflow directory for automated build/deploy steps.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data storage | MongoDB (via PyMongo) |
| ML / Modeling | scikit-learn, XGBoost |
| Experiment tracking | MLflow, DagsHub |
| API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Packaging / Deployment | Docker, AWS CLI |
| CI | GitHub Actions |

## Project Structure

```
Network-Security/
├── networksecurity/
│   ├── component/        # data_ingestion, data_validation, data_transformation, model_trainer
│   ├── entity/            # config & artifact entity classes
│   ├── exception/         # custom exception handling
│   ├── logging/           # logging setup
│   └── utils/             # shared ML/model utilities
├── Network_Data/          # raw phishing dataset (CSV)
├── data_schema/           # expected schema for validation
├── valid_data/            # validated train/test splits
├── predicted_output/      # inference output
├── .github/workflows/     # CI pipeline
├── app.py                 # FastAPI backend (train/predict endpoints)
├── frontend.py             # Streamlit UI
├── main.py                 # runs the full training pipeline locally
├── push_data.py             # loads CSV data into MongoDB
├── Dockerfile
├── requirements.txt
└── setup.py
```

## Setup

```bash
git clone https://github.com/garvagrawal9084/Network-Security.git
cd Network-Security
pip install -r requirements.txt
```

Create a `.env` file with:
```
MONGODB_URI=<your-mongodb-connection-string>
```

Push the dataset to MongoDB:
```bash
python push_data.py
```

Run the training pipeline directly:
```bash
python main.py
```

Or start the API:
```bash
python app.py
# then visit http://localhost:8000/docs
```

Or run the Streamlit app:
```bash
streamlit run frontend.py
```

### Run with Docker

```bash
docker build -t network-security .
docker run -p 8501:8501 network-security
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Redirects to interactive API docs |
| `/train` | GET | Runs the full training pipeline |
| `/predict` | POST | Upload a CSV of features, returns predictions per row |

## License

This project is for educational/portfolio purposes.
