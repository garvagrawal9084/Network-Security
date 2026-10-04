# Network Security: Phishing Website Detection (End-to-End MLOps)

An end-to-end machine learning system that detects phishing websites from URL and network-derived features. It is built as a modular pipeline with experiment tracking, a REST API for training and inference, and a Streamlit UI, all containerized with Docker.

![Demo](assets/demo.gif)
<!-- Add a short GIF of uploading a CSV in Streamlit and getting predictions -->

## Highlights

- Modular pipeline: ingestion → validation → transformation → training, run through a single entry point
- MongoDB as the data source instead of static files
- Schema validation and data drift checks before training
- Experiment tracking with MLflow, with remote tracking on DagsHub
- FastAPI service with `/train` and `/predict` endpoints
- Streamlit frontend for uploading data and viewing predictions
- Docker image for reproducible deployment
- GitHub Actions workflow: [DESCRIBE WHAT IT DOES, e.g. builds the Docker image on every push to main]

## Results

Evaluated on a held-out test set ([N] samples, [X]% phishing).

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| [Model 1] | [ ] | [ ] | [ ] | [ ] |
| [Model 2, e.g. XGBoost] | [ ] | [ ] | [ ] | [ ] |
| [Model 3] | [ ] | [ ] | [ ] | [ ] |

**Selected model:** [name], chosen because [reason, e.g. best recall at an acceptable false-positive rate].

## Architecture

```
CSV data → push_data.py → MongoDB
                              │
                              ▼
                     ┌──────────────────┐
                     │  Data Ingestion   │  Pull from MongoDB, train/test split
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │ Data Validation   │  Schema checks, drift detection
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │Data Transformation│  Imputation, feature engineering
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │  Model Trainer    │  Train, log metrics/params to MLflow
                     └────────┬─────────┘
                              ▼
               final_model/ (model.pkl, preprocessor.pkl)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              FastAPI (app.py)    Streamlit (frontend.py)
              /train  /predict      interactive UI
```

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data storage | MongoDB (PyMongo) |
| Modeling | scikit-learn, XGBoost |
| Experiment tracking | MLflow, DagsHub |
| API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Deployment | Docker, AWS CLI |
| CI | GitHub Actions |

## Project Structure

```
Network-Security/
├── networksecurity/
│   ├── component/     # ingestion, validation, transformation, model trainer
│   ├── entity/        # config and artifact classes
│   ├── exception/     # custom exception handling
│   ├── logging/       # logging setup
│   └── utils/         # shared ML utilities
├── Network_Data/      # raw phishing dataset (CSV)
├── data_schema/       # expected schema for validation
├── valid_data/        # validated train/test splits
├── predicted_output/  # inference output
├── .github/workflows/ # CI pipeline
├── app.py             # FastAPI backend
├── frontend.py        # Streamlit UI
├── main.py            # runs the training pipeline locally
├── push_data.py       # loads CSV data into MongoDB
├── Dockerfile
├── requirements.txt
└── setup.py
```

## Getting Started

```bash
git clone https://github.com/garvagrawal9084/Network-Security.git
cd Network-Security
pip install -r requirements.txt
```

Create a `.env` file:

```
MONGODB_URI=<your-mongodb-connection-string>
```

Load the dataset into MongoDB, then choose how to run it:

```bash
python push_data.py          # load data into MongoDB
python main.py               # run the training pipeline
python app.py                # start the API at http://localhost:8000/docs
streamlit run frontend.py    # start the UI
```

### Docker

```bash
docker build -t network-security .
docker run -p 8501:8501 network-security
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Redirects to interactive API docs |
| `/train` | GET | Runs the full training pipeline |
| `/predict` | POST | Upload a CSV of features, returns a prediction per row |

## Limitations and Future Work

- Trained on [DATASET NAME]; performance on newer phishing patterns is untested
- [Add one or two honest items, e.g. no live URL feature extraction yet, no model monitoring in production]
- Planned: [e.g. extract features directly from a raw URL, add a browser extension]

## License

For educational and portfolio purposes.
