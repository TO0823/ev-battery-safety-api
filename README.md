# EV Battery Fault Predictor & Safety API

An end-to-end machine learning microservice built to predict electric vehicle (EV) battery faults in real-time using vehicle telemetry data.

This project takes raw battery sensor data, processes it through a highly accurate GPU-accelerated XGBoost model, and serves the predictions via a blazing-fast FastAPI web server, all fully containerized in Docker for production deployment.

## Features

* **Real-Time Inference:** Millisecond response times for battery health assessments.

* **XGBoost Machine Learning:** Trained on massive PyTorch tensor/CSV datasets of EV telemetry to identify complex failure patterns.

* **Interactive API Documentation:** Built-in Swagger UI for easy testing and endpoint interaction.

* **Production Ready:** Fully containerized with Docker, ensuring the "it works on my machine" promise across any environment.

## Tech Stack

* **Machine Learning:** Python, XGBoost, Scikit-Learn, Pandas

* **API Framework:** FastAPI, Uvicorn, Pydantic

* **DevOps / Deployment:** Docker, Git

## Project Structure

```text
ev-charging-project/
├── data/                       # Raw and processed CSV/Tensor data (Git-ignored)
├── notebooks/                  # Jupyter notebooks for EDA, data cleaning, and model training
├── scripts/                    # Production microservice files
│   ├── Dockerfile              # Docker image blueprint
│   ├── requirements.txt        # API dependencies
│   ├── main.py                 # FastAPI application
│   └── ev_battery_model.json   # Pre-trained XGBoost model
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## How to Run (Docker)

You don't need to install Python, XGBoost, or any dependencies on your local machine. You only need Docker!

**1. Clone the repository:**

```bash
git clone [https://github.com/YourUsername/ev-battery-safety-api.git](https://github.com/YourUsername/ev-battery-safety-api.git)
cd ev-battery-safety-api/scripts
```

**2. Build the Docker Image:**

```bash
docker build -t ev-battery-ai .
```

**3. Run the Container:**

```bash
docker run -p 8000:8000 ev-battery-ai
```

## Testing the API

Once the container is running, open your web browser and navigate to the interactive Swagger UI:

http://localhost:8000/docs

### Example Request (`POST /predict`)

Feed the API real-time vehicle telemetry:

```json
{
  "mileage": 150000,
  "avg_cell_voltage": 3.2,
  "max_cell_voltage": 3.9,
  "min_cell_voltage": 2.1,
  "avg_current": -350,
  "avg_temp": 45,
  "max_temp": 85,
  "voltage_gap": 1.8
}
```

### Example Response

The AI will instantly return a health assessment and fault probability:

```json
{
  "status": "success",
  "prediction_label": 1,
  "fault_probability_percent": 96.64,
  "assessment": "FAULT DETECTED"
}
```

## Model Training Details

The core of this API is an `XGBClassifier` trained on historical EV data. Through exploratory data analysis (EDA) using heatmaps, scatter plots, and distributions, key danger zones (such as high max temperatures combined with negative current draw) were identified and learned by the model, resulting in an accuracy score of over 96%.
