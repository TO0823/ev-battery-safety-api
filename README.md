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
