🔋 EV Battery Fault Predictor & Safety APIAn end-to-end machine learning microservice built to predict electric vehicle (EV) battery faults in real-time using vehicle telemetry data.This project takes raw battery sensor data, processes it through a highly accurate GPU-accelerated XGBoost model, and serves the predictions via a blazing-fast FastAPI web server, all fully containerized in Docker for production deployment.🚀 FeaturesReal-Time Inference: Millisecond response times for battery health assessments.XGBoost Machine Learning: Trained on massive PyTorch tensor/CSV datasets of EV telemetry to identify complex failure patterns.Interactive API Documentation: Built-in Swagger UI for easy testing and endpoint interaction.Production Ready: Fully containerized with Docker, ensuring the "it works on my machine" promise across any environment.🛠️ Tech StackMachine Learning: Python, XGBoost, Scikit-Learn, PandasAPI Framework: FastAPI, Uvicorn, PydanticDevOps / Deployment: Docker, Git📂 Project Structureev-charging-project/
├── data/                       # Raw and processed CSV/Tensor data (Git-ignored)
├── notebooks/                  # Jupyter notebooks for EDA, data cleaning, and model training
├── scripts/                    # Production microservice files
│   ├── Dockerfile              # Docker image blueprint
│   ├── requirements.txt        # API dependencies
│   ├── main.py                 # FastAPI application
│   └── ev_battery_model.json   # Pre-trained XGBoost model
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
🐳 How to Run (Docker)You don't need to install Python, XGBoost, or any dependencies on your local machine. You only need Docker!1. Clone the repository:git clone [https://github.com/YourUsername/ev-battery-safety-api.git](https://github.com/YourUsername/ev-battery-safety-api.git)
cd ev-battery-safety-api/scripts
2. Build the Docker Image:docker build -t ev-battery-ai .
3. Run the Container:docker run -p 8000:8000 ev-battery-ai
🧪 Testing the APIOnce the container is running, open your web browser and navigate to the interactive Swagger UI:👉 http://localhost:8000/docsExample Request (POST /predict)Feed the API real-time vehicle telemetry:{
  "mileage": 150000,
  "avg_cell_voltage": 3.2,
  "max_cell_voltage": 3.9,
  "min_cell_voltage": 2.1,
  "avg_current": -350,
  "avg_temp": 45,
  "max_temp": 85,
  "voltage_gap": 1.8
}
Example ResponseThe AI will instantly return a health assessment and fault probability:{
  "status": "success",
  "prediction_label": 1,
  "fault_probability_percent": 96.64,
  "assessment": "FAULT DETECTED"
}
🧠 Model Training DetailsThe core of this API is an XGBClassifier trained on historical EV data. Through exploratory data analysis (EDA) using heatmaps, scatter plots, and distributions, key danger zones (such as high max temperatures combined with negative current draw) were identified and learned by the model, resulting in an accuracy score of over 96%.