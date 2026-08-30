# 1. Start with a lightweight, official Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install the exact libraries directly (Bypassing requirements.txt to guarantee installation)
RUN pip install --no-cache-dir fastapi uvicorn xgboost pandas numpy pydantic scikit-learn

# 4. Copy your application code, the trained model, and its feature metadata
#    (model_features.json is produced by train_model.py -- rebuild the image
#    after retraining so this stays in sync with ev_battery_model.json)
COPY main.py .
COPY ev_battery_model.json .
COPY model_features.json .

# 5. Expose port 8000 so the outside world can talk to the container
EXPOSE 8000

# 6. The command to boot up the FastAPI server when the container turns on
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]