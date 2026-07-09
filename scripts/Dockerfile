# 1. Start with a lightweight, official Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install the exact libraries directly (Bypassing requirements.txt to guarantee installation)
RUN pip install --no-cache-dir fastapi uvicorn xgboost pandas pydantic scikit-learn

# 4. Copy your application code and the saved AI model into the container
COPY main.py .
COPY ev_battery_model.json .

# 5. Expose port 8000 so the outside world can talk to the container
EXPOSE 8000

# 6. The command to boot up the FastAPI server when the container turns on
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]