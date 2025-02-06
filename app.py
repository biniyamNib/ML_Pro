from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib  # Load your trained model
import numpy as np

# Load the trained model (Ensure the model file exists in your project directory)
MODEL_PATH = "rain_prediction_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI()

# Define request body format
class WeatherData(BaseModel):
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustSpeed: float
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    Temp9am: float
    Temp3pm: float
    WindGustDir: str
    WindDir9am: str
    WindDir3pm: str
    RainToday: str

# Encode categorical values manually (You should match this with how your model was trained)
wind_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
rain_today_map = {"No": 0, "Yes": 1}

@app.post("/predict")
async def predict_rain(data: WeatherData):
    try:
        # Encode categorical variables
        wind_gust_dir_index = wind_directions.index(data.WindGustDir)
        wind_dir_9am_index = wind_directions.index(data.WindDir9am)
        wind_dir_3pm_index = wind_directions.index(data.WindDir3pm)
        rain_today_encoded = rain_today_map[data.RainToday]

        # Prepare input for the model
        features = np.array([
            data.MinTemp, data.MaxTemp, data.Rainfall, data.Evaporation, data.Sunshine,
            data.WindGustSpeed, wind_gust_dir_index, data.WindSpeed9am, wind_dir_9am_index,
            data.WindSpeed3pm, wind_dir_3pm_index, data.Humidity9am, data.Humidity3pm,
            data.Pressure9am, data.Pressure3pm, data.Cloud9am, data.Cloud3pm, data.Temp9am,
            data.Temp3pm, rain_today_encoded
        ]).reshape(1, -1)

        # Make prediction
        prediction_prob = model.predict_proba(features)[0][1]  # Probability of rain
        prediction = "Yes" if prediction_prob > 0.5 else "No"

        return {"prediction": prediction, "probability": prediction_prob}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
