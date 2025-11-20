from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define input matching your Phase 2 Model 1 inputs
class TripRequest(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_distance: float
    hour: int

# Endpoint 1: The Fare Predictor (Wraps Model 1/4)
@app.post("/predict-fare")
def predict_fare(trip: TripRequest):
    # In Phase 3, this calls the Spark model. 
    # For Dry Run, we return a dummy value.
    return {
        "predicted_fare": 25.50,
        "model_version": "Spark_RF_v1",
        "weather_impact_applied": True # Proof we are thinking about weather
    }

# Endpoint 2: The Hotspot Finder (Wraps Model 5/KMeans)
@app.get("/get-hotspots")
def get_hotspots(hour: int):
    return {
        "hour": hour,
        "top_zones": [74, 42, 16], # Example LocationIDs
        "cluster_type": "Commuter Rush"
    }

# Run with: uvicorn main:app --reload