from fastapi import FastAPI
from inference import model_instance

app = FastAPI()

@app.get("/")
def home():
    return {"status": "IPO price prediction model running"}

@app.post("/predict")
def predict(data: dict):
    result = model_instance.predict(data)
    return {"prediction": result}
