import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the model using joblib
model = joblib.load("predict_diabetes.pkl")

class ScoringItem(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)

    return {"prediction": int(yhat[0])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
 
