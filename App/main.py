from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI(title="Diabetes Prediction API")

# Load trained model
model = joblib.load("../model/final_model.pkl")


# Input schema with descriptions
class InputData(BaseModel):

    HighBP: int = Field(
        ...,
        description="0 = No high blood pressure, 1 = Has high blood pressure"
    )

    HighChol: int = Field(
        ...,
        description="0 = No high cholesterol, 1 = Has high cholesterol"
    )

    CholCheck: int = Field(
        ...,
        description="0 = No cholesterol check in last 5 years, 1 = Had cholesterol check"
    )

    BMI: float = Field(
        ...,
        description="Body Mass Index value (example: 27.5)"
    )

    Smoker: int = Field(
        ...,
        description="0 = Never smoked, 1 = Smoked at least 100 cigarettes in life"
    )

    Stroke: int = Field(
        ...,
        description="0 = No stroke history, 1 = Had a stroke"
    )

    HeartDiseaseorAttack: int = Field(
        ...,
        description="0 = No heart disease/attack, 1 = Has heart disease or attack history"
    )

    PhysActivity: int = Field(
        ...,
        description="0 = No physical activity, 1 = Performed physical activity"
    )

    Fruits: int = Field(
        ...,
        description="0 = Does not consume fruits regularly, 1 = Consumes fruits regularly"
    )

    Veggies: int = Field(
        ...,
        description="0 = Does not consume vegetables regularly, 1 = Consumes vegetables regularly"
    )

    HvyAlcoholConsump: int = Field(
        ...,
        description="0 = No heavy alcohol consumption, 1 = Heavy alcohol consumption"
    )

    AnyHealthcare: int = Field(
        ...,
        description="0 = No healthcare coverage, 1 = Has healthcare coverage"
    )

    NoDocbcCost: int = Field(
        ...,
        description="0 = Could afford doctor visit, 1 = Could not afford doctor visit"
    )

    GenHlth: int = Field(
        ...,
        description="General health rating: 1 = Excellent to 5 = Poor"
    )

    MentHlth: int = Field(
        ...,
        description="Number of days mental health was not good (0–30)"
    )

    PhysHlth: int = Field(
        ...,
        description="Number of days physical health was not good (0–30)"
    )

    DiffWalk: int = Field(
        ...,
        description="0 = No difficulty walking, 1 = Has difficulty walking"
    )

    Sex: int = Field(
        ...,
        description="0 = Female, 1 = Male"
    )

    Age: int = Field(
        ...,
        description="""Age categories:
    1 = 18-24
    2 = 25-29
    3 = 30-34
    4 = 35-39
    5 = 40-44
    6 = 45-49
    7 = 50-54
    8 = 55-59
    9 = 60-64
    10 = 65-69
    11 = 70-74
    12 = 75-79
    13 = 80+"""
    )
    

    Education: int = Field(
        ...,
        description="""Education level:
    1 = Never attended school or only kindergarten
    2 = Grades 1-8 (Elementary)
    3 = Grades 9-11 (Some high school)
    4 = High school graduate
    5 = College (1–3 years)
    6 = College graduate"""
    )

    Income: int = Field(
        ...,
        description="""Income category:
    1 = Less than $10,000
    2 = $10,000–$14,999
    3 = $15,000–$19,999
    4 = $20,000–$24,999
    5 = $25,000–$34,999
    6 = $35,000–$49,999
    7 = $50,000–$74,999
    8 = $75,000 or more"""
    )


# Home route
@app.get("/")
def home():
    return {
        "message": "Diabetes Prediction API is running",
        "usage": "Send POST request to /predict with all required health features"
    }


# Prediction route
@app.post("/predict")
def predict(data: InputData):

    # Convert input into numpy array
    input_array = np.array([[
        data.HighBP,
        data.HighChol,
        data.CholCheck,
        data.BMI,
        data.Smoker,
        data.Stroke,
        data.HeartDiseaseorAttack,
        data.PhysActivity,
        data.Fruits,
        data.Veggies,
        data.HvyAlcoholConsump,
        data.AnyHealthcare,
        data.NoDocbcCost,
        data.GenHlth,
        data.MentHlth,
        data.PhysHlth,
        data.DiffWalk,
        data.Sex,
        data.Age,
        data.Education,
        data.Income
    ]])

    # Prediction
    prediction = model.predict(input_array)[0]

    # Return result
    return {
        "prediction": int(prediction),
        "result": "Diabetic" if prediction == 1 else "Non-Diabetic"
    }