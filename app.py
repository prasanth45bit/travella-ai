from fastapi import FastAPI, Query
from pydantic import BaseModel
from planner import TripPlanner

app = FastAPI()
planner = TripPlanner(db_uri="mongodb+srv://travella:travella@prasanth.kxcqdtd.mongodb.net/travella?retryWrites=true&w=majority", db_name="travella")


class TripRequest(BaseModel):
    destination: str
    budget: float
    start_date: str
    end_date: str
    travelers: int
    preferences: list[str] = [] 


@app.post("/generate_trip")
def generate_trip(data: TripRequest):
    plan = planner.generate_plan(
        destination=data.destination,
        budget=data.budget,
        start_date=data.start_date,
        end_date=data.end_date,
        travelers=data.travelers,
        preferences=data.preferences
    )
    return plan



@app.get("/")
def read_root():
    return {"message": "Welcome to the Travel Itinerary Planner API!"}

