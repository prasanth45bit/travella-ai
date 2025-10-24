from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from planner import TripPlanner

app = FastAPI()

# Allow CORS for local and production frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace '*' with specific frontend URL in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TripPlanner instance with MongoDB connection details
planner = TripPlanner(
    db_uri="mongodb+srv://travella:travella@prasanth.kxcqdtd.mongodb.net/travella?retryWrites=true&w=majority",
    db_name="travella"
)


@app.post("/api/generate_trip")
async def generate_plan(request: Request):
    data = await request.json()
    destination_id = data.get("destination_id")
    budget = data.get("budget", 0)
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    travelers = data.get("travelers", 1)
    preferences = data.get("preferences", [])

    if not destination_id or not start_date or not end_date:
        return {"error": "destination_id, start_date and end_date are required."}

    result = planner.generate_plan(
        destination_id=destination_id,
        budget=budget,
        start_date=start_date,
        end_date=end_date,
        travelers=travelers,
        preferences=preferences
    )
    return result


@app.get("/")
def read_root():
    return {"message": "Welcome to the Travel Itinerary Planner API!"}
