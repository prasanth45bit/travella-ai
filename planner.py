# planner.py

from pymongo import MongoClient, ReturnDocument  # correct import
import pandas as pd
import random
from datetime import datetime, timedelta

class TripPlanner:
    def __init__(self, db_uri, db_name):
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.places = pd.DataFrame(list(self.db.places.find()))

    def generate_plan(self, destination, budget, start_date, end_date, travelers, preferences=None):
        subset = self.places[self.places["destination_name"].str.lower() == destination.lower()]
        if subset.empty:
            return {"error": "No data for this destination."}

        total_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days + 1
        plan = {}

        for i in range(total_days):
            current_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
            plan[current_date] = {}
            for slot in ["morning", "afternoon", "evening"]:
                slot_places = subset[subset["time_slot"] == slot]

                if preferences:
                    slot_places = slot_places[
                        slot_places["place_name"].str.lower().apply(
                            lambda x: any(p.lower() in x for p in preferences)
                        )
                    ]

                # fallback to full slot if filtered is empty
                if slot_places.empty:
                    slot_places = subset[subset["time_slot"] == slot]

                if not slot_places.empty:
                    act = slot_places.sample(1).iloc[0]["place_name"]
                    plan[current_date][slot] = act
                else:
                    plan[current_date][slot] = "Free time / Explore local areas"

        return {
            "destination": destination,
            "days": total_days,
            "budget": budget,
            "travelers": travelers,
            "plan": plan,
        }
