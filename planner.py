from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta
from bson import ObjectId


class TripPlanner:
    def __init__(self, db_uri, db_name):
        """Initialize TripPlanner with MongoDB connection."""
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        # Load places data once into a DataFrame for efficient filtering
        self.places = pd.DataFrame(list(self.db.places.find()))

    def generate_plan(self, destination_id, budget, start_date, end_date, travelers, preferences=None):
        """
        Generate a travel plan based on destination_id, dates, preferences, and budget.
        
        Args:
            destination_id (str): MongoDB ObjectId of the destination
            budget (int): Total budget for the trip
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            travelers (int): Number of travelers
            preferences (list): List of preference strings (optional)
            
        Returns:
            dict: Trip plan with daily itinerary and cost estimate
        """
        

        # Filter places matching the destination_id
        subset = self.places[self.places["destination"] == destination_id].copy()
        
        if subset.empty:
            return {"error": "No places found for this destination."}

        # Calculate total travel days
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            total_days = (end - start).days + 1
            
            if total_days <= 0:
                return {"error": "End date must be after start date."}
        except ValueError as e:
            return {"error": f"Invalid date format: {str(e)}"}

        plan = []
        total_cost = 0
        used_places = set()  # Track used places to avoid repetition

        # Generate plan for each day
        for i in range(total_days):
            current_date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            day_plan = {"date": current_date, "places": []}

            # Process each time slot (morning, afternoon, evening)
            for slot in ["morning", "afternoon", "evening"]:
                # Filter places for current time slot
                slot_places = subset[subset["time_slot"] == slot].copy()

                # Apply preference filtering if preferences are provided
                if preferences and len(preferences) > 0:
                    preferences_lower = [p.lower().strip() for p in preferences]
                    
                    # Filter places that match any preference
                    mask = slot_places["place_name"].str.lower().apply(
                        lambda x: any(pref in x for pref in preferences_lower)
                    )
                    preference_filtered = slot_places[mask]
                    
                    # Use preference-filtered places if available
                    if not preference_filtered.empty:
                        slot_places = preference_filtered

                # Exclude already used places (for variety)
                available_places = slot_places[~slot_places["_id"].isin(used_places)]
                
                # If all places have been used, reset the pool
                if available_places.empty:
                    available_places = slot_places

                # Select a place for this time slot
                if not available_places.empty:
                    # Prioritize places within budget per person per activity
                    budget_per_activity = budget / (total_days * 3) / travelers if budget > 0 else float('inf')
                    
                    affordable_places = available_places[
                        available_places["price"] <= budget_per_activity * 1.5
                    ]
                    
                    # If no affordable places, use all available
                    if affordable_places.empty:
                        affordable_places = available_places
                    
                    # Randomly select one place
                    chosen_place = affordable_places.sample(1).iloc[0]
                    place_cost = int(chosen_place["price"])
                    total_cost += place_cost
                    
                    # Mark as used
                    used_places.add(chosen_place["_id"])
                    
                    day_plan["places"].append({
                        "placeId": str(chosen_place["_id"]),
                        "place": chosen_place["place_name"],
                        "price": place_cost,
                        "timeSlot": slot
                    })
                else:
                    # No place available for this slot
                    day_plan["places"].append({
                        "placeId": None,
                        "place": "Free time / Explore local areas",
                        "price": 0,
                        "timeSlot": slot
                    })

            plan.append(day_plan)

        # Calculate budget status
        budget_status = "within_budget" if total_cost <= budget else "over_budget"
        budget_difference = budget - total_cost

        # Return the assembled trip plan
        return {
            "destination_id": destination_id,
            "startDate": start_date,
            "endDate": end_date,
            "travelers": travelers,
            "requestedBudget": int(budget),
            "estimatedCost": int(total_cost),
            "budgetStatus": budget_status,
            "budgetDifference": int(budget_difference),
            "plan": plan,
            "totalDays": total_days
        }

    def get_destination_info(self, destination_id):
        """Get destination details by ID."""
        try:
            dest_obj_id = ObjectId(destination_id)
            destination = self.db.destinations.find_one({"_id": dest_obj_id})
            
            if destination:
                destination["_id"] = str(destination["_id"])
                return destination
            return {"error": "Destination not found"}
        except Exception as e:
            return {"error": f"Invalid destination ID: {str(e)}"}

    def close(self):
        """Close MongoDB connection."""
        self.client.close()