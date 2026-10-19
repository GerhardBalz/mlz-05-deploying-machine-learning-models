import pickle

with open("pipeline.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

print(pipeline)
print("Classes:", pipeline.classes_)

datapoint = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

probability = pipeline.predict_proba([datapoint])[0, 1]

print("Conversion probability:", probability)
print(f"Q3 answer: {probability:.3f}")