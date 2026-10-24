import pickle

from model import normalize_record


model_file = 'pipeline.bin'

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)



url = 'http://127.0.0.1:9696/predict'



lead_q3 = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}



record_q3 = normalize_record(lead_q3)
first_q3 = float(pipeline.predict_proba([record_q3])[0, 1])
second_q3 = float(pipeline.predict_proba([record_q3])[0, 1])

assert 0 < first_q3 < 1
assert first_q3 == second_q3
print({"conversion_probability": round(first_q3, 3), "conversion": first_q3 >= 0.5})

lead_q4 = {
  "lead_source": "organic_search",
  "industry": "technology",
  "employment_status": "employed",
  "location": "europe",
  "number_of_courses_viewed": 4,
  "annual_income": 80304.0,
  "interaction_count": 7,
  "lead_score": 0.74
}

record_q4 = normalize_record(lead_q4)
first_q4 = float(pipeline.predict_proba([record_q4])[0, 1])
second_q4 = float(pipeline.predict_proba([record_q4])[0, 1])




## assert 0 < first_q4 < 1
## assert first_q4 == second_q4
## print({"conversion_probability": round(first_q4, 3), "conversion": first_q4 >= 0.5})
