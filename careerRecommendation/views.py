from django.shortcuts import render
import pandas as pd
import joblib
import numpy as np
import os
from django.conf import settings

# Load trained pipeline
MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_model", "career_recommender_model.pkl")
model = joblib.load(MODEL_PATH)

def index(request):
    return render(request, 'index.html')


def service(request):
    if request.method == 'POST':
        age = float(request.POST.get('age'))
        education = request.POST.get('education')
        skills = request.POST.get('skills')
        interests = request.POST.get('interests')

        # Combine text exactly like training:
        # Combined_Text = Skills + ";" + Interests
        combined_text = f"{skills};{interests}"

        # Make the dataframe EXACTLY like training
        input_data = pd.DataFrame([{
            "Combined_Text": combined_text,
            "Age": age,
            "Education": education
        }])

        # Predict probabilities
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        # Combine and sort by probability
        result = sorted(
            zip(classes, probabilities * 100),
            key=lambda x: x[1],
            reverse=True
        )

        top_results = result[:3]  # Top 3 careers

        return render(request, 'results.html', {
            'results': top_results,
            'skills': skills,
            'interests': interests
        })

    return render(request, 'service.html')
