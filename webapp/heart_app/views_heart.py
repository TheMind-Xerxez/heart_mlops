from django.shortcuts import render
import joblib
import os
import json
import pandas as pd
import numpy as np
from .models import heart_disease
import psycopg2

# Create your views here.


def index(request):
    return render(request, "heart_disease.html")


def heart_result(request):
    model = joblib.load("../models/model_heart.joblib")
    list = []
    list.append(float(request.GET["age"]))
    list.append(float(request.GET["sex"]))
    list.append(float(request.GET["cp"]))
    list.append(float(request.GET["trestbps"]))
    list.append(float(request.GET["chol"]))
    list.append(float(request.GET["fbs"]))
    list.append(float(request.GET["restecg"]))
    list.append(float(request.GET["thalach"]))
    list.append(float(request.GET["exang"]))
    list.append(float(request.GET["oldpeak"]))
    list.append(float(request.GET["slope"]))
    list.append(float(request.GET["ca"]))
    list.append(float(request.GET["thal"]))

    answer = model.predict([list]).tolist()[0]

    b = heart_disease(
        age=request.GET["age"],
        sex=request.GET["sex"],
        cp=request.GET["cp"],
        trestbps=request.GET["trestbps"],
        chol=request.GET["chol"],
        fbs=request.GET["fbs"],
        restecg=request.GET["restecg"],
        thalach=request.GET["thalach"],
        exang=request.GET["exang"],
        oldpeak=request.GET["oldpeak"],
        slope=request.GET["slope"],
        ca=request.GET["ca"],
        thal=request.GET["thal"],
        target=answer,
    )
    b.save()

    return render(request, "heart_disease.html", {"answer": answer})
