# Diabetes Prediction System

## Overview
This project is a Machine Learning-based system that predicts whether a person is diabetic or non-diabetic based on health-related indicators. The system is built using a structured dataset and deployed using FastAPI for real-time predictions.

## Problem Statement
Diabetes is a major health issue worldwide. Early prediction can help in prevention and timely treatment. This project aims to build a reliable predictive model using machine learning techniques.

## Dataset
The model is trained on the BRFSS 2015 Diabetes Health Indicators Dataset containing approximately 253,000 records and 21 health-related features.

Dataset Source:
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

## Features Used
- High Blood Pressure
- High Cholesterol
- BMI
- Smoking
- Stroke
- Heart Disease
- Physical Activity
- General Health
- Mental Health
- Physical Health
- Age
- Education
- Income
- And other lifestyle indicators

## Model Used
- XGBoost Classifier
- SMOTE used for handling class imbalance
- Hyperparameter tuning using RandomizedSearchCV

## API Deployment
The trained model is deployed using FastAPI for real-time prediction.

### Run API locally:
uvicorn app.main:app --reload

Then open:
http://127.0.0.1:8000/docs

## Project Structure

Diabetes Prediction System/
│
├── app/
├── model/
├── notebooks/
├── .gitignore
├── README.md
├── requirements.txt
├── data_dictionary.md

## Result
The model is optimized to improve recall for diabetic cases, making it suitable for screening purposes.

## Author
Abdul Wahab
