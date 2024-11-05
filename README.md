# Will It Rain Tomorrow? 🌧️

https://will-it-rain-tomorrow.streamlit.app/

## Overview
A machine learning model that predicts rainfall in Australia based on weather data. 
The model uses the ["Rain in Australia"](https://kaggle.com/jsphyg/weather-dataset-rattle-package) dataset from Kaggle, containing about 10 years of daily weather observations from various Australian weather stations.

## Goal
To build an automated system that can predict whether it will rain tomorrow at a specific location based on today's weather data.

## Project Components

### Data Preprocessing
- Handling missing values
- Splitting data into training, validation, and test sets
- Encoding categorical variables
- Scaling numerical features

### Model Development
- Logistic Regression model for binary classification
- Model evaluation using F1 score, ROC curve & AUROC and confusion matrix
- Saving trained model with preprocessing components

### Deployment
- Interactive web application using Streamlit
- Real-time predictions based on user input
- Visual representation of prediction results

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Streamlit
- Joblib

## Results
The project demonstrates a complete machine learning workflow from data preprocessing to model deployment, focusing on a practical weather prediction problem.