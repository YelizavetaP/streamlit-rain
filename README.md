# Will It Rain Tomorrow? 🌧️

https://will-it-rain-tomorrow.streamlit.app/

## Overview
A machine learning model that predicts rainfall in Australia based on weather data. 
The model uses the ["Rain in Australia"](https://kaggle.com/jsphyg/weather-dataset-rattle-package) dataset from Kaggle, containing about 10 years of daily weather observations from various Australian weather stations.

- **Variables**: The dataset includes various features such as:
  - **Location**: Categorical variable indicating the weather station.
  - **MinTemp, MaxTemp**: Numerical variables representing minimum and maximum temperatures.
  - **Rainfall**: Numerical variable for the amount of rainfall recorded.
  - **Evaporation, Sunshine**: Numerical variables for evaporation and sunshine hours.
  - **WindGustDir, WindGustSpeed**: Categorical and numerical variables for wind gust direction and speed.
  - **Humidity9am, Humidity3pm**: Numerical variables for humidity levels at 9 am and 3 pm.
  - **Pressure9am, Pressure3pm**: Numerical variables for atmospheric pressure.
  - **Cloud9am, Cloud3pm**: Numerical variables for cloud coverage.
  - **RainToday**: Categorical variable indicating if it rained today (Yes/No).
  - **RainTomorrow**: Target variable indicating if it will rain tomorrow (Yes/No).


## Goal
To build an automated system that can predict whether it will rain tomorrow at a specific location based on today's weather data.

## Approach to the Solution

### Data Preprocessing
- Handling missing values
- Splitting data into training, validation, and test sets
- Encoding categorical variables
- Scaling numerical features

### Model Development
- Various classification models were evaluated, including `Logistic Regression`, `Decision Tree Classifier`, `XGBoost`, and `LGBM`.
- Hyperparameter tuning was conducted using techniques like `Random Search` and `Hyperopt` to optimize model performance.
- Saving trained model with preprocessing components

### Evaluation Metrics
- The models were evaluated using `ROC AUC scores`, `confusion matrices`, and `F1 scores` to assess their predictive capabilities.


### Deployment
- Interactive ["web application using Streamlit"](https://will-it-rain-tomorrow.streamlit.app/)
- Real-time predictions based on user input

### Obtained Results and Conclusions

The evaluation of different classification models yielded the following insights:

1. **Logistic Regression**: Showed the lowest performance with ROC AUC scores of 0.61 (train) and 0.57 (val), indicating poor pattern capture.

2. **Decision Tree Classifier**: Achieved a perfect ROC AUC of 1.00 on the training set but dropped to 0.50 on validation, indicating severe overfitting.

3. **Decision Tree + RandomSearch**:Addressed overfitting issues but resulted in lower performance, yielding ROC AUC scores of 0.36 (train) and 0.28 (val).

4. **XGBoost and LGBM**: Both models demonstrated good performance with ROC AUC scores around 0.74 to 0.75 on validation, indicating effective generalization.

5. **XGBoost + Hyperopt and LGBM + Hyperopt**: Achieved excellent scores of 0.81 and 0.82, respectively, showing significant improvement from hyperparameter tuning.

6. **Overfitting Concerns**: High training ROC AUC scores (0.93) for Hyperopt models raised overfitting concerns, but validation scores suggested good generalization.

 
Using XGBoost and LGBM with Hyperopt leads to significant performance improvements, making them the preferred choices for this classification task.


The project demonstrates a complete machine learning workflow from data preprocessing to model deployment, focusing on a practical weather prediction problem.

