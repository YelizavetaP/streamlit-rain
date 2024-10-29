import streamlit as st
import pandas as pd
import numpy as np
import joblib


data = pd.read_csv('data/weatherAUS.csv')
features = data.columns[1:-1]


def predict(Location, MinTemp,MaxTemp, Rainfall, Evaporation, 
            Sunshine, WindGustDir, WindGustSpeed, WindDir9am,
            WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am,
            Humidity3pm, Pressure9am, Pressure3pm, Cloud9am,
            Cloud3pm, Temp9am, Temp3pm, RainToday):

    single_input = [Location, MinTemp,MaxTemp, Rainfall, Evaporation, 
                        Sunshine, WindGustDir, WindGustSpeed, WindDir9am,
                        WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am,
                        Humidity3pm, Pressure9am, Pressure3pm, Cloud9am,
                        Cloud3pm, Temp9am, Temp3pm, RainToday]

    input_data = pd.DataFrame([single_input], columns=features)

    model_info = joblib.load('models/aussie_rain.joblib')

    model = model_info['model']
    imputer = model_info['imputer']
    scaler = model_info['scaler']
    encoder = model_info['encoder']
    input_cols = model_info['input_cols']
    target_col = model_info['target_col']
    numeric_cols = model_info['numeric_cols']
    categorical_cols = model_info['categorical_cols']
    encoded_cols = model_info['encoded_cols']

    input_data[numeric_cols] = imputer.transform(input_data[numeric_cols])
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    input_data[encoded_cols] = encoder.transform(input_data[categorical_cols])

    X_input = input_data[numeric_cols + encoded_cols]

    pred = model.predict(X_input)
    prob = model.predict_proba(X_input)
    return [pred[0], prob[0][1]]

# Заголовок застосунку
st.title('Weather Prediction')
st.markdown("This model can use today's weather data for a specific location to predict whether it will rain in that location tomorrow")
st.image('images/rain1.jpg', use_column_width=True)

st.header("Today's indicators")


# Location
Location = st.selectbox('Location', data.Location.dropna().unique())
st.divider()


# MinTemp, MaxTemp
st.text("Temperature in degrees celsius")
MinTempСol, MaxTempCol= st.columns(2)

with MinTempСol:
    MinTemp = st.number_input('MinTemp', -15.0, 50.0, value=data['MinTemp'].mean())

with MaxTempCol:
    MaxTemp = st.number_input('MaxTemp', -15.0, 50.0, value=data['MaxTemp'].mean())

st.divider()


# Rainfall, Evaporation, Sunshine
st.text("The amount of rainfall recorded for the day in mm")
Rainfall = st.slider('Rainfall', data['Rainfall'].min(), data['Rainfall'].max(), data['Rainfall'].mean())

st.text("The so-called Class A pan evaporation (mm) in the 24 hours to 9am")
Evaporation = st.slider('Evaporation', data['Evaporation'].min(), data['Evaporation'].max(), data['Evaporation'].mean())

st.text("The number of hours of bright sunshine in the day.")
Sunshine = st.slider('Sunshine', data['Sunshine'].min(), data['Sunshine'].max(), data['Sunshine'].mean())
st.divider()


#WindGustDir, WindGustSpeed
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div style='word-wrap: break-word;'>The direction of the strongest wind gust in the 24 hours to midnight</div>", unsafe_allow_html=True)
    WindGustDir = st.selectbox('WindGustDir', data.WindGustDir.dropna().unique())

with col2:
    st.markdown("<div style='word-wrap: break-word;'>The speed (km/h) of the strongest wind gust in the 24 hours to midnight</div>", unsafe_allow_html=True)
    WindGustSpeed = st.slider('WindGustSpeed', data.WindGustSpeed.min(),  data.WindGustSpeed.max(), data.WindGustSpeed.mean())

st.divider()


# 9am, 3pm cols
col9am, col3pm = st.columns(2)
with col9am:
    
    WindDir9am = st.selectbox('Wind Direction at 9am', data.WindDir9am.dropna().unique())
    WindSpeed9am = st.slider('Wind Speed at 9am', data.WindSpeed9am.min(),  data.WindSpeed9am.max(), data.WindSpeed9am.mean())
    Humidity9am = st.slider('Humidity at 9am', data.Humidity9am.min(),  data.Humidity9am.max(), data.Humidity9am.mean())
    Pressure9am = st.slider('Pressure at 9am', data.Pressure9am.min(),  data.Pressure9am.max(), data.Pressure9am.mean())
    Cloud9am = st.slider('Cloud at 9am', data.Cloud9am.min(),  data.Cloud9am.max(), data.Cloud9am.mean())
    Temp9am = st.number_input('Temperature 9am', -15.0, 50.0, value=data['Temp9am'].mean())

with col3pm:
    
    WindDir3pm = st.selectbox('Wind Direction at 3pm', data.WindDir3pm.dropna().unique())
    WindSpeed3pm = st.slider('WindSpeed3pmst', data.WindSpeed3pm.min(),  data.WindSpeed3pm.max(), data.WindSpeed3pm.mean())
    Humidity3pm = st.slider('Humidity at 3pm', data.Humidity3pm.min(),  data.Humidity3pm.max(), data.Humidity3pm.mean())
    Pressure3pm = st.slider('Pressure at 3pm', data.Pressure3pm.min(),  data.Pressure3pm.max(), data.Pressure3pm.mean())
    Cloud3pm = st.slider('Cloud at 3pm', data.Cloud3pm.min(),  data.Cloud3pm.max(), data.Cloud3pm.mean())
    Temp3pm = st.number_input('Temperature 3pm', -15.0, 50.0, value=data['Temp3pm'].mean())

st.divider()


# RainToday
st.text("Is it raining today?")
RainToday = st.radio('', ['Yes', 'No'], index=0)
st.divider()


# Кнопка для прогнозування
if st.button("Predict"):
    # Викликаємо функцію прогнозування
    result = predict(Location, MinTemp,MaxTemp, Rainfall, Evaporation, 
                        Sunshine, WindGustDir, WindGustSpeed, WindDir9am,
                        WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am,
                        Humidity3pm, Pressure9am, Pressure3pm, Cloud9am,
                        Cloud3pm, Temp9am, Temp3pm, RainToday)
    # Виводимо результат
    if result[0] =='No': result[0] += ':dark_sunglasses::sunny:' 
    else: result[0] += ':umbrella::rain_cloud:'

    st.write(f"Will it rain tomorrow?:  {result[0]}")
    st.write(f"Probability of rain:  {result[1]}")

