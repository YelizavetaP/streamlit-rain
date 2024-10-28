import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict(sepal_l, sepal_w, petal_l, petal_w):
    model = joblib.load('models/rf_model.joblib')
    data = np.expand_dims(np.array([sepal_l, sepal_w, petal_l, petal_w]), axis=0)
    predictions = model.predict(data)
    return predictions[0]

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

# Заголовок застосунку
st.title('Weather Prediction')
st.markdown("This model can use today's weather data for a specific location to predict whether it will rain in that location tomorrow")
# st.image('images/')

# # Відображення таблиці середніх значень
# st.header("Середні значення характеристик для кожного типу Ірису")
# iris_df = pd.read_csv("data/iris.csv")
# mean_values = iris_df.groupby('Species').mean().reset_index()
# st.dataframe(mean_values)

# Заголовок секції з характеристиками рослини
st.header("Характеристики рослини")
col1, col2 = st.columns(2)

# Введення характеристик чашолистків
with col1:
    st.text("Характеристики чашолистків (Sepal)")
    sepal_l = st.slider('Довжина чашолистка (см)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Ширина чашолистка (см)', 2.0, 4.4, 0.5)

# Введення характеристик пелюсток
with col2:
    st.text("Характеристики пелюсток (Petal)")
    petal_l = st.slider('Довжина пелюстки (см)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Ширина пелюстки (см)', 0.1, 2.5, 0.5)

# Кнопка для прогнозування
if st.button("Прогнозувати тип ірису"):
    # Викликаємо функцію прогнозування
    result = predict(sepal_l, sepal_w, petal_l, petal_w)
    st.write(f"Прогнозований тип ірису: {result}")
