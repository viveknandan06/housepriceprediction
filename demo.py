import streamlit as st
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = load('HousePricePrediction.joblib')
pipeline = load('pipeline.joblib')
label_encoder = load('label_encoder.joblib')

st.title("House Price Prediction")
mssubclass = st.slider("Enter the MSSubClass", 20, 190)
mszoning = st.selectbox("Select MSZoning", label_encoder.classes_)
lotarea = st.slider("Enter the LotArea", 1300, 215000)
lotconfig = st.selectbox("Select LotConfig", label_encoder.classes_)
bldgtype = st.selectbox("Select BldgType", label_encoder.classes_)
overallcond = st.slider("Enter the OverallCond", 1, 10)
yearbuilt = st.slider("Enter the YearBuilt", 1800, 2010)
yearremodadd = st.slider("Enter the YearRemodAdd", 1800, 2010)
exterior1st = st.selectbox("Select Exterior1st", label_encoder.classes_)
bsmtfinsf2 = st.slider("Enter the BsmtFinSF2", 0, 1400)
totalbsmtsf = st.slider("Enter the TotalBsmtSF", 0, 2500)

try:
    mszoning_numeric = label_encoder.transform([mszoning])[0]
    lotconfig_numeric = label_encoder.transform([lotconfig])[0]
    bldgtype_numeric = label_encoder.transform([bldgtype])[0]
    exterior1st_numeric = label_encoder.transform([exterior1st])[0]
except ValueError as e:
    st.error(str(e))
    st.stop()

input_data = {
    'MSSubClass': [mssubclass],
    'MSZoning': [mszoning_numeric],
    'LotArea': [lotarea],
    'LotConfig': [lotconfig_numeric],
    'BldgType': [bldgtype_numeric],
    'OverallCond': [overallcond],
    'YearBuilt': [yearbuilt],
    'YearRemodAdd': [yearremodadd],
    'Exterior1st': [exterior1st_numeric],
    'BsmtFinSF2': [bsmtfinsf2],
    'TotalBsmtSF': [totalbsmtsf]
}

if st.button("Predict Price"):
    prepared_input_data = pipeline.transform(pd.DataFrame(input_data))
    predicted_price = model.predict(prepared_input_data)
    st.success("The predicted price of the house is $%.2f" % predicted_price)


