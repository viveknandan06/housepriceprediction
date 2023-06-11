import streamlit as st
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = load('HousePricePrediction.joblib')
pipeline = load('pipeline.joblib')
# label_encoder = load('label_encoder.joblib')

st.title("House Price Prediction")
id=st.slider("Id", 0, 3000)
mssubclass = st.slider("Enter the MSSubClass", 20, 190)
mszoning = st.selectbox("Select MSZoning", [1,2,3,4,5])
lotarea = st.slider("Enter the LotArea", 1300, 215000)
lotconfig = st.selectbox("Select LotConfig", [1,2,3,4,5 ])
bldgtype = st.selectbox("Select BldgType", [1,2,3,4,5])
overallcond = st.slider("Enter the OverallCond", 1, 10)
yearbuilt = st.slider("Enter the YearBuilt", 1800, 2010)
yearremodadd = st.slider("Enter the YearRemodAdd", 1800, 2010)
exterior1st = st.selectbox("Select Exterior1st", [1,2,3])
bsmtfinsf2 = st.slider("Enter the BsmtFinSF2", 0, 1400)
totalbsmtsf = st.slider("Enter the TotalBsmtSF", 0, 2500)

# Convert string values to numeric using LabelEncoder
# mszoning_numeric = label_encoder.transform([mszoning])[0]
# lotconfig_numeric = label_encoder.transform([lotconfig])[0]
# bldgtype_numeric = label_encoder.transform([bldgtype])[0]
# exterior1st_numeric = label_encoder.transform([exterior1st])[0]

input_data = {
    'Id': [id],
    'MSSubClass': [mssubclass],
    'MSZoning': [mszoning],
    'LotArea': [lotarea],
    'LotConfig': [lotconfig],
    'BldgType': [bldgtype],
    'OverallCond': [overallcond],
    'YearBuilt': [yearbuilt],
    'YearRemodAdd': [yearremodadd],
    'Exterior1st': [exterior1st],
    'BsmtFinSF2': [bsmtfinsf2],
    'TotalBsmtSF': [totalbsmtsf]
}

if st.button("Predict Price"):

    train_data = pd.read_csv('train.csv')
    pipeline.fit(train_data)
    
    prepared_input_data = pipeline.transform(pd.DataFrame(input_data))
    predicted_price = model.predict(prepared_input_data)
    st.success("The predicted price of the house is $%.2f" % predicted_price)
    
st.markdown("*Project by Vivek Nandan, Mayank Vatsa, Supreeti Ghosh and Muskan under guidance of Dr Parth Pratim Sarangi*")
