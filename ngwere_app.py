import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model

# Load a pre-trained model

#loaded_model = load_model('ann_model.h5')

loaded_model = pickle.load(open('ngwere_model.sav', 'rb'))

# Define a function to classify a client
def classify_client(data):
    # Convert the data to an array and normalize it
    data = np.asarray(data)
    data = data / 1000.0
    # Expand the dimensions of the data to match the model input shape
    data = np.expand_dims(data, axis=0)
    # Predict the class of the client using the model
    prediction = loaded_model.predict(data)
    # Return the class with the highest probability
    return (prediction)

st.title('African banking crisis prediction App')

st.write('**The objective of the application is to classify whether a banking crisis is likely to occur or not in a given African country based on the values inputed. The objective is to identify early warning signs or indicators that can help anticipate and prevent financial crises**')

st.write('Developed by: *Craig (R204740C)*, *Takomborerwa*, *Kevin (R198132W)*, *Tafadzwa (R205761T)*, *Rudo (R205743Q)*, *Takudzwa*')

# Display inputs for features
st.header('Input Parameters')


# Sample DataFrame with years
data = {
    'Feature1': ['No']*300,  # Initialize as string column
    'Feature2': [0]*300,
    'Feature3': [0]*300,
    'Feature4': [0]*300,
    'Feature5': [0]*300,
    'Feature6': [0]*300,
    'Feature7': [0]*300,
    'Feature8': [0]*300,
    'Feature9': [0]*300,
    'Year': list(range(1800, 2100))
}
df = pd.DataFrame(data)

st.write('**The exchange rate of the country vis-a-vis the USD**')
exch_usd = st.number_input("*exch_usd*", min_value=0.0, max_value=None, step=0.1)
st.write('**The annual CPI Inflation rate**')
inflation_annual_cpi = st.number_input("*inflation_annual_cpi*", min_value=-100.0, max_value=None, step=0.1)
st.write('**A number which denotes a certain country**')
case = st.number_input('*case*', min_value=0, max_value=70, step=1)
st.write("**Sovereign domestic debt default (It is debt issued by the national government in a foreign currency in order to finance the issuing country's growth and development)**")
domestic_debt_in_default = st.selectbox("*domestic_debt_in_default*", ['Yes', 'No'])
st.write("**Sovereign External debt default (it is the portion of a country's debt that was borrowed from foreign lenders)**")
sovereign_external_debt_default = st.selectbox("*sovereign_external_debt_default*", ['Yes', 'No'])
st.write("**GDP default (The total debt in default vis-a-vis the GDP)**")
gdp_weighted_default = st.selectbox("*gdp_weighted_default*", ['Yes', 'No'])
st.write("**Inflation crisis (Inflation is a quantitative measure of the rate at which the average price level of a basket of selected goods and services in an economy increases over a period of time)**")
inflation_crisis = st.selectbox("*inflation_crisis*", ['Yes', 'No'])
st.write("**Currency crisis (It is the devaluation in a nation's currency matched by volatile markets and a lack of faith in the nation's economy)**")
currency_crisis = st.selectbox("*currency_crisis*", ['Yes', 'No'])
st.write("**Independence**")
independence = st.selectbox("*independence*", ['Yes', 'No'])
st.write("**Systemic crisis occured in the year**")
systemic_crisis = st.selectbox("systemic_crisis", ['Yes', 'No'])

year = st.selectbox('Select Year', df['Year'])

# Data transformation
# Map binary inputs to 1s and 0s
binary_mapping = {'Yes': 1, 'No': 0}
domestic_debt_in_default = binary_mapping[domestic_debt_in_default]
sovereign_external_debt_default = binary_mapping[sovereign_external_debt_default]
gdp_weighted_default = binary_mapping[gdp_weighted_default]
inflation_crisis = binary_mapping[inflation_crisis]
currency_crisis = binary_mapping[currency_crisis]
independence = binary_mapping[independence]
systemic_crisis = binary_mapping[systemic_crisis]

# Prepare input data for prediction
input_data = pd.DataFrame({
    'exch_usd': [exch_usd],
    'inflation_crisis': [inflation_crisis],
    'domestic_debt_in_default': [domestic_debt_in_default],
    'sovereign_external_debt_default': [sovereign_external_debt_default],
    'gdp_weighted_default' : [gdp_weighted_default],
    'inflation_crisis' : [inflation_crisis],
    'currency_crisis' : [currency_crisis],
    'independence' : [independence],
    'systemic_crisis' : [systemic_crisis],
    'Year' : [year],
    'case' : [case] 
})

# Assuming 'input_data' contains both numerical and categorical columns
# Convert categorical columns to numeric
input_data['domestic_debt_in_default'] = input_data['domestic_debt_in_default'].map(binary_mapping)
input_data['sovereign_external_debt_default'] = input_data['sovereign_external_debt_default'].map(binary_mapping)
input_data['gdp_weighted_default'] = input_data['gdp_weighted_default'].map(binary_mapping)
input_data['inflation_crisis'] = input_data['inflation_crisis'].map(binary_mapping)
input_data['currency_crisis'] = input_data['currency_crisis'].map(binary_mapping)
input_data['independence'] = input_data['independence'].map(binary_mapping)
input_data['systemic_crisis'] = input_data['systemic_crisis'].map(binary_mapping)

# Ensure 'Year' is numeric
input_data['Year'] = input_data['Year'].astype(int)
# Create a button to submit the input data
submit = st.button("Predict")

st.subheader("Prediction Results")
# If the button is clicked, classify the client and display the result
if submit:
    # Collect the input data into a list
    data = [case, year, systemic_crisis, exch_usd, domestic_debt_in_default, sovereign_external_debt_default, gdp_weighted_default, inflation_annual_cpi, independence, currency_crisis, inflation_crisis]
    # Classify the client and display the result
    result = classify_client(data)
    if result <= 0.5:
        st.write("The country is likely to have a banking crisis")
    else:
        st.write("The country is likely not to have a banking crisis")
