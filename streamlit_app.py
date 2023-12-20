import streamlit as st
import requests

st.title('Real Estate Price Prediction')

# User input form
property_type = st.selectbox('Property Type', ['Apartment', 'House', 'Condo'])
city = st.selectbox('City', ['City1', 'City2', 'City3'])
baths = st.number_input('Number of Baths', value=1.0)
bedrooms = st.number_input('Number of Bedrooms', value=1.0)
area_type = st.selectbox('Area Type', ['Urban', 'Suburban', 'Rural'])
area_size = st.number_input('Area Size', value=1000.0)

# Make a request to the Flask app
response = requests.post('http://your-flask-app-url/predict', json={
    'propertyType': property_type,
    'city': city,
    'baths': baths,
    'bedrooms': bedrooms,
    'areaType': area_type,
    'areaSize': area_size
})

# Display prediction
if 'prediction' in response.json():
    st.success(f'Predicted Price: {response.json()["prediction"]}')
elif 'error' in response.json():
    st.error(f'Error: {response.json()["error"]}')
