import pickle
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import requests
from datetime import datetime, timedelta
from meteostat import Point, Daily
from keras.models import load_model
import cv2
import streamlit as st

soil_types = ['Mary', 'Loamy', 'Peaty', 'Sandy', 'Red soil', 'Chalky', 'Clay', 'Silt']

def get_coordinates(district_name):
    geolocator = Nominatim(user_agent="your_app")
    location = geolocator.geocode(district_name)
    return location.latitude, location.longitude

def get_next_days_avg_temperature(District_Name):
    base_url = "https://api.openweathermap.org/data/2.5/onecall"
    lat,lon= get_coordinates(District_Name)
    days = 12
    params = {
        'lat': lat,
        'lon': lon,
        'exclude': 'current,minutely,hourly',  # Exclude unnecessary details
        'appid': '30d4741c779ba94c470ca1f63045390a',
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code == 200:
        daily_forecast = data['daily'][:days]
        temperatures = [day['temp']['day'] for day in daily_forecast]
        avg_temperature = sum(temperatures) / len(temperatures)
        return avg_temperature/12
    else:
        print(f"Error {response.status_code}: {data['message']}")
        return None
    
def get_next_days_avg_humidity(District_Name):
    base_url = "https://api.openweathermap.org/data/2.5/onecall"
    lat,lon= get_coordinates(District_Name)
    days = 12

    params = {
        'lat': lat,
        'lon': lon,
        'exclude': 'current,minutely,hourly',  # Exclude unnecessary details
        'appid': '30d4741c779ba94c470ca1f63045390a',
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code == 200:
        daily_forecast = data['daily'][:days]
        humidities = [day['humidity'] for day in daily_forecast]
        avg_humidity = sum(humidities) / len(humidities)
        return avg_humidity/12
    else:
        print(f"Error {response.status_code}: {data['message']}")
        return None

def get_avg_precipitation(District_Name):
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)
    lat,lon= get_coordinates(District_Name)
    location = Point(lat,lon)
    data = Daily(location, start, end)
    data = data.fetch()
    avg_prcp = data['prcp'].mean()
    return avg_prcp

def get_crop_season():
    date = datetime.now()
    month = date.month

    if 7 <= month <= 10:
        return "Kharif"
    elif 10 < month < 12 or 1 <= month <= 4:
        return "Rabi"
    elif 3 <= month <= 6:
        return "Zaid"
    else:
        return "No specific crop season"

# Load the saved model
loaded_fine_tuned_model = load_model('loaded_model.h5')

def preprocess_single_image(img, dimension=(128, 128)):
    #img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        #print(f"Failed to read image: {image_path}")
        return None# Return None for flattened data and target

    img_resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

    if img_resized.size == 0:
        #print(f"Empty image: {image_path}")
        return None # Return None for flattened data and target

    # Flatten and normalize the image
    flat_data = img_resized.flatten() / 255.0
    images = np.array(img_resized)

    return images


def Soil_type_recognion(img):
    preprocessed_data = preprocess_single_image(img)
    if preprocessed_data is not None:
        preprocessed_data = np.expand_dims(preprocessed_data, axis=0)  
        predictions = loaded_fine_tuned_model.predict(preprocessed_data)
        predicted_class = np.argmax(predictions)
        return soil_types[predicted_class]

@st.cache_data
def get_best_crop():
    # Crop types
    crops = ['Jowar', 'Wheat', 'Bajra']

# Initialize variables to store maximum yield and corresponding crop
    max_yield = float('-inf')
    best_crop = None

# Predict yield for each crop and find the one with the maximum yield
    for crop in crops:
        if crop=='Jowar':
            predicted_yield = predict_yield1(area, temperature, precipitation, humidity,soil_type, district, crop, season)
            st.write(f'\nJowar predicted yield: {predicted_yield*1000}')
        if crop=='Wheat':
            predicted_yield = predict_yield2(area, temperature, precipitation, humidity,soil_type, district, crop, season)
            st.write(f'\nWheat predicted yield: {predicted_yield*1000}')
        if crop=='Bajra':
            predicted_yield = predict_yield3(area, temperature, precipitation, humidity,soil_type, district, crop, season)
            st.write(f'\nBajra predicted yield: {predicted_yield*1000}')
        if predicted_yield > max_yield:
            max_yield = predicted_yield
            best_crop = crop
    
    return best_crop,max_yield

def predict_yield1(area, temperature, precipitation, humidity, soil_type, district, crop, season):
    model_path1 = 'new2_rf_model.pickle'
    model1 = pickle.load(open(model_path1, 'rb'))
    input_data = pd.DataFrame({
        'Area': [area],
        'Temperature': [temperature],
        'Precipitation': [precipitation],
        'Humidity': [humidity],
        'Soil_type': [soil_type],
        'District': [district],
        'Crop': [crop],
        'Season': [season]
    })
    categorical_cols = ['Soil_type', 'District', 'Crop', 'Season']
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)
    model_feature_names = model1.feature_names_in_
    input_data_aligned = input_data_encoded.reindex(columns=model_feature_names, fill_value=0)
    predicted_yield = model1.predict(input_data_aligned)
    return predicted_yield[0]
def predict_yield2(area, temperature, precipitation, humidity, soil_type, district, crop, season):
    model_path2 = 'new3_rf_model.pickle' 
    model2 = pickle.load(open(model_path2, 'rb'))

    input_data = pd.DataFrame({
        'Area': [area],
        'Temperature': [temperature],
        'Precipitation': [precipitation],
        'Humidity': [humidity],
        'Soil_type': [soil_type],
        'District': [district],
        'Crop': [crop],
        'Season': [season]
    })
    categorical_cols = ['Soil_type', 'District', 'Crop', 'Season']
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)
    model_feature_names = model2.feature_names_in_
    input_data_aligned = input_data_encoded.reindex(columns=model_feature_names, fill_value=0)
    predicted_yield = model2.predict(input_data_aligned)
    return predicted_yield[0]
def predict_yield3(area, temperature, precipitation, humidity, soil_type, district, crop, season):
    model_path3 = 'new4_rf_model.pickle' 
    model3 = pickle.load(open(model_path3, 'rb'))

    input_data = pd.DataFrame({
        'Area': [area],
        'Temperature': [temperature],
        'Precipitation': [precipitation],
        'Humidity': [humidity],
        'Soil_type': [soil_type],
        'District': [district],
        'Crop': [crop],
        'Season': [season]
    })
    categorical_cols = ['Soil_type', 'District', 'Crop', 'Season']
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)
    model_feature_names = model3.feature_names_in_
    input_data_aligned = input_data_encoded.reindex(columns=model_feature_names, fill_value=0)
    predicted_yield = model3.predict(input_data_aligned)
    return predicted_yield[0]


import streamlit as st
# Streamlit app

 
style = """
<style>
#MainMenu {font-size: 16px;}  /* Adjust font size here */
</style>
"""
st.markdown(style, unsafe_allow_html=True)


st.title("Crop Yield Prediction and Recommendation")
st.image("cover-crops2.png")

nav = st.sidebar.radio("Navigation",['Automated Input','Manual Input'],help='choose the mode of input')

if nav == 'Automated Input':
    area = st.number_input("Enter the Area:",value=0)
    district = st.text_input("Enter the District:",value= 'N/A')
    soil_type = 'null'
    st.markdown("### Upload Soil Image")
    soil = st.file_uploader("Upload an image of the soil", type=["jpg", "jpeg", "png"])
    
    if soil is not None:
    # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(soil.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        soil_type = Soil_type_recognion(img)
        #st.image(img, channels="BGR")
    st.write("soil type is:  "+ soil_type)
    temperature = get_next_days_avg_temperature(district)
    precipitation = get_avg_precipitation(district)
    humidity = get_next_days_avg_humidity(district)
    season = get_crop_season()


    # Call predict_yield function
    if st.button("Recommend Crop"):
        get_best_crop.clear()
        best_crop , max_yield = get_best_crop()
        st.success(f'\nCrop with the highest predicted yield: {best_crop} ({max_yield*1000})')

elif nav == 'Manual Input':
    area = st.number_input("Enter the Area: ",value=0)
    temperature = st.number_input("Enter the Temperature: ",value=0)
    precipitation = st.number_input("Enter the Precipitation: ",value=0)
    humidity = st.number_input("Enter the Humidity: ",value=0)
    soil_type = st.text_input("Enter the soil type: ",value='N/A')
    district = st.text_input("Enter the District: ",value='N/A')
    season = st.text_input("Enter the Season: ",value='N/A')

    if st.button("Recommend Crop"):
        get_best_crop.clear()
        best_crop , max_yield = get_best_crop()
        st.success(f'\nCrop with the highest predicted yield: {best_crop} ({max_yield*1000})')


