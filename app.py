import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load model
model = pk.load(open('model.pkl', 'rb'))

# Set page config
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add background style
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: #0d1b2a;
}
.css-1d391kg p, .css-1d391kg label {
    color: white !important;
    font-weight: 500;
}
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 12px;
    padding: 10px 24px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #dd2476, #ff512f);
}
.result-box {
    background: rgba(255,255,255,0.1);
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    text-align: center;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center; color:white;'>🚘 Car Price Prediction ML Model</h1>", unsafe_allow_html=True)

# Load data
cars_data = pd.read_csv('Cardetails.csv')

# Function
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Sidebar
st.sidebar.header("⚙️ Customize Inputs")

name = st.sidebar.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.sidebar.slider('Car manufactured year', 1994 , 2024)
km_driven = st.sidebar.slider('No of kms Driven', 11 , 200000)
fuel = st.sidebar.selectbox('Fuel type ', cars_data['fuel'].unique())
seller_type = st.sidebar.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.sidebar.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.sidebar.selectbox('Owner type', cars_data['owner'].unique())
Mileage = st.sidebar.slider(' Car Mileage', 10 , 40)
Engine = st.sidebar.slider('Engine Capacity', 700 , 5000)
Max_Power = st.sidebar.slider('Max Power', 0 , 200)
Seats = st.sidebar.slider(' No of Seats', 5 , 10)

# Prediction
if st.sidebar.button('🔮 Predict Price'):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, Mileage, Engine, Max_Power, Seats]],
        columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats']
    )

    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],
                          [1,2,3,4,5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4],inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3],inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'],[1,2],inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
                          ,inplace=True)

    Car_prize = model.predict(input_data_model)

    st.markdown(
        f"<div class='result-box'><h2>💰 Predicted Car Price: ₹ {Car_prize[0]:,.2f}</h2></div>",
        unsafe_allow_html=True
    )
