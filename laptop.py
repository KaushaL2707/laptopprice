from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import streamlit as st


df = pd.read_csv('modified_laptop_data.csv')

X = df.drop(columns=['Price'])
y = np.log(df['Price'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)
#Random Forest
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

# Streamlit app
st.title("Laptop Price Prediction")

# User inputs
company = st.selectbox("Company", df['Company'].unique())
type_name = st.selectbox("Type", df['TypeName'].unique())
ram = st.selectbox("Ram (GB)", sorted(df['Ram'].unique()))
weight = st.number_input("Weight (kg)", min_value=0.0, max_value=10.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS", ["No", "Yes"])
ppi = st.number_input("PPI", min_value=0.0, step=0.1)
cpu_brand = st.selectbox("CPU Brand", df['Cpu brand'].unique())
hdd = st.number_input("HDD (GB)", min_value=0, step=1)
ssd = st.number_input("SSD (GB)", min_value=0, step=1)
gpu_brand = st.selectbox("GPU Brand", df['Gpu brand'].unique())
os = st.selectbox("Operating System", df['os'].unique())

# Prepare input data
try:
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    input_data = pd.DataFrame([[company, type_name, ram, weight, touchscreen, ips, ppi, cpu_brand, hdd, ssd, gpu_brand, os]],
                              columns=X.columns)

    # Prediction
    if st.button("Predict Price"):
        prediction = np.exp(pipe.predict(input_data))
        st.success(f"The predicted price of the laptop is ₹{prediction[0]:.2f}")
except Exception as e:
    st.error(f"An error occurred: {e}")
