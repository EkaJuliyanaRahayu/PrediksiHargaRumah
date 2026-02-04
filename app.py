import streamlit as st
import pickle
import numpy as np

st.title("🏡 Prediksi Harga Rumah (Boston Dataset)")
st.write("Masukkan nilai fitur berikut untuk memprediksi harga rumah:")

# load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# input form
CRIM    = st.number_input("CRIM (Crime Rate)", 0.0)
ZN      = st.number_input("ZN", 0.0)
INDUS   = st.number_input("INDUS", 0.0)
CHAS    = st.number_input("CHAS (0 atau 1)", 0)
NOX     = st.number_input("NOX", 0.0)
RM      = st.number_input("RM (Jumlah Kamar)", 0.0)
AGE     = st.number_input("AGE", 0.0)
DIS     = st.number_input("DIS", 0.0)
RAD     = st.number_input("RAD", 0)
TAX     = st.number_input("TAX", 0.0)
PTRATIO = st.number_input("PTRATIO", 0.0)
B       = st.number_input("B", 0.0)
LSTAT   = st.number_input("LSTAT", 0.0)

features = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]

if st.button("Prediksi Harga"):
    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]
    st.success(f"💰 Prediksi Harga Rumah: **{pred:.2f}** (ribu USD)")
