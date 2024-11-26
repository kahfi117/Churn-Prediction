import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model yang disimpan menggunakan Pickle
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

# Fungsi untuk prediksi probabilitas churn
def predict_probability(features):
    probabilities = model.predict_proba([features])
    return probabilities[0]  # Mengembalikan array [probabilitas_tidak_churn, probabilitas_churn]

# Halaman utama Streamlit
st.title("Prediksi Probabilitas Churn Pelanggan")

# Sidebar untuk input pengguna
st.sidebar.header("Masukkan Data Pelanggan")

# Input dari pengguna
credit_score = st.sidebar.number_input("Credit Score", min_value=0, max_value=1000, value=600)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.sidebar.number_input("Tenure (Years)", min_value=0, max_value=20, value=5)
balance = st.sidebar.number_input("Balance", min_value=0.0, max_value=1e6, value=50000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=0)
has_credit_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Geography (One-hot encoding)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
geography_france = 1 if geography == "France" else 0
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0

# Gender (One-hot encoding)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender_male = 1 if gender == "Male" else 0
gender_female = 1 if gender == "Female" else 0

# Mengkategorikan Balance ke dalam segmen (One-hot encoding)
balance_low_threshold = 1 / 3  # Quantile 1/3
balance_high_threshold = 2 / 3  # Quantile 2/3

if balance <= balance_low_threshold * 1e6:  # Ganti 1e6 dengan skala maksimum balance saat pelatihan
    balance_segment_rendah = 1
    balance_segment_sedang = 0
    balance_segment_tinggi = 0
elif balance <= balance_high_threshold * 1e6:
    balance_segment_rendah = 0
    balance_segment_sedang = 1
    balance_segment_tinggi = 0
else:
    balance_segment_rendah = 0
    balance_segment_sedang = 0
    balance_segment_tinggi = 1
    
# Menghitung TenureByAge dan CreditScoreGivenAge
tenure_by_age = tenure / age if age > 0 else 0
credit_score_given_age = credit_score / age if age > 0 else 0

# Memproses fitur menjadi input model
features = [
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    1 if has_credit_card == "Yes" else 0,
    1 if is_active_member == "Yes" else 0,
    estimated_salary,
    tenure_by_age,
    credit_score_given_age,
    geography_france,
    geography_germany,
    geography_spain,
    gender_female,
    gender_male,
    balance_segment_rendah,
    balance_segment_sedang,
    balance_segment_tinggi,
    1
]

# Tombol prediksi
if st.sidebar.button("Predict"):
    probabilities = predict_probability(features)
    prob_not_churn = probabilities[1]
    prob_churn = probabilities[0]

    # Tampilkan hasil prediksi
    st.write("### Hasil Prediksi")
    st.write(f"**Probabilitas Tidak Churn:** {prob_not_churn * 100:.2f}%")
    st.write(f"**Probabilitas Churn:** {prob_churn * 100:.2f}%")

    # Berikan feedback berdasarkan probabilitas churn
    if prob_churn > 0.5:
        st.error("Pelanggan kemungkinan besar akan churn!")
    else:
        st.success("Pelanggan kemungkinan besar tidak akan churn.")

    # Visualisasi pie chart untuk probabilitas
    labels = ['Tidak Churn', 'Churn']
    sizes = [prob_not_churn, prob_churn]
    colors = ['#4CAF50', '#F44336']  # Warna hijau untuk tidak churn, merah untuk churn
    explode = (0.1, 0)  # Sedikit menonjolkan churn

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Membuat pie chart berbentuk lingkaran
    st.pyplot(fig)
