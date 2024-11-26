# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 

#model XGB
import pickle
import numpy as np
import pandas as pd


def xgbModel(data):
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    predict = model.predict([data])
    
    return predict

def boolToInteger(data)->int: 
    value = 0 if data == False else 1
    return value

def nationalEncode(data):
    # One-hot encoding manual untuk Geography (Nationality)
    input_data = {}
    if data == "France":
        input_data["Geography_France"] = 1
        input_data["Geography_Germany"] = 0
        input_data["Geography_Spain"] = 0
    elif data == "Germany":
        input_data["Geography_France"] = 0
        input_data["Geography_Germany"] = 1
        input_data["Geography_Spain"] = 0
    else:
        input_data["Geography_France"] = 0
        input_data["Geography_Germany"] = 0
        input_data["Geography_Spain"] = 1
        
    
def main():
    
    st.title("Churn Prediction")
    
    with st.form("Form"): 
        row1 = st.columns([2,2])
        creditScore = row1[0].number_input("Credit Score",  step=1, value=1)
        age = row1[1].number_input("Age", step=1, value=1)
        tenure =  row1[0].number_input("tenure")
        balance = row1[1].number_input("balance")
        numberOfProducts =  row1[0].number_input("Number Of Products")
        estimatedSalary =  row1[1].number_input("Estimated Salary")
        hasCrCard = row1[0].radio("hasCrCard", options=['No','Yes'], index=1, horizontal=True)
        isActiveMember = row1[1].radio("isActiveMember", options=['No','Yes'], index=1, horizontal=True)
        gender = row1[0].selectbox('Gender', options=['Male', 'Female'])
        nationality = row1[1].selectbox('Nationality', options=['Germany', 'Spain', 'France'])
        submit = st.form_submit_button('check for churn')
    
    if submit: 
        with open("xgb_model.pkl", "rb") as file:
            encoder = pickle.load(file)
       # Transformasi ke format encoding
        originalFeatures = [[gender, nationality]]

        # Buat dictionary input_data berdasarkan kolom hasil encoding
        input_data = {}
        for col in encoder:  # encoded_columns diambil dari encoder.pkl
            if col.startswith("Geography_") and col.endswith(nationality):
                input_data[col] = 1
            elif col.startswith("Gender_") and col.endswith(gender):
                input_data[col] = 1
            else:
                input_data[col] = 0

        # Tambahkan fitur numerik ke dictionary input_data
        input_data["CreditScore"] = creditScore
        input_data["Age"] = age
        input_data["Tenure"] = tenure
        input_data["Balance"] = balance
        input_data["NumberOfProducts"] = numberOfProducts
        input_data["HasCrCard"] = hasCrCard
        input_data["IsActiveMember"] = isActiveMember
        input_data["EstimatedSalary"] = estimatedSalary
        
        finalFeatures = pd.DataFrame([input_data])
        return finalFeatures
        # data = [creditScore, 
        #     age, 
        #     tenure, 
        #     balance, 
        #     numberOfProducts, 
        #     boolToInteger(hasCrCard),
        #     boolToInteger(isActiveMember),
        #     estimatedSalary,
        #     boolToInteger(gender),
        #     finalFeatures
        #     ]
        # return st.json(data)
    


if __name__ == '__main__':
	main()