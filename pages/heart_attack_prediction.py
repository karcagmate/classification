from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd

model=BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,random_state=20)

def predict_risk(featrues):
    prediction=model.predict([featrues])
    return prediction

def main():
    st.title('Heart Attack Risk Prediction App')
    age=st.sidebar('Age',18,100,25)
    sex=st.sidebar('Sex',['Male','Female'])
    cholesterol=st.sidebar('Cholesterol',150,400,200)
    heartrate=st.sidebar('Heart Rate',40,110,60)
    diabetes=st.sidebar('Diabetes',['Yes','No'])
    familyhistory=st.sidebar('Family History',['Yes','No'])
    smoking=st.sidebar('Smoking',['Yes','No'])
    obesity=st.sidebar('Obesity',['Yes','No'])
    alcoholconsumption=st.sidebar('Alcohol Consumption',['Yes','No'])
    exercisehoursperweek=st.sidebar('Exercise Hours Per Week',2,50,10)
    diet=st.sidebar('Diet',['Unhealthy','Average','Healthy'])
    previousheartproblems=st.sidebar('Previous Heart Problems',['Yes','No'])
    medicationuse=st.sidebar('Medication Use',['Yes','No'])
    stresslevel=st.sidebar('Stress Level',0,10,5)
    sedentaryhoursperday=st.sidebar('Sedentary Hours Per Day',0,40,10)
    bmi=st.sidebar('BMI',18,40,25)
    triglycerides=st.sidebar('Triglycerides',150,800,300)
    sleephoursperday=st.sidebar('Sleep Hours Per Day',3,12,8)
    systolic=st.sidebar('Systolic',100,180,120)
    diastolic=st.sidebar('Diastolic',60,110,80)

    user_input=pd.DataFrame({
        'Age':[age],
        "Gender":[sex],
        "Chlosterol":[cholesterol],
        "Heart Rate":[heartrate],
        "Diabetes":[diabetes],
        "Family History":[familyhistory],
        "Smoking":[smoking],
        "Obesity":[obesity],
        "Alcohol Consumption":[alcoholconsumption],
        "Exercise Hours Per Week":[exercisehoursperweek],
        "Diet":[diet],
        "Previous Heart Problems":[previousheartproblems],
        "Medication Use":[medicationuse],
        "Stress Level":[stresslevel],
        "Sedentary Hours Per Day":[sedentaryhoursperday],
        "BMI":[bmi],
        "Triglycerides":[triglycerides],
        "Sleep Hours Per Day":[sleephoursperday],
        "Systolic":[systolic],
        "Diastolic":[diastolic]

        


    })



if __name__ == '__main__':
    main()
