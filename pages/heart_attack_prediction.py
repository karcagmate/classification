from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
from TrainTestSplit import Split
from DataCleaning import DataCleaning

import numpy as np
st.set_page_config(page_title="Heart Attack Prediction",page_icon="❤️")
dc=DataCleaning()
def load_data():
 # data=dc.data
  dc.read_csv()
  dc.drop_columns()
  dc.split()
  dc.to_numeric('Systolic')
  dc.to_numeric('Diastolic')
  dc.label_encoding()
  return dc.data
tts=Split(load_data())

def split():
  X,y=tts.split_to_x_y()
  X_scaled,y_rescaled=tts.rescale(X,y)
  X_train,X_test,y_train,y_test=tts.split(X,y)
  return X_train,y_train,X_test,y_test


X_train,y_train,X_test,y_test=split()
model=BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,random_state=20)

def fit_model(X_train,y_train):
 
 model.fit(X_train,y_train)
def predict_risk(featrues):
    fit_model(X_train,y_train)
    x=featrues.values
   # x=np.array(x)
    #st.write( model.predict(x))
    prediction=model.predict_proba(x)[:,1]
    return prediction
    


def main():
    st.title('Heart Attack Risk Prediction')
    sex_mapping = {'Male': 1, 'Female': 0}
    diet_mapping={'Unhealthy':1,'Average':0,'Healthy':2}
    yes_no_mappping={'Yes':1,'No':0}
    age=st.slider('Age',18,100,25)
    sex=st.radio('Sex',['Male','Female'])
    cholesterol=st.slider('Cholesterol',150,400,200)
    heartrate=st.slider('Heart Rate',40,110,60)
    diabetes=st.radio('Diabetes',['Yes','No'])
    familyhistory=st.radio('Family History',['Yes','No'])
    smoking=st.radio('Smoking',['Yes','No'])
    obesity=st.radio('Obesity',['Yes','No'])
    alcoholconsumption=st.radio('Alcohol Consumption',['Yes','No'])
    exercisehoursperweek=st.slider('Exercise Hours Per Week',2,50,10)
    diet=st.radio('Diet',['Unhealthy','Average','Healthy'])
    previousheartproblems=st.radio('Previous Heart Problems',['Yes','No'])
    medicationuse=st.radio('Medication Use',['Yes','No'])
    stresslevel=st.slider('Stress Level',0,10,5)
    sedentaryhoursperday=st.slider('Sedentary Hours Per Day',0,18,10)
    bmi=st.slider('BMI',18,40,25)
    triglycerides=st.slider('Triglycerides',150,800,300)
    sleephoursperday=st.slider('Sleep Hours Per Day',3,12,8)
    systolic=st.slider('Systolic',100,180,120)
    diastolic=st.slider('Diastolic',60,110,80)

    user_input=pd.DataFrame({
        'Age':[age],
        "Gender":[sex_mapping[sex]],
        "Chlosterol":[cholesterol],
        "Heart Rate":[heartrate],
        "Diabetes":[yes_no_mappping[diabetes]],
        "Family History":[yes_no_mappping[familyhistory]],
        "Smoking":[yes_no_mappping[smoking]],
        "Obesity":[yes_no_mappping[obesity]],
        "Alcohol Consumption":[yes_no_mappping[alcoholconsumption]],
        "Exercise Hours Per Week":[exercisehoursperweek],
        "Diet":[diet_mapping[diet]],
        "Previous Heart Problems":[yes_no_mappping[previousheartproblems]],
        "Medication Use":[yes_no_mappping[medicationuse]],
        "Stress Level":[stresslevel],
        "Sedentary Hours Per Day":[sedentaryhoursperday],
        "BMI":[bmi],
        "Triglycerides":[triglycerides],
        "Sleep Hours Per Day":[sleephoursperday],
        "Systolic":[systolic],
        "Diastolic":[diastolic]
    })
    if st.button('Predict'):
        predicion=predict_risk(user_input)
        st.write(f'Predicted Heart attack risk:{predicion}')



if __name__ == '__main__':
    main()
