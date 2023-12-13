from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
from TrainTestSplit import Split
from DataCleaning import DataCleaning
from sklearn.preprocessing import LabelEncoder
import numpy as np
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
@st.cache_data
def split():
  X,y=tts.split_to_x_y()
  X_scaled,y_rescaled=tts.rescale(X,y)
  X_train,X_test,y_train,y_test=tts.split(X_scaled,y_rescaled)
  return X_train,y_train,X_test,y_test
X_train,y_train,X_test,y_test=split()
model=BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,random_state=20)
@st.cache_resource
def fit_model(X_train,y_train):
 st.write(X_train)

 model.fit(X_train,y_train)
def predict_risk(featrues):
    fit_model(X_train,y_train)
    x=list(featrues.values)
    x=np.array(x)
    st.write(x)
    st.write(x.shape)
    model.predict(x)
    prediction=model.predict_proba(featrues)[:,1]
    return prediction

def main():
    st.title('Heart Attack Risk Prediction App')
    sex_mapping = {'Male': 1, 'Female': 0}
    diet_mapping={'Unhealthy':0,'Average':1,'Healthy':2}
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
    sedentaryhoursperday=st.slider('Sedentary Hours Per Day',0,40,10)
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
    if st.sidebar.button('Predict'):
        
       # data_objects=user_input.select_dtypes(include=['object'])
       # for columns in data_objects:
        #   if columns=='Diet':
         #     if user_input[columns]=='Unhealthy':
          #       user_input[columns]=0
           #   elif   user_input[columns]=='Average':
            #     user_input[columns]=1
             # else:
              #   user_input[columns]=2
              #data['Sex']=data['Sex'].replace({'Female':0,'Male':1})
              #user_input[columns]=user_input[columns].replace({'Unhealthy':0,'Average':1,'Healthy':2})
          # elif columns=='Sex':
           #   if user_input[columns]=='Female':
            #     user_input[columns]=0
             # else:
              #   user_input[columns]=1
              #user_input[columns]=user_input[columns].replace({'Male':1,'Female':0})
          # else:
           #   if user_input[columns]=='No':
              #   user_input[columns]=0
            #  else:
             #    user_input[columns]=1
              #user_input[columns]=user_input[columns].replace
               #({'Yes':1,'No':0})
         
               
           #user_input[columns]=le.fit_transform(user_input[columns])
       # st.write(user_input.dtypes)
       # st.write(user_input.describe())
       # fit_model()
       
        st.write(user_input.dtypes)
        st.write(user_input)
       

        predicion=predict_risk(user_input)
        st.write(f'Predicted Heart attack risk:{predicion}')



if __name__ == '__main__':
    main()
