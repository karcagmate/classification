import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class DataCleaning:
    def __init__(self) -> None:
        self.data = None
        self.path='heart_attack_prediction_dataset.csv'
        #read csv
    def read_csv(self):
        df=pd.read_csv(self.path)
        self.data=df.copy()
        return self .data
    
    #drop unnecessary columns
    def drop_columns(self):
        cols=['Patient ID','Country','Continent',
              'Hemisphere','Income','Physical Activity Days Per Week']
        self.data=self.data.drop(cols,axis=1)
        return self.data
    #splitting Blood pressure into systolicand diastolic
    def split(self):
     self.data['Systolic']=self.data['Blood Pressure'].apply(lambda x: x.split('/')[0])
     self.data['Diastolic']=self.data['Blood Pressure'].apply(lambda x: x.split('/')[1])
     #drop blood pressure
     self.data=self.data.drop('Blood Pressure',axis=1)
     return self.data
    
    #Data analysis
    def cols_analysis(self):
     #numeric cols
     data_numerics=self.data.select_dtypes(include=[np.number])
     numeric_cols=data_numerics.columns.values
     #non numeric cols
     data_nonnumeric=self.data.select_dtypes(exclude=[np.number])
     nonnumeric_cols=data_nonnumeric.columns.values
     message=f"\nNumeric Columns:\n {numeric_cols}  \n Non Numeric Columns:\n{nonnumeric_cols}"
     #print("Numeric Columns")
     #print(numeric_cols)
     #print("\nNon-Numeric Columns")
     #print(nonnumeric_cols)
     return message
    #handling missing values
    def missing_values(self):
     j=0
     values_list=list()
     cols_list=list()
     for col in self.data.columns:
      pct_missinng=np.mean(self.data[col].isnull())*100 #százalékosan
      cols_list.append(col)
      values_list.append(pct_missinng)
      if(pct_missinng==0): #ha nincs üres érték
         j+=1

     if(j!=len(cols_list)): #van oszlop amiben van üres érték
         missing_df=pd.DataFrame()
         missing_df['columns']=cols_list
         missing_df['missing']=values_list
         missing_df.loc[missing_df.missing>0].plot(kind='bar',figsize=(100,20))
         plt.title('Missing values')
         plt.show()
     else:          #nincs üres érték az adatbázisban
         print("No Missing Values")
    def searching_duplicates(self):
       message=""
       search=self.data.duplicated()
       duplicates=search[search==True]
       if len(duplicates)>0:
           message="Duplicate entries found"
           #print("Duplicate entries found")
       else:
          message="No duplicate entries found"
          #print("There are no duplicate entries")
       return message
        #convert into numeric values
    def to_numeric(self,column):
       self.data[column]=pd.to_numeric(self.data[column])
       #self.data.describe
    #Label object types columns
    def label_encoding(self):
       le=LabelEncoder()
       #data['Sex']=data['Sex'].replace({'Female':0,'Male':1})
       #data['Diet']=data['Diet'].replace({'Unhealthy':0,'Average':1,'Healthy':2})
       data_objects=self.data.select_dtypes(include=['object'])
       for columns in data_objects:
         self.data[columns]=le.fit_transform(self.data[columns])

       return self.data
    
   #searching for outlier vallues
    def detect_outliers(self,selected_columns):
     #IQR=Q3-Q1
     #Lower Bound = Q1 - 1.5 * IQR 
     #Upper Bound = Q3 + 1.5 * IQR  
     
        self.data[selected_columns].plot(kind='box',figsize=(12,8))
        #plt.show()
     #self.data.Age.plot(kind='box',figsize=(12,8))
     #plt.show()
     #self.data.Cholesterol.plot(kind='box',figsize=(12,8))
     #plt.show()
     #self.data.Income.plot(kind='box',figsize=(12,8))
     #plt.show()
     #self.data.BMI.plot(kind='box',figsize=(12,8))
     #plt.show()
     #self.data.Systolic.plot(kind='box',figsize=(12,8))
     #plt.show()
     #self.data.Diastolic.plot(kind='box',figsize=(12,8))
     #plt.show()
     #self.data.Triglycerides.plot(kind='box',figsize=(12,8))
     #plt.show()
    def summarise_cleaning(self):
       
       self.drop_columns()
       self.split()
       self.to_numeric('Systolic')
       self.to_numeric('Diastolic')
       self.label_encoding()
      
       

   

    

   
   




     
     




    



    
    
        

